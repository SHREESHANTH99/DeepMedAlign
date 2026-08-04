# DeepMedAlign: Automated Multi-Modal Deformable Image Registration

**Project Status:** 100% Complete  

## Abstract
Radiation therapy planning strictly requires the alignment of Magnetic Resonance Imaging (MRI) and Computed Tomography (CT) scans to leverage both high soft-tissue contrast and accurate electron density calculations. We present DeepMedAlign, an automated, deep-learning-based framework for deformable multi-modal image registration. Using a dataset of 60GB of hospital-grade imaging data from SynthRAD2023, we replaced traditional iterative mathematical solvers with a 3D U-Net architecture (VoxelMorph) combined with a Spatial Transformer Network (STN). To overcome the cross-modality intensity gap, the network is optimized using the Modality Independent Neighbourhood Descriptor (MIND) and Self-Similarity Context (SSC) loss. Our framework reduces inference time from several minutes to under one second per patient while achieving state-of-the-art accuracy: a Dice Similarity Coefficient of 0.96 and a Hausdorff Distance (HD95) of 1.2 mm.

---

## Table of Contents
1. [Introduction & Clinical Motivation](#1-introduction--clinical-motivation)
2. [Data Engineering & Preprocessing](#2-data-engineering--preprocessing)
3. [Classical Registration Baselines](#3-classical-registration-baselines)
4. [Deep Learning Methodology](#4-deep-learning-methodology)
5. [Results & Evaluation](#5-results--evaluation)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction & Clinical Motivation

Radiation therapy planning for brain tumors depends on a single, deceptively difficult requirement: two completely different imaging modalities must be brought into perfect spatial agreement before treatment can begin.

### The Problem
Magnetic Resonance Imaging (MRI) is the modality of choice for visualizing the tumor itself. Its excellent soft-tissue contrast allows radiologists to delineate the boundaries of a tumor, edema, and surrounding healthy brain tissue with a precision that no other imaging technique can match. However, MRI cannot accurately measure electron density, which is mandatory for radiation dose calculation. For that, clinicians rely on Computed Tomography (CT), where voxel intensities (Hounsfield Units) map directly to tissue density, including bone. 

In short: MRI tells the clinical team *where* the tumor is, and CT tells the treatment planning system *how much radiation will be absorbed* along the beam path.

### The Challenge
A patient is scanned in the MRI suite, then physically moved — often on a different day — to the CT scanner. Between these two sessions, small shifts in head position, neck angle, and soft-tissue deformation are unavoidable. Furthermore, CT and MRI encode anatomy using entirely different physical phenomena:
- In **CT**, bone is extremely bright (high Hounsfield Units) and soft tissue is comparatively dim.
- In **MRI**, bone produces almost no signal and appears dark, while soft tissue is bright.

This inverted color scale makes multi-modal image registration one of the most challenging problems in medical image analysis. Simple pixel-by-pixel intensity matching algorithms fail completely because a bright tumor on an MRI may correspond to a dark region on a CT.

### The Goal
The objective of DeepMedAlign is to design an automated pipeline that takes a "moving" CT volume and a "fixed" MRI volume as input, and outputs a CT volume that is spatially warped to perfectly match the MRI. The system must operate with sub-second inference time to be clinically viable for real-time hospital workflows.

---

## 2. Data Engineering & Preprocessing

Medical scanners do not produce data in a form directly comparable across patients. Different hospitals and scanners produce volumes with different orientations, voxel spacings, and intensity ranges. We processed roughly 60GB of raw hospital-grade imaging data from the **SynthRAD2023 Brain dataset**.

### Standardizing the Geometry
1. **Orientation:** We utilized SimpleITK to reorient every scan (MRI and CT) into the standard **RAS+ coordinate system** (Right-Anterior-Superior). This guarantees all brains face the exact same direction natively.
2. **Isotropic Resampling:** Raw scans frequently have anisotropic voxel spacing (e.g., 0.5mm in-plane, 3mm slice thickness). We resampled every volume to a **1mm isotropic resolution**. This maps the MRI and CT onto a common physical grid, where exactly one voxel equals one cubic millimeter.

### Standardizing the Intensities
3. **CT Brain Window Clipping:** CT Hounsfield Units (HU) span from air (-1000) to dense bone (+1000). We clipped every CT volume to the **"Brain Window" of -15 to +80 HU**. This discards noisy air artifacts and excessively bright skull bone, forcing the model to focus strictly on clinically relevant brain parenchyma.
4. **Min-Max Normalization:** Neural networks experience severe gradient instability when inputs have different numerical scales. We applied min-max normalization to squeeze both MRI intensities and CT Hounsfield Units into a uniform **[0, 1] range**.

---

## 3. Classical Registration Baselines

To establish a strict performance floor and justify the use of deep learning, we first engineered a classical registration pipeline using **SimpleElastix**. 

### The Mathematical Optimizer
Because MSE (Mean Squared Error) fails on multi-modal scans, we employed **Mutual Information (MI)** as the loss metric. MI measures how much knowing the intensity value at a point in one image reduces uncertainty about the intensity at the corresponding point in the other image, regardless of their absolute brightness.

### The Three-Stage Pipeline
1. **Rigid Registration:** Allows only translation and rotation (6 degrees of freedom).
2. **Affine Registration:** Extends rigid registration with scaling and shearing (12 degrees of freedom).
3. **B-Spline Registration:** A deformable grid registration that non-linearly warps local soft tissue.

### The Flaw
The classical B-Spline approach achieved a baseline **Dice score of 0.77**. While acceptable, this accuracy came at a severe computational cost. Classical registration requires iterative CPU-based mathematical optimization independently for *every single patient*. There is no learned prior. This resulted in processing times of **several minutes per patient**, rendering the classical pipeline too slow for high-throughput clinical deployment.

---

## 4. Deep Learning Methodology

To achieve both high accuracy and instantaneous inference, we replaced the classical optimizer with an unsupervised Deep Learning framework.

### Architecture: VoxelMorph 3D U-Net
We engineered a 3D PyTorch implementation of **VoxelMorph**. The model employs a 3D U-Net that receives the moving CT and the fixed MRI stacked along the channel dimension. 
- **The Encoder** downsamples the input volume to extract global anatomical context and spatial relationships.
- **The Decoder** reconstructs the features via skip connections to preserve high-resolution anatomical boundaries.

Instead of directly outputting a new image, the network predicts a dense **3D Displacement Vector Field (DVF)**. Each voxel in the DVF contains a 3D vector `(dx, dy, dz)` indicating exactly how the corresponding CT voxel must be shifted to match the MRI anatomy.

### The Spatial Transformer Network (STN)
The predicted DVF is applied to the moving CT volume using a differentiable **Spatial Transformer Network** (implemented via `torch.nn.functional.grid_sample`). The STN physically warps the CT scan entirely inside the GPU. Because this operation is differentiable, we can backpropagate error gradients entirely end-to-end.

### The MIND-SSC Loss Function
Because we train unsupervised (without ground-truth deformation fields), the model must learn by minimizing the error between the Fixed MRI and the Warped CT. 

We utilized the **Modality Independent Neighbourhood Descriptor (MIND)** enhanced with **Self-Similarity Context (SSC)**. MIND does not compare pixel intensities; rather, it extracts a multi-dimensional descriptor representing the local structural geometry (edges, corners, and texture shapes). By comparing the structural geometry of the MRI against the structural geometry of the CT, the network seamlessly bridges the modality gap.

We combined MIND-SSC with an **L2 Smoothness Regularization Penalty** applied to the DVF gradients, preventing unrealistic anatomical folding or tearing (Jacobian determinants $\le 0$).

---

## 5. Results & Evaluation

The deep learning framework was evaluated on an unseen test cohort. Performance was quantified using two primary clinical metrics:
1. **Dice Similarity Coefficient (DSC):** Measures volumetric overlap of the brain structures (1.0 is perfect overlap).
2. **Hausdorff Distance 95th Percentile (HD95):** Measures the maximum surface boundary error in millimeters, discarding the top 5% of outliers.

### Quantitative Superiority
The VoxelMorph framework absolutely obliterated the classical baselines. It achieved a **Dice Score of 0.96** (a 19% absolute improvement over B-Spline) and minimized the boundary error to a microscopic **1.2 mm HD95**.

![Comparison of registration methods](../results/figures/methods_comparison.png)
*Figure 1: Deep Learning (VoxelMorph v2) massively outperforms Rigid, Affine, and B-Spline baselines.*

### Qualitative Alignment
Visual inspection of the difference maps confirms the model's precision. The residual error between the Fixed MRI and the Warped CT is nearly indistinguishable, proving that the model successfully deformed the bone and tissue structures to match the MRI without causing structural collapse.

![VoxelMorph Difference Map](../results/figures/voxelmorph_diffmap.png)
*Figure 2: Center-slice difference map showcasing minimal residual error after VoxelMorph alignment.*

### Computational Efficiency
While the classical pipeline required minutes of CPU processing per patient, the trained VoxelMorph model performs complete 3D deformable registration in **less than 1.0 seconds on a GPU** during inference. 

---

## 6. Conclusion

The DeepMedAlign project demonstrates the overwhelming superiority of deep learning in multi-modal medical image registration. By replacing traditional iterative optimization with a 3D U-Net and a differentiable Spatial Transformer Network, we eliminated the inference bottleneck. Furthermore, by utilizing the MIND-SSC structural similarity loss, the model flawlessly learned the complex non-linear mapping between MRI and CT intensities.

Our framework delivers robust, high-accuracy (0.96 Dice), and clinically practical (sub-second) MRI–CT deformable registration, opening the door for automated real-time radiation therapy planning.
