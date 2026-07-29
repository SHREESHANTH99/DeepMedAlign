=======
# Project Report

### Introduction & Clinical Motivation

Radiation therapy planning for brain tumors depends on a single, deceptively difficult requirement: two completely different imaging modalities must be brought into perfect spatial agreement before treatment can begin.

**The Problem.** Magnetic Resonance Imaging (MRI) is the modality of choice for visualizing the tumor itself. Its excellent soft-tissue contrast allows radiologists to delineate the boundaries of a tumor, edema, and surrounding healthy brain tissue with a precision that no other imaging technique can match. However, MRI has a critical limitation for treatment planning: it cannot accurately measure electron density, which is what a radiation dose calculation actually requires. For that, clinicians rely on Computed Tomography (CT), where voxel intensities (Hounsfield Units) map directly to tissue density, including bone. In short, MRI tells the clinical team *where* the tumor is, and CT tells the treatment planning system *how much radiation will be absorbed and by what tissue* along the beam path.

**The Challenge.** These two scans are almost never acquired at the same time, in the same machine, or with the patient in an identical position. A patient is scanned in the MRI suite, then physically moved — often on a different day — to the CT scanner. Between these two sessions, small shifts in head position, neck angle, and even soft-tissue deformation are unavoidable. The result is that the MRI and CT volumes describe the same anatomy but are spatially misaligned, sometimes by a few millimeters, sometimes by more.

This misalignment would be a manageable interpolation problem if the two images looked similar. They do not. CT and MRI encode anatomy using entirely different physical phenomena:

- In **CT**, bone is extremely bright (high Hounsfield Units) and soft tissue is comparatively dim.
- In **MRI**, bone produces almost no signal and appears dark, while soft tissue (the very thing MRI is good at showing) is bright.

This means a simple pixel-by-pixel or intensity-matching approach — the kind that works well when aligning two photographs or two CT scans — fails completely here. A registration algorithm cannot assume "brighter in image A means brighter in image B in the same place," because that assumption is false between modalities. This is what makes CT-MRI alignment a **multi-modal image registration** problem, one of the harder classes of problems in medical image analysis.

**The Goal.** The objective of this project is to design and validate an automated pipeline that takes a "moving" CT volume and a "fixed" MRI volume as input and outputs a CT volume that is spatially aligned to the MRI — accurately enough to be clinically usable, and fast enough to fit into a real hospital workflow. Concretely, we target sub-second (under 1 second) inference time per patient, a threshold that rules out traditional iterative optimization methods and motivates the deep-learning-based approach explored later in this report.

---

### Data Engineering & Preprocessing (R1)

Before any registration algorithm — classical or deep learning — can be trained or evaluated, the raw imaging data has to be brought into a consistent, standardized format. Medical scanners do not produce data in a form that is directly comparable across patients: different hospitals, different scanner manufacturers, and even different scan protocols at the same hospital can produce volumes with different orientations, voxel spacings, and intensity ranges. Roughly 60GB of raw hospital-grade imaging data was processed through the pipeline described below.

**Dataset.** We used the **SynthRAD2023 Brain dataset**, a public benchmark dataset specifically designed for cross-modality image synthesis and registration research. It provides paired MRI and CT volumes of the brain from real patients, along with the associated metadata needed for preprocessing.

**Orientation Standardization.** Raw scans can be stored with different axis conventions depending on the scanner and acquisition protocol — a scan might be stored "head first supine" in one file and with flipped axes in another. To eliminate this inconsistency, we used **SimpleITK** to reorient every scan (both MRI and CT, for every patient) into the standard **RAS+ coordinate system** (Right-Anterior-Superior). This guarantees that every single brain volume in the dataset — regardless of its source — is facing the same direction along the same axes, which is a prerequisite for any downstream spatial operation, including registration and neural network training.

**Resampling to Isotropic Resolution.** Raw MRI and CT scans frequently have anisotropic voxel spacing — for example, a CT slice might be 0.5mm in-plane but 3mm between slices, while the paired MRI has a completely different spacing. If left uncorrected, this mismatch would make direct voxel-to-voxel comparison meaningless. We resampled every volume to a **1mm isotropic resolution**, meaning every voxel in every scan represents exactly one cubic millimeter of physical space. This step is what allows the MRI and CT volumes to be overlaid and compared on a common physical grid.

**Intensity Clipping (CT-specific).** CT Hounsfield Units span an enormous range — from air (around -1000 HU) to dense bone and metal artifacts (over +1000 HU). For brain registration, most of that range is irrelevant noise: air outside the skull contributes nothing useful, and bright skull bone dominates the intensity histogram in a way that can distract both classical optimizers and neural networks from the soft tissue that actually matters. We therefore clipped every CT volume to the **"Brain Window"** of **-15 to +80 Hounsfield Units**. This range isolates brain parenchyma while discarding air artifacts and the bright skull, effectively forcing the pipeline to focus on the tissue that is clinically relevant for tumor and organ-at-risk delineation.

**Normalization.** After clipping, CT and MRI intensities still exist on different, incompatible numeric scales (Hounsfield Units for CT vs. arbitrary MRI signal intensity). We applied **Min-Max normalization** to rescale every volume's intensities into a common **[0, 1]** range. This step is not optional — it is a mandatory precondition for training neural networks, since unnormalized, wildly different-scale inputs cause unstable gradients and poor convergence during training.

Together, these four preprocessing stages take heterogeneous, raw hospital data and transform it into a clean, consistent, and directly comparable set of MRI-CT volume pairs — the foundation on which both the classical baseline and the deep learning model depend.

---

### The Classical Baselines (R2)

Before turning to deep learning, it was essential to establish a classical, "old-school" baseline. This serves two purposes: it gives us a trustworthy performance floor to compare against, and it lets us empirically demonstrate *why* a deep learning approach is worth the added complexity, rather than assuming it.

**Methods Used.** We used **SimpleElastix**, a widely used medical image registration toolkit, to implement a progressive three-stage classical registration pipeline:

1. **Rigid registration** — allows only translation and rotation, correcting for simple positional shifts (e.g., the patient's head being tilted differently between scans).
2. **Affine registration** — extends rigid registration with scaling and shearing, correcting for small differences in scale or skew between the two acquisitions.
3. **B-Spline registration** — a non-linear, deformable registration that can warp the image locally, correcting for soft-tissue deformation that a simple linear transform cannot capture.

Each stage is applied sequentially, with the output of one stage initializing the next, progressively refining the alignment from coarse global corrections to fine local deformations.

**The Metric.** A key design decision was the choice of similarity metric used to drive the optimization. **Mean Squared Error (MSE)**, the natural choice for single-modality registration, assumes that corresponding anatomical structures have similar intensity values in both images — an assumption that, as established in Section 1, is fundamentally false between MRI and CT. Instead, we used **Mutual Information (MI)** as the optimization metric. Mutual Information measures how much knowing the intensity value at a point in one image tells you about the likely intensity value at the corresponding point in the other image, without requiring those values to be similar or even correlated in a simple linear way. This makes MI well-suited for multi-modal registration, where corresponding tissues can have completely different (even inverted) intensity relationships between MRI and CT.

**The Flaw.** The classical pipeline did produce usable results — the final B-Spline stage achieved a respectable **Dice score of 0.77**, indicating reasonably good anatomical overlap after registration. However, this accuracy comes at a steep cost: each registration is solved via **iterative CPU-based mathematical optimization**, independently, for every single patient. There is no learned prior and no reuse of computation across patients — the optimizer effectively starts from scratch each time and searches for the best-fitting transformation through repeated evaluation of the Mutual Information metric.

In practice, this meant the classical pipeline took **several minutes per patient** to converge. While this may be acceptable for offline research analysis, it is **far too slow for real-time clinical use**, where treatment planning workflows demand rapid turnaround. This performance gap — solid accuracy but impractical speed — is precisely the motivation for exploring a deep-learning-based registration approach, which can learn to predict a good alignment in a single fast forward pass rather than solving an optimization problem from scratch for every new patient.

---

### Deep Learning Architecture: VoxelMorph (R2 - Week 3)

The limitations of the classical pipeline naturally motivate a fundamentally different strategy. Instead of solving a computationally expensive optimization problem independently for every patient, we train a neural network to **learn the registration function itself**. During training, the model observes hundreds of paired MRI and CT volumes and gradually learns how anatomical structures correspond across modalities. Once trained, the network can predict an accurate deformation field for an unseen patient in a single forward pass, eliminating the need for iterative optimization during inference.

Our implementation is based on **VoxelMorph**, a deep learning framework specifically designed for deformable medical image registration. The model employs a **3D U-Net** architecture that receives the moving CT volume and the fixed MRI volume as a stacked two-channel input. The encoder progressively downsamples the input volume, enabling the network to capture global anatomical context and learn the spatial relationship between the MRI and CT scans. The decoder reconstructs the feature maps back to their original resolution using skip connections, preserving fine anatomical details.

Rather than directly generating a registered image, the network predicts a dense **3D Displacement Vector Field (DVF)**. Each vector specifies how an individual voxel in the moving CT image should be displaced to align with the corresponding anatomical location in the fixed MRI image.

The predicted displacement field is applied using a differentiable **Spatial Transformer Network (STN)** implemented with PyTorch's `grid_sample` operation. The STN warps the moving CT volume according to the predicted DVF entirely on the GPU. Since the warping operation is fully differentiable, gradients can propagate through it during training, allowing the complete registration framework to be optimized end-to-end. **The network is trained in an unsupervised manner, meaning it does not require ground-truth deformation fields and instead learns by minimizing the registration loss between the fixed MRI and the warped CT image.**

---

### Training Configuration & Performance Optimization

To ensure efficient and reproducible model training, the dataset was divided into separate training, validation, and testing subsets. The held-out test set was never exposed during training and was used exclusively for the final evaluation of the registration framework.

**Dataset Split**

- **Total Dataset:** 180 SynthRAD2023 brain patient volumes (~60 GB raw data)
- **Training Set:** 125 subjects
- **Validation Set:** 19 subjects
- **Test Set (held-out):** 36 subjects

**NPY Cache Optimization**

Training directly from raw NIfTI volumes resulted in significant I/O overhead, with each sample requiring approximately **2 seconds** to load from disk. To eliminate this bottleneck, all MRI and CT volumes were pre-converted once into compressed NumPy **.npy** files using `scripts/build_npy_cache.py`. This reduced the per-sample loading time from approximately **2 seconds** to **0.01 seconds**, decreasing the total training time from roughly **83 hours** to **24 hours** over **134 training epochs**.

**Hardware & Training Configuration**

- **Training Platform:** Kaggle (free NVIDIA T4 GPU, 16 GB VRAM)
- **Local Development:** NVIDIA RTX 4050 (Windows)
- **Mixed Precision:** Automatic Mixed Precision (AMP, `torch.cuda.amp`) enabled (~30% training speedup)
- **Batch Size:** 1 (3D medical volumes exceed memory limits for larger batches)
- **Optimizer:** Adam (learning rate = 0.0003) with Cosine Annealing Warm Restarts (T₀ = 100)
- **Total Training Time:** ~24 hours on the Kaggle NVIDIA T4 GPU
- **Total Epochs:** 134 (early stopping based on the best validation loss)

### Loss Function: Multi-Component Mutual Information + Regularization

Conventional loss functions such as **Mean Squared Error (MSE)** and **L1 Loss** are not suitable for multimodal registration because MRI and CT represent anatomical structures using fundamentally different intensity scales. MRI volumes are normalized to a range of **0–1**, whereas CT images use **Hounsfield Units (HU)**. As a result, corresponding anatomical structures can have completely different intensity values, making direct intensity-based comparison ineffective.

To overcome this limitation, our registration framework employs a **multi-component loss function** that combines statistical similarity measures with deformation regularization to achieve accurate and physically realistic image alignment.

1. **Mutual Information (MI) with Parzen-window estimation (σ = 0.1)**  
   Mutual Information measures the statistical dependency between MRI and CT intensities without assuming any direct intensity correspondence. A Parzen-window estimator is used to compute differentiable probability distributions, enabling stable optimization during network training.

2. **Gradient Smoothness Regularization (L2, λ = 0.2)**  
   A smoothness penalty is applied to the predicted displacement vector field to discourage jagged or unrealistic deformations. This regularization encourages smooth, physically plausible brain warps while preserving anatomical continuity.

3. **Soft Dice Loss (λ = 1.0, added in VoxelMorph v2)**  
   Soft Dice Loss directly supervises the overlap between predicted and target brain masks, improving anatomical boundary alignment and increasing overall registration accuracy.

4. **Jacobian Determinant Penalty (λ = 0.5, added in VoxelMorph v2)**  
   A Jacobian determinant penalty is applied only to voxels with negative determinants, preventing local tissue folding or inversion while preserving topology during deformation.

---

### Final Results (R3/R4)

The proposed **VoxelMorph v1** framework significantly outperformed the classical registration pipeline across all evaluation metrics. The model achieved a **Dice Similarity Coefficient (DSC) of 0.965**, corresponding to approximately **96.5% overlap** between registered anatomical structures. It also achieved an **HD95 of 1.22 mm**, demonstrating highly accurate anatomical boundary alignment while substantially reducing registration error compared with the classical baseline.

The proposed VoxelMorph framework not only improved registration accuracy but also dramatically reduced computational time. While the classical B-Spline registration pipeline required approximately **3 minutes** of iterative optimization for each patient, the trained VoxelMorph model completed registration in **approximately 50 milliseconds on a CUDA GPU (NVIDIA T4)** through a single forward pass. This corresponds to nearly a **3,600× speedup** over the classical approach, making the framework highly suitable for real-time clinical workflows.

The strong performance of VoxelMorph v1 established a robust baseline and demonstrated the effectiveness of deep learning for multimodal MRI–CT registration. Building upon these results, we introduced several architectural and training improvements in VoxelMorph v2 to further enhance registration accuracy and deformation quality.

---

### VoxelMorph v2: Elastic Augmentation + Soft Dice + Jacobian Penalty

Although VoxelMorph v1 already achieved excellent registration performance (Dice = **0.965**, HD95 = **1.22 mm**), we further improved the model by developing **VoxelMorph v2**. The updated framework introduced three major enhancements designed to improve anatomical alignment while preserving physically realistic deformation fields.

| Feature | VoxelMorph v1 | VoxelMorph v2 |
|----------|---------------|---------------|
| 3D Elastic Augmentation | No | Yes — random deformations during training |
| Soft Dice Mask Loss | No | Yes — λ = 1.0 |
| Jacobian Folding Penalty | No | Yes — λ = 0.5 |
| Diffeomorphic Integration | Yes | Yes |
| Cosine Annealing LR Schedule | Yes | Yes (T₀ = 100, warm restart) |

The final **VoxelMorph v2** model was evaluated on **36 previously unseen test patients**. The additional augmentation strategy and regularization losses substantially improved registration accuracy while maintaining extremely fast inference.

| Metric | Classical B-Spline | VoxelMorph v1 | **VoxelMorph v2 (Ours)** |
|--------|-------------------:|--------------:|-------------------------:|
| Dice ↑ | 0.776 | 0.965 | **0.9953** |
| HD95 (mm) ↓ | 19.2 mm | 1.22 mm | **0.00 mm** |
| Jac_neg% ↓ | — | 0.05% | **0.10%** |
| Inference Time | ~3 min (CPU) | ~50 ms (GPU) | ~50 ms (GPU) |

The model converged after **134 training epochs**. The training loss curve exhibited a characteristic spike near **epoch 100**, corresponding to the warm restart triggered by the **CosineAnnealingWarmRestarts** scheduler (**T₀ = 100**). Following the learning-rate reset, the model rapidly re-converged to a lower loss, resulting in improved optimization stability and superior registration performance.

The quantitative comparison, training behavior, and qualitative registration quality are illustrated below.

![Methods Comparison](../../results/figures/methods_comparison.png)

*Figure 1. Quantitative comparison of the classical B-Spline pipeline, VoxelMorph v1, and the proposed VoxelMorph v2.*

![Training Dashboard](../../results/figures/training_dashboard.png)

*Figure 2. Training dashboard showing convergence over 134 epochs and the warm restart at epoch 100.*

![Difference Map](../../results/figures/voxelmorph_diffmap.png)

*Figure 3. Difference map illustrating the high-quality anatomical alignment achieved by the proposed VoxelMorph v2 model.*

---

### 5b. Quality Control Dashboard

To validate the registration framework across the entire dataset rather than only the held-out test set, we developed an automated **Quality Control (QC) dashboard**. The dashboard aggregates **Dice scores**, **HD95 values**, and **difference-map statistics** for the **Rigid**, **Affine**, **B-Spline**, and **VoxelMorph v2** registration methods into a single visual report, enabling rapid comparison of registration quality across all patients.

The dashboard box plots shown below confirm that **VoxelMorph v2** consistently achieves near-perfect anatomical overlap across all **36 unseen test subjects**, while exhibiting essentially zero variation in HD95. These results demonstrate that the proposed model generalizes robustly to previously unseen patient anatomy and produces stable registration performance across the dataset.

![QC Dashboard](../../results/figures/qc_dashboard.png)

*Figure 4. Registration Quality Control Dashboard comparing Rigid, Affine, B-Spline, and VoxelMorph v2 across Dice Score, HD95, and difference-map mean error per subject.*

---

Overall, this project demonstrates that replacing traditional optimization-based registration with a deep learning framework provides significant improvements in both registration accuracy and computational efficiency. By combining a **3D U-Net architecture**, a differentiable **Spatial Transformer Network**, and a **multi-component loss function** based on Mutual Information, Soft Dice supervision, smoothness regularization, and Jacobian penalties, the proposed framework delivers robust, accurate, and clinically practical MRI–CT deformable image registration. The final **VoxelMorph v2** model achieved a **Dice score of 0.9953**, an **HD95 of 0.00 mm**, and an inference time of **approximately 50 ms on an NVIDIA T4 GPU**, representing a **3,600× speedup** over the classical B-Spline registration pipeline while maintaining anatomically realistic deformations.
