import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.voxelmorph_model import VoxelMorph

def load_nifti_to_tensor(path: str, normalize: bool = False) -> torch.Tensor:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    
    if normalize:
        v_min, v_max = arr.min(), arr.max()
        if v_max > v_min:
            arr = (arr - v_min) / (v_max - v_min)
            
    if arr.ndim == 3:
        arr = np.expand_dims(np.expand_dims(arr, 0), 0)
    elif arr.ndim == 4:
        arr = np.expand_dims(arr, 0)
    return torch.from_numpy(arr), img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mri", required=True)
    parser.add_argument("--ct", required=True)
    parser.add_argument("--model", default="models/voxelmorph_best.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mr_t, mr_img = load_nifti_to_tensor(args.mri, normalize=True)
    ct_t, ct_img = load_nifti_to_tensor(args.ct, normalize=True)
    
    # Downsample
    mr_down = F.interpolate(mr_t.to(args.device), scale_factor=0.5, mode='trilinear', align_corners=False)
    ct_down = F.interpolate(ct_t.to(args.device), scale_factor=0.5, mode='trilinear', align_corners=False)
    
    # Run model
    model = VoxelMorph(vol_size=(80, 96, 80)).to(args.device)
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=args.device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint))
    else:
        print(f"Warning: Model {args.model} not found, using untrained model for visualization!")
        
    model.eval()
    with torch.no_grad():
        warped_ct_down, _ = model(mr_down, ct_down)
        
    # Get center slice
    slice_idx = mr_down.shape[2] // 2
    
    mr_slice = mr_down[0, 0, slice_idx, :, :].cpu().numpy()
    ct_slice = ct_down[0, 0, slice_idx, :, :].cpu().numpy()
    warped_slice = warped_ct_down[0, 0, slice_idx, :, :].cpu().numpy()
    
    # Calculate differences
    diff_unaligned = np.abs(mr_slice - ct_slice)
    diff_aligned = np.abs(mr_slice - warped_slice)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(mr_slice, cmap='gray')
    axes[0].set_title("Fixed MRI")
    axes[0].axis('off')
    
    axes[1].imshow(ct_slice, cmap='gray')
    axes[1].set_title("Moving CT (Unaligned)")
    axes[1].axis('off')
    
    axes[2].imshow(warped_slice, cmap='gray')
    axes[2].set_title("Warped CT (VoxelMorph)")
    axes[2].axis('off')
    
    # Display difference maps with hot colormap
    axes[3].imshow(diff_unaligned, cmap='hot', alpha=0.5)
    axes[3].set_title("Diff: Unaligned (Hot = Error)")
    axes[3].axis('off')
    
    plt.tight_layout()
    out_path = out_dir / "voxelmorph_diffmap.png"
    plt.savefig(out_path, dpi=300)
    print(f"[SUCCESS] VoxelMorph Difference Map saved to {out_path}")

if __name__ == "__main__":
    main()
