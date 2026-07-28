import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.voxelmorph_model import VoxelMorph

def load_nifti_to_tensor(path: str) -> torch.Tensor:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    
    # Min-max normalization
    v_min, v_max = arr.min(), arr.max()
    if v_max > v_min:
        arr = (arr - v_min) / (v_max - v_min)
    
    # Ensure shape is (1, 1, D, H, W)
    if arr.ndim == 3:
        arr = np.expand_dims(np.expand_dims(arr, 0), 0)
    elif arr.ndim == 4:
        arr = np.expand_dims(arr, 0)
        
    return torch.from_numpy(arr), img

def main():
    parser = argparse.ArgumentParser(description="Run VoxelMorph on a new patient (MRI and CT)")
    parser.add_argument("--mri", required=True, help="Path to preprocessed MRI NIfTI")
    parser.add_argument("--ct", required=True, help="Path to preprocessed CT NIfTI")
    parser.add_argument("--output", required=True, help="Path to save warped CT NIfTI")
    parser.add_argument("--model", default="models/voxelmorph_best.pth", help="Path to trained .pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading MRI: {args.mri}")
    print(f"Loading CT: {args.ct}")
    
    # 1. Load images
    mr_tensor, mr_img = load_nifti_to_tensor(args.mri)
    ct_tensor, ct_img = load_nifti_to_tensor(args.ct)
    
    original_shape = mr_tensor.shape[2:] # (D, H, W)
    
    mr_tensor = mr_tensor.to(args.device)
    ct_tensor = ct_tensor.to(args.device)
    
    # 2. Downsample to (80, 96, 80) since model was trained on this
    mr_down = F.interpolate(mr_tensor, scale_factor=0.5, mode='trilinear', align_corners=False)
    ct_down = F.interpolate(ct_tensor, scale_factor=0.5, mode='trilinear', align_corners=False)
    
    print(f"Loaded tensors. Original shape: {original_shape}, Downsampled to: {mr_down.shape[2:]}")
    
    # 3. Load Model
    print(f"Loading VoxelMorph model from {args.model}...")
    model = VoxelMorph(vol_size=(80, 96, 80)).to(args.device)
    
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found! Run training first.")
        return
        
    checkpoint = torch.load(args.model, map_location=args.device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 4. Predict
    print("Running deep learning alignment...")
    with torch.no_grad():
        warped_ct_down, dvf = model(mr_down, ct_down)
        
    # 5. Upsample back to original resolution
    warped_ct = F.interpolate(warped_ct_down, size=original_shape, mode='trilinear', align_corners=False)
    
    # 6. Save output
    warped_arr = warped_ct.squeeze().cpu().numpy()
    
    # Denormalize (optional, but standard NIfTI is just saved as float)
    out_img = sitk.GetImageFromArray(warped_arr)
    out_img.CopyInformation(ct_img)
    
    sitk.WriteImage(out_img, args.output)
    print(f"[SUCCESS] Aligned CT saved to {args.output}")

if __name__ == "__main__":
    main()
