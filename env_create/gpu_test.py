import os
import torch

print("=== SLURM Info ===")
for key in ["SLURM_JOB_ID", "SLURM_NODELIST", "CUDA_VISIBLE_DEVICES"]:
    print(f"{key}: {os.environ.get(key)}")

print("\n=== PyTorch / CUDA Info ===")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"GPU {i}: {name}, compute capability {cap}")

if torch.cuda.is_available():
    x = torch.randn((2048, 2048), device='cuda')
    y = torch.randn((2048, 2048), device='cuda')
    z = x @ y
    print("\nMatmul OK:", z[0,0].item())
else:
    print("\nCUDA not available.")

print("\n=== GPU Test Complete ===")
