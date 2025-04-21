import torch
import torch_directml

dml = torch_directml.device()
print(f"Device: {dml}")
# print(f"GPU Name: {torch_directml.device_name(dml)}")

# ทดสอบการคำนวณ
a = torch.tensor([1,2,3], device=dml)
b = torch.tensor([4,5,6], device=dml)
print(f"Result: {a + b}")



