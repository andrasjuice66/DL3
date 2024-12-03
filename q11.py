import torch

# Global max pool: reduces h,w dimensions to 1, then squeezes them out
global_max = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]  # (b,c)

# Global mean pool: takes mean across both h,w dimensions at once
global_mean = torch.mean(x, dim=(-2,-1))  # (b,c)

