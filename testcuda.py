import torch

use_gpu = torch.cuda.is_available()
print(use_gpu)
print(torch.backends.cudnn.enabled)