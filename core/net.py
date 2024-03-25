import torch
from torch import nn
import torch.nn.functional as F
from net_module import *

import torchvision
from torchvision.models import swin_t

backbone = swin_t(torchvision.models.Swin_T_Weights)

img = torch.zeros((3, 3, 224, 224), dtype=torch.float32)

feat = backbone(img)
