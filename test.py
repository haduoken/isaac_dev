import numpy as np
from matplotlib import pyplot as plt
import cv2
from torchvision.models import swin_b

b = np.stack(np.meshgrid([1, 2, 3], [4, 5, 6], indexing='xy'))

print(b[:, 0, 0])
print(b[:, 0, 1])
print(b[:, 0, 2])

print(b[:, 1, 0])
print(b[:, 1, 1])
print(b[:, 1, 2])
