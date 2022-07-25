import torch
import re
from torchvision import transforms
import numpy as np
import cv2


def detect_black(input_img, threshold = -0.95):
    """
    Detecting background regions.
    """
    black_detect = torch.bitwise_and(input_img[:,0:1]<=threshold, torch.bitwise_and(input_img[:, 1:2]<= threshold, input_img[:, 2:3]<= threshold))
    return black_detect

def load_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)

def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )

def read_depth(path):
    depth =cv2.imread(path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mask = np.isinf(depth)
    depth[mask] = 0.0
    return depth[:,:, 0]