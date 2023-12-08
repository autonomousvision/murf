import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils
import cv2
from matplotlib.cm import get_cmap
import matplotlib as mpl
import matplotlib.cm as cm


def viz_depth_tensor(disp, return_numpy=False, colormap='plasma'):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz

    