
import numpy as np
import torch


def spherical_sample_fn(x):
    '''
    Converts an x value to an rgb color based off of its normal

    Args:
        x: (N, 3)
    Returns:
        col: (N, 3), [0, 1] rgb
    '''
    x = x / np.linalg.norm(x)
    return x


def normuv(uv, H, W):
    """
    Normalize pixel coordinates to lie in [-1, 1]
    
    :param uv (..., 2) unnormalized pixel coordinates for HxW image
    """
    u = uv[..., 0] / H * 2.0 - 1.0
    v = uv[..., 1] / W * 2.0 - 1.0
    return torch.stack([u, v], dim=-1)


def unnormuv(uv, H, W):
    """
    Un-normalize pixel coordinates
    
    :param uv (..., 2) normalized pixel coordinates in [-1, 1]
    """
    u = (uv[..., 0] + 1.0) / 2.0 * H
    v = (uv[..., 1] + 1.0) / 2.0 * W
    return torch.stack([u, v], dim=-1)
    

def get_uv(H, W):
    """
    Get normalized uv coordinates for image
    :param height (int) source image height
    :param width (int) source image width
    :return uv (N, 2) pixel coordinates in [-1.0, 1.0]
    """
    yy, xx = torch.meshgrid(
        (torch.arange(H, dtype=torch.float32) + 0.5),
        (torch.arange(W, dtype=torch.float32) + 0.5),
        indexing='ij',
    )
    uv = torch.stack([xx, yy], dim=-1) # (H, W, 2)
    uv = normuv(uv, W, H) # (H, W, 2)
    return uv.view(H * W, 2)


def create_spherical_uv_sampling(im_shape):
    '''
    creates the sampling of directions from the origin around 360 space.
    Args:
        im_shape: (H, W)
    Returns:
        uv sampling map (H, W)
    '''
    H, W = im_shape
    phis_thetas = get_uv(H, W).numpy() * [np.pi, 2 * np.pi] # phi, theta
    uv_directions = np.column_stack(
        [
            np.sin(phis_thetas[:, 0]) * np.cos(phis_thetas[:, 1]),
            np.sin(phis_thetas[:, 0]) * np.sin(phis_thetas[:, 1]),
            np.cos(phis_thetas[:, 0]),
        ],
    )
    return uv_directions

def get_spherical_coords(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    theta = np.arccos(X[:, 2] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 1], X[:, 0])

    # Normalize both to be between [-1, 1]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv],1)