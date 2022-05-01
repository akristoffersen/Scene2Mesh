
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


def calculate_interpolation(v_0, v_1, v_2, points):
    delta = v_1 - v_0

    point_l = -1 * (v_2[0] - v_0[0]) * delta[1] + (v_2[1] - v_0[1]) * delta[0]
    eval_l = -1 * (points[:, 0] - v_0[0]) * delta[1] + (points[:, 1] - v_0[1]) * delta[0]

    return eval_l / point_l


def uv_to_barycentric(vertices, points):
    '''
    Args:
        vertices: [3, 2]
        points: [N, 2]
    Returns:
        bary coords: [N, 3]
    '''
    N = points.shape[0]
    abcs = np.zeros((N, 3))

    v_0, v_1, v_2 = vertices

    abcs[:, 0] = calculate_interpolation(v_0, v_1, v_2, points)
    abcs[:, 1] = calculate_interpolation(v_1, v_2, v_0, points)
    abcs[:, 2] = calculate_interpolation(v_2, v_0, v_1, points)

    return abcs


def bary_to_xyz(barys, verts):
    '''
    Args:
        barys: [N, 3]
        verts: [3, 3]
    Returns:
        xyzs: [N, 3]
    '''
    N = barys.shape[0]
    xyzs = verts[0] * barys[:, 0] + verts[1] * barys[:, 1] + verts[2] * barys[:, 2]
    return xyzs


def draw_texture_map(H, W, vertices, uv_map, sample_fn=spherical_sample_fn, sub_sample=4):
    '''
        Creates texture map from vertex positions and uv_map.
    Args:
        H: int
        W: int
        vertices: np.array, [|V|, 3]
        uv_map: dict, see format from blender script
        sample_fn: fn that takes in [..., 3], outputs rgbs
        sub_sample: int, increase to decrease aliasing
    Returns:
        result: [H, W, 3]
    '''
    V = vertices.shape[0]
    sqrt_sub = int(np.sqrt(sub_sample))
    buffer = np.zeros(H * sqrt_sub, W * sqrt_sub, 3)

    pixel_offset = 1 / sqrt_sub
    for i in range(len(uv_map)):
        loop = uv_map[i]
        vertex_ids = list(loop.keys())

        v_positions = np.stack(
            [vertices[vertex_ids[i]] for i in range(3)]
        ) # (3, 3)

        uv_locs = np.stack(
            [loop[vertex_ids[i]] for i in range(3)]
        ) # (3, 2)

        u_min = np.min(uv_locs[:, 0])
        u_max = np.max(uv_locs[:, 0])

        v_min = np.min(uv_locs[:, 1])
        v_max = np.max(uv_locs[:, 1])

        # these uv min/max correspond to locations in the buffer

        buffer_h_min = int(np.floor(u_min * buffer.shape[0]))
        buffer_h_max = max(
            int(np.ceil(u_max * buffer.shape[0])),
            buffer.shape[0] - 1,
        )
        buffer_w_min = int(np.floor(v_min * buffer.shape[1]))
        buffer_w_max = max(
            int(np.ceil(v_max * buffer.shape[1])),
            buffer.shape[1] - 1,
        )

        buffer_locs = np.meshgrid(
            range(buffer_h_min, buffer_h_max),
            range(buffer_w_min, buffer_w_max),
        )
        buffer_hh, buffer_ww = buffer_locs

        # convert these to positions in uv space
        buffer_hh += 0.5 # get in the middle of each sub pixel
        buffer_ww += 0.5

        uus = buffer_hh / buffer.shape[0]
        vvs = buffer_ww / buffer.shape[1]

        uv_positions = np.dstack([uus.ravel(), vvs.ravel()])[0]

        # mask for which pixels are inside the triangle

        # from that, call sample_fn on them

        # write to buffer
    
    # average buffer (don't know how to do this yet)
    return buffer