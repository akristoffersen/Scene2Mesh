
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
    x = 0.5 + 0.5 * x
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

    abcs[:, 2] = calculate_interpolation(v_0, v_1, v_2, points)
    abcs[:, 0] = calculate_interpolation(v_1, v_2, v_0, points)
    abcs[:, 1] = calculate_interpolation(v_2, v_0, v_1, points)

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
    xyzs = (verts[0][:, None] * barys[:, 0]).T
    xyzs += (verts[1][:, None] * barys[:, 1]).T
    xyzs += (verts[2][:, None] * barys[:, 2]).T
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
    buffer = np.zeros((H * sqrt_sub, W * sqrt_sub, 3))

    pixel_offset = 1 / sqrt_sub
    for i in range(len(uv_map)): # len(uv_map)
        print(i)
        loop = uv_map[i]
        vertex_ids = (list(loop.keys()))
        
        v_positions = np.stack(
            [vertices[vertex_ids[i]] for i in range(3)]
        ) # (3, 3)
        # print("v_positions:", v_positions)
        uv_locs = np.stack(
            [loop[vertex_ids[i]] for i in range(3)]
        )
        uv_locs[:, [0, 1]] = uv_locs[:, [1, 0]] # swap
        uv_locs[:, 0] = 1 - uv_locs[:, 0]
        # print("uv_locs:", uv_locs)
         # (3, 2)
        u_min = np.min(uv_locs[:, 0])
        u_max = np.max(uv_locs[:, 0])

        v_min = np.min(uv_locs[:, 1])
        v_max = np.max(uv_locs[:, 1])

        # these uv min/max correspond to locations in the buffer

        buffer_h_min = int(np.floor(u_min * buffer.shape[0]))
        buffer_h_max = min(
            int(np.ceil(u_max * buffer.shape[0])),
            buffer.shape[0] - 1,
        )
        buffer_w_min = int(np.floor(v_min * buffer.shape[1]))
        buffer_w_max = min(
            int(np.ceil(v_max * buffer.shape[1])),
            buffer.shape[1] - 1,
        )

        buffer_locs = np.meshgrid(
            range(buffer_h_min, buffer_h_max),
            range(buffer_w_min, buffer_w_max),
            indexing='ij',
        )
        buffer_hh, buffer_ww = buffer_locs


        # convert these to positions in uv space
        uus = buffer_hh / buffer.shape[0] + 1.0 / (2 * buffer.shape[0])
        vvs = buffer_ww / buffer.shape[1] + 1.0 / (2 * buffer.shape[1])
        uv_positions = np.dstack([uus.ravel(), vvs.ravel()])[0]
        barycentrics = uv_to_barycentric(uv_locs, uv_positions)
        # return buffer_locs
        mask = np.all(barycentrics >= 0.0, axis= 1) * np.all(barycentrics <= 1.0, axis= 1)
        mask *= np.isclose(np.sum(barycentrics, axis=1), 1.0)
        ma_box = mask.reshape((buffer_h_max - buffer_h_min, buffer_w_max - buffer_w_min))
        # return ma_box
        # calculate xyzs of each texel
        xyzs = bary_to_xyz(barycentrics[mask], v_positions)
        
        # sample rgbs
        rgbs = sample_fn(xyzs)
        # buffer_temp = np.zeros(
        #     (buffer_h_max - buffer_h_min, buffer_w_max - buffer_w_min, 3),
        # )
        # buffer_temp[ma_box] = rgbs
        # return buffer_temp
        
        # write to the buffer
        buffer[buffer_h_min: buffer_h_max, buffer_w_min: buffer_w_max][ma_box] = rgbs
    
    # TODO: average buffer (don't know how to do this yet)
    # buffer = np.flip(buffer, axis=0)
    return buffer