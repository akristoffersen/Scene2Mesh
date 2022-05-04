import numpy as np
import json
import os

def calc_focal(camera_angle_x, w):
    return 0.5 * w / np.tan(0.5 * camera_angle_x)

def create_c2ws(json_chair_fn):
    with open(json_chair_fn) as f:   
        chair_json = json.load(f)
    N = len(np.array(chair_json['frames']))
    result = np.zeros((N, 4, 4))
    for i in range(N):
        result[i] = np.array(chair_json['frames'][i]['transform_matrix'])
    camera_angle_x = chair_json['camera_angle_x']
    return result, camera_angle_x


def find_best_im(c2ws, camera_angle_x, ims, depth_ims, xyzs):
    N = c2ws.shape[0]
    H, _, _ = ims.shape[1:] # assumes square ims
    best = -1
    dist = -1
    homo_xyzs = np.hstack([xyzs, np.ones((3,1))])

    for i in range(N):
        c2w = c2ws[i]
        w2c = np.linalg.inv(c2w)

        verts_camera_space = w2c @ homo_xyzs.T
        verts_camera_space[:2] /= verts_camera_space[2]
        px_coords = (verts_camera_space * calc_focal(camera_angle_x, H))[:2].T

        # adding principles
        px_coords += H / 2

        # converting to opencv coords
        px_coords[:, 0] = H - px_coords[:, 0] # swap
        px_coords[:, [1, 0]] = px_coords[:, [0, 1]]

        # average depth_ims

        avg_depth = (
            depth_ims[i, int(px_coords[0, 0]), int(px_coords[0, 1])] + \
            depth_ims[i, int(px_coords[1, 0]), int(px_coords[1, 1])] + \
            depth_ims[i, int(px_coords[1, 0]), int(px_coords[1, 1])]
        ) / 3.0
        print(i, avg_depth)
        if avg_depth > dist:
            best = i
            dist = avg_depth
    print("best:", best, dist)
    return best

