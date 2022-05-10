import numpy as np
import json
import os
import cv2

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
    camera_angle_y = None
    if "camera_angle_y" in chair_json:
        camera_angle_y = chair_json['camera_angle_y']
    return result, camera_angle_x, camera_angle_y

def undistort(im):
    H, W, _ = im.shape
    k1 = -0.0095054676697
    k2 = 0.00245190
    p1 = 0.0021999729
    p2 = 0.00064887
    dist_coeffs = np.array([k1, k2, p1, p2])
    camera_mat = np.array(
        [
            [594.77387449, 0.0, 359.3926146],
            [0.0, 593.33534984512, 645.9771252],
            [0.0, 0.0, 1.0],
        ],
    )
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coeffs, (W, H), 1, (W, H))
    return cv2.undistort(im, camera_mat, dist_coeffs, None, newcameramtx), newcameramtx


def sample_xyzs(im, c2w, camera_angle_x, xyzs, camera_angle_y=None, distort=False):
    H, W, = im.shape[0], im.shape[1]
    w2c = np.linalg.inv(c2w)
    homo_xyzs = np.hstack([xyzs, np.ones((xyzs.shape[0],1))])
    a = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    a = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [+1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    if camera_angle_y is None:
        camera_angle_y = camera_angle_x
    verts_camera_space = w2c @ a @ homo_xyzs.T
    verts_camera_space[:2] /= verts_camera_space[2]
    px_coords = (
        verts_camera_space[:2].T * \
        [calc_focal(camera_angle_x, W), calc_focal(camera_angle_y, H)]
    )
    px_coords += [H / 2, W / 2]
    # print(px_coords[:4])
    # im, new_cam_K = undistort(im)

    # px_coords = (
    #     verts_camera_space[:2].T * \
    #     [utils.nerf_utils.calc_focal(camera_angle_x, W), utils.nerf_utils.calc_focal(camera_angle_y, H)]
    # )
    # px_coords += [359.3, 645.97] # [W / 2, H / 2]
    # px_coords = (
    #     verts_camera_space[:2].T * [new_cam_K[0, 0], new_cam_K[1, 1]]
    # )
    # px_coords += new_cam_K[:2, 2]


    px_coords[:, 0] = W - px_coords[:, 0] # swap
    px_coords[:, [1, 0]] = px_coords[:, [0, 1]]
    y, x = px_coords[:, 0].astype(int), px_coords[:, 1].astype(int)
    mask = (y >= H) + (y < 0) + (x >= W) + (x < 0)
    result = np.zeros((len(y), 3))
    result[~mask] = im[y[~mask], x[~mask]]
    return result, mask


def find_best_im(c2ws, camera_angle_x, xyzs, ims=None, depth_ims=None):
    N = c2ws.shape[0]

    dists = []
    # idxs = [77, 100, 123, 154, 181, 214, 303, 404, 419, 465]
    idxs = list(range(len(c2ws)))
    for i in idxs: # range(N - 100, N):
        if depth_ims is not None:
            print(depth_ims[i].shape)
            sample_depths = sample_xyzs(depth_ims[i], c2ws[i], camera_angle_x, xyzs)
            # average depth_ims
            avg_dist = np.average(sample_depths)
        else:
            camera_center = c2ws[i, :3, 3]
            avg_dist = np.average(np.linalg.norm(xyzs - camera_center, axis=1))

        # print(i, avg_depth)
        dists.append(avg_dist)

    indices = list(range(len(dists)))
    sorted_indices = sorted(indices, key= lambda i: dists[i])

    best = sorted_indices[5]
    dist = dists[best]

    # print("best:", best, dist)
    return idxs[best]

