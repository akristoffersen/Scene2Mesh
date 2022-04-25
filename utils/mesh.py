import torch
import torch.nn as nn
import numpy as np
import plyfile

# import bpy


def mesh_centroid(vertices, faces):
    # really should be volume centroid, but this 
    # works for now. below is code i couldn't get 
    # to return meaningful stuff, not sure whats wrong.

    # mesh_volume = 0
    # temp = np.array([0.0,0.0,0.0])

    # for face in faces:
    #     v_a = vertices[int(face[0])]
    #     v_b = vertices[int(face[1])]
    #     v_c = vertices[int(face[2])]

    #     center = (v_a + v_b + v_c) / 4
    #     volume = np.dot(v_a, np.cross(v_b, v_c)) / 6.0
    #     mesh_volume += volume
    #     temp = center * volume

    # mesh_center = temp / mesh_volume
    # return mesh_center

    return np.average(vertices, axis=0)


def triangle_direction_intersection(tri, trg):
    '''
    Finds where an origin-centered ray going in direction trg intersects a triangle.
    Args:
        tri: 3 X 3 vertex locations. tri[0, :] is 0th vertex.
    Returns:
        alpha, beta, gamma
    '''
    p0 = np.copy(tri[0, :])
    # Don't normalize
    d1 = np.copy(tri[1, :]) - p0
    d2 = np.copy(tri[2, :]) - p0
    d = trg / np.linalg.norm(trg)

    mat = np.stack([d1, d2, d], axis=1)

    try:
      inv_mat = np.linalg.inv(mat)
    except np.linalg.LinAlgError:
      return False, 0

    # inv_mat = np.linalg.inv(mat)
    
    a_b_mg = -1*np.matmul(inv_mat, p0)
    is_valid = (a_b_mg[0] >= 0) and (a_b_mg[1] >= 0) and ((a_b_mg[0] + a_b_mg[1]) <= 1) and (a_b_mg[2] < 0)
    if is_valid:
        return True, -a_b_mg[2]*d
    else:
        return False, 0


def load_obj(filename_obj, normalization=True, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1

    # load textures
    # textures = None
    # if load_texture:
    #     for line in lines:
    #         if line.startswith('mtllib'):
    #             filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
    #             textures = load_textures(filename_obj, filename_mtl, texture_size,
    #                                      texture_wrapping=texture_wrapping,
    #                                      use_bilinear=use_bilinear)
    #     if textures is None:
    #         raise Exception('Failed to load textures.')

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    return vertices, faces # , textures


def numpy_to_ply(vertices, faces, vert_colors=None, filename=None):
    '''
    Converts numpy arrays to PLY file, which can be used
    in blender or converted into an obj file.
    Args:
        # vertices: (|V|, 3), <x,y,z>
        # faces: (|F|, 3), vertex indices, 3 for each triangle
        # vert_colors (optional): (|V|, 3) RGB, [0, 1] range
        # filename: string. if not None, write to given filepath.
    Returns:
        PlyData obj
    '''
    assert vertices.shape[1] == 3 and faces.shape[1] == 3
    vertices = np.array(
        [tuple(vertices[i]) for i in range(vertices.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
    )
    faces = np.array(
        [tuple([faces[i]]) for i in range(faces.shape[0])],
        dtype=[('vertex_indices', 'i4', (3,))],
    )
    ply_data = plyfile.PlyData(
        [
            plyfile.PlyElement.describe(
                vertices, 
                'vertex',
                comments=['model vertices'],
            ),
            plyfile.PlyElement.describe(
                faces,
                'face',
            ),
        ],
    )
    if filename:
        ply_data.write(filename)
    
    return ply_data

# import neural_renderer as nr

# class Mesh(object):
#     '''
#     A simple class for creating and manipulating trimesh objects
#     '''
#     def __init__(self, vertices, faces, textures=None, texture_size=4):
#         '''
#         vertices, faces and textures(if not None) are expected to be Tensor objects
#         '''
#         self.vertices = vertices
#         self.faces = faces
#         self.num_vertices = self.vertices.shape[0]
#         self.num_faces = self.faces.shape[0]

#         # create textures
#         if textures is None:
#             shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
#             self.textures = nn.Parameter(0.05*torch.randn(*shape))
#             self.texture_size = texture_size
#         else:
#             self.texture_size = textures.shape[0]

#     @classmethod
#     def fromobj(cls, filename_obj, normalization=True, load_texture=False, texture_size=4):
#         '''
#         Create a Mesh object from a .obj file
#         '''
#         if load_texture:
#             vertices, faces, textures = nr.load_obj(filename_obj,
#                                                     normalization=normalization,
#                                                     texture_size=texture_size,
#                                                     load_texture=True)
#         else:
#             vertices, faces = nr.load_obj(filename_obj,
#                                           normalization=normalization,
#                                           texture_size=texture_size,
#                                           load_texture=False)
#             textures = None
#         return cls(vertices, faces, textures, texture_size)