import numpy as np
import open3d as o3d

# from sklearn.decomposition import PCA
# import pandas as pd

input_path="pointCloudSamples/"
output_path="mesh/"
dataname="sample_w_normals.xyz" #".xyz"

def process(n):
    point_cloud= np.loadtxt(input_path+dataname, delimiter=' ',skiprows=1) #of pts by xyzrgb, [76---, 6]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)

    if len(point_cloud[0]) == 6:
        # pcd.estimate_normals() #estimate_normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30)) #search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30)
        pcd.orient_normals_consistent_tangent_plane(n)
        # pcd.orient_normals_to_align_with_direction()
        # pcd.orient_normals_towards_camera_location()

    else:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])

    pcd, avg = test_normals(pcd)
    # avg = 0

    # Optional Visualization Strategy
    o3d.visualization.draw_geometries([pcd])

    # o3d.io.write_point_cloud(output_path + "pc.obj", pcd)

    return pcd, avg

def process_pc(pcd):
    newpcd = pcd.voxel_down_sample(voxel_size=0.05) #voxel_size=10
    print(pcd)
    print("--")
    print(newpcd)
    return newpcd

def test_normals(pcd):
    #step 0) randomly sample points in cloud
    pts = np.array(pcd.points)
    normals = np.array(pcd.normals)
    N = pts.shape[0]

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    ratios = []
    for samp in range(100): #random samples
        # print("Sample: ", samp)
        rind = np.random.randint(0, N)
        rpt = pts[rind]
        rnorm = normals[rind]

        # find X closest points to random points
        max_search = avg_dist * 10
        neighbors = 0
        count = 0
        for i in pts:
            if dist(rpt, i) < max_search:
                if inside_region(rpt, rnorm, max_search / 3, i):
                    count += 1
                neighbors += 1

        ratio = count / neighbors
        ratios.append(ratio)

    avg = np.mean(ratios)

    return pcd, avg

def inside_region(c, major_dir, offset, pt):

    offset_major = major_dir * offset

    plane1_pt = np.add(c, offset_major)
    plane2_pt = np.add(c, -0.7*offset_major)
    diff1 = np.subtract(pt, plane1_pt)
    diff2 = np.subtract(pt, plane2_pt)

    if np.dot(major_dir, diff1) < 0 and np.dot(major_dir, diff2) > 0:
        return True

    return False


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

# BPA
def BPA(pcd, n):
    print("---Beginning BPA")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))

    # Downsample number of triangles
    dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

    # Optional - Eliminate weird artifacts
    # dec_mesh.remove_degenerate_triangles()
    # dec_mesh.remove_duplicated_triangles()
    # dec_mesh.remove_duplicated_vertices()
    # dec_mesh.remove_non_manifold_edges()

    # dec_mesh = dec_mesh.filter_smooth_simple(number_of_iterations=5) #1
    # dec_mesh = dec_mesh.filter_smooth_laplacian(number_of_iterations=50) #10
    # dec_mesh = dec_mesh.filter_smooth_taubin(number_of_iterations=100) #10

    o3d.visualization.draw_geometries([dec_mesh])

    # Export
    print("---Exporting Mesh")
    o3d.io.write_triangle_mesh(output_path+"ex_bpa_chair_mesh_" + str(n) + ".obj", dec_mesh)
    return "Exported"

# Poisson
def poisson(pcd):
    print("---Beginning Poisson")
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

    # Cropping to clean unwanted elements
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    o3d.visualization.draw_geometries([poisson_mesh])

    o3d.io.write_triangle_mesh(output_path+"p_mesh_c.obj", p_mesh_crop)

# Alpha Shapes
def alpha(pcd):
    print("---Beginning Alpha Shapes")
    # Determine alpha imperically 
    alpha = 1.5
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


    o3d.io.write_triangle_mesh(output_path+"alpha_mesh.obj", mesh)

if __name__ == "__main__":
    # for i in range(4, 12):
    #     print("N neighbors: ", i)
    #     pcd, avg = process(i)
    #     print("AVG: ", avg)
    #     BPA(pcd, i)

    #


    #Chair
    # pcd, avg = process(3)
    # BPA(pcd, 3)
    # print("AVG: ", avg)

    # pcd, avg = process(7)
    # BPA(pcd, 7)
    # print("AVG: ", avg)

    # pcd, avg = process(9)
    # BPA(pcd, 9)
    # print("AVG: ", avg)

    #Gum
    # pcd, avg = process(11)
    # BPA(pcd, 11)
    # pcd, avg = process(8)
    # BPA(pcd, 8)



