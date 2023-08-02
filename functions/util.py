# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 17:04:53 2022

@author: Azhad
"""
import os 
import open3d as o3d
import numpy as np


def get_file_tuple_ply(folder_name):
    a_tuple = ()
    print(os.getcwd())
    current_dir = os.getcwd()
    i=0
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.ply'):
            f_name = os.path.basename(file_name)
            new_path = os.path.join(current_dir,folder_name,f_name)
            new_path = new_path.replace("\\","/")
            i+=1
            y = list(a_tuple)
            y.append(new_path)
            a_tuple = tuple(y)
    return a_tuple

def get_file_tuple_xyz(folder_name):
    a_tuple = ()
    print(os.getcwd())
    current_dir = os.getcwd()
    i=0
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.xyz'):
            f_name = os.path.basename(file_name)
            new_path = os.path.join(current_dir,folder_name,f_name)
            new_path = new_path.replace("\\","/")
            i+=1
            y = list(a_tuple)
            y.append(new_path)
            a_tuple = tuple(y)
    return a_tuple

def get_file_tuple_lm3(folder_name):
    a_tuple = ()
    print(os.getcwd())
    current_dir = os.getcwd()
    i=0
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.lm3'):
            f_name = os.path.basename(file_name)
            new_path = os.path.join(current_dir,folder_name,f_name)
            new_path = new_path.replace("\\","/")
            i+=1
            y = list(a_tuple)
            y.append(new_path)
            a_tuple = tuple(y)
    return a_tuple

def get_file_tuple_bnt(folder_name):
    a_tuple = ()
    print(os.getcwd())
    current_dir = os.getcwd()
    i=0
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.bnt'):
            f_name = os.path.basename(file_name)
            new_path = os.path.join(current_dir,folder_name,f_name)
            new_path = new_path.replace("\\","/")
            i+=1
            y = list(a_tuple)
            y.append(new_path)
            a_tuple = tuple(y)
    return a_tuple

def get_file_tuple_obj(folder_name):
    a_tuple = ()
    print(os.getcwd())
    current_dir = os.getcwd()
    i=0
    for file_name in os.listdir(folder_name):
        if file_name.endswith('.obj'):
            f_name = os.path.basename(file_name)
            new_path = os.path.join(current_dir,folder_name,f_name)
            new_path = new_path.replace("\\","/")
            i+=1
            y = list(a_tuple)
            y.append(new_path)
            a_tuple = tuple(y)
    return a_tuple

def path_clean(new_path):
    new_path = new_path.replace("\\","/")
    return new_path

def preview_ply(pcd, preview ='y'):
    print("Load a ply point cloud, print it, and render it")
    #pcd.paint_uniform_color([1, 0.706, 0])
    print(pcd)
    print(np.asarray(pcd.points))
    if preview == 'y':
        o3d.visualization.draw_geometries([pcd])
    return pcd

def preview_obj(mesh, preview ='y'):
    print("Load a obj TriangleMesh, print it, and render it")
    #mesh.paint_uniform_color([1, 0.706, 0])
    print(mesh)
    print('Vertices:')
    print(np.asarray(mesh.vertices))
    print('Triangles:')
    print(np.asarray(mesh.triangles))
    print("Try to render a mesh with normals (exist: " +
    str(mesh.has_vertex_normals()) + ") and colors (exist: " +
    str(mesh.has_vertex_colors()) + ")")
    print("Computing normal and rendering it.")
    if preview == 'y':
        o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
    return mesh

def st_out_rm(pcd, preview = 'y'):
    print("Statistical oulier removal step")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors= 30,std_ratio= 3)
    if preview =='y':
        display_inlier_outlier(pcd, ind)
    pcd = pcd.select_by_index(ind)
    if preview =='y':
        o3d.visualization.draw_geometries([pcd])
    return pcd
    
def display_inlier_outlier(cloud, ind):
    #sub function
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return

def nose_tip_detection_ply(pcd, preview = 'y' ):
    print("find nose tip by depth z")
    xyz_load = np.asarray(pcd.points)
    z_load = xyz_load[:, 2]
    index_max_z = np.where(xyz_load == np.amax(z_load))
    index_max_z = index_max_z[0]
    if index_max_z.size > 1:
        index_max_z = np.reshape(index_max_z,(1,len(index_max_z)))
        index_max_z = index_max_z[0][0]
    maxElement_z = np.amax(z_load)
    print('Max element from Numpy Array : ', maxElement_z)
    x_max = xyz_load[index_max_z, 0]
    x_max = float(x_max)
    y_max = xyz_load[index_max_z, 1]
    y_max = float(y_max)
    list1 = index_max_z.tolist()
    if preview =='y':
        display_max_nose(pcd, list1, x_max, y_max, maxElement_z)
    return pcd,list1


def display_max_nose(cloud, ind, x, y, z):
    points = [
        [-60+x, -120+y, -120+z],
        [60+x, -120+y, -120+z],
        [-60+x, 80+y, -120+z],
        [60+x, 80+y, -120+z],
        [-60+x, -120+y, 10+z],
        [60+x, -120+y, 10+z],
        [-60+x, 80+y, 10+z],
        [60+x, 80+y, 10+z],
        ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        ]
    colors = [[0, 0, 1] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
        )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud,line_set])

def in_range_crop (pcd ,ind, x_range = 60, y_range_u=80, y_range_l = 120, z_range_u =10,z_range_l=80, preview = 'y'):
    
    xyz_load = np.asarray(pcd.points)
    x = xyz_load[ind, 0]
    x =float(x)
    y = xyz_load[ind, 1]
    y =float(y)
    z = xyz_load[ind, 2]
    z =float (z)
    
    print("crop from nose tip")
    #put range limit
    x_ulmt = x + x_range
    x_llmt = x - x_range
    y_ulmt = y + y_range_u
    y_llmt = y - y_range_l
    z_ulmt = z + z_range_u
    z_llmt = z - z_range_l
    #take out points into numpy
    xyz_load = np.asarray(pcd.points)
    x_load = xyz_load[:, 0]
    #cropped if out of range x_axis
    result = np.where(np.logical_and(x_load>=x_llmt, x_load<=x_ulmt))
    result = result[0]
    ind = result.tolist()
    pcd = pcd.select_by_index(ind)
    if preview =='y':
        o3d.visualization.draw_geometries([pcd])
    
    xyz_load = np.asarray(pcd.points)
    y_load = xyz_load[:, 1]
    #cropped if out of range y_axis
    result_1 = np.where(np.logical_and(y_load>=y_llmt, y_load<=y_ulmt))
    result_1 = result_1[0]
    ind = result_1.tolist()
    pcd = pcd.select_by_index(ind)
    if preview =='y':
        o3d.visualization.draw_geometries([pcd])
    
    xyz_load = np.asarray(pcd.points)
    z_load = xyz_load[:, 2]
    #cropped if out of range of z_axis
    result_2 = np.where(np.logical_and(z_load>=z_llmt, z_load<=z_ulmt))
    result_2 = result_2[0]
    ind = result_2.tolist()
    pcd = pcd.select_by_index(ind)
    if preview =='y':
        o3d.visualization.draw_geometries([pcd])
    
    return pcd

def safe_mkdir( file_dir ):
    if not os.path.exists( file_dir ):
        os.mkdir( file_dir ) 
        
def write_ply(pcd, filepath, output_folder, verbose=True, overwrite =True): 
    from os.path import exists
    i = 0
    safe_mkdir(output_folder)    
    current_dir = os.getcwd()
    f_name = os.path.basename(filepath)
    f_name = os.path.splitext(f_name)[0]
    f_name = f_name +".ply"
    new_path = os.path.join(current_dir,output_folder,f_name)
    if overwrite == False:
        while (os.path.exists(new_path)):
            i += 1
            num = '(' + str(i) + ')'
            f_name = os.path.splitext(f_name)[0]
            f_name = f_name + num +".ply"
            new_path = os.path.join(current_dir,output_folder,f_name)
        else: 
            o3d.io.write_point_cloud(new_path, pcd)
    if overwrite == True:
        o3d.io.write_point_cloud(new_path, pcd)
    if verbose:
        print('mesh saved to: ', new_path)
    return new_path

def write_obj(pcd, filepath, output_folder, verbose=True, overwrite = True): 
    i = 0
    safe_mkdir(output_folder)    
    current_dir = os.getcwd()
    f_name = os.path.basename(filepath)
    f_name = os.path.splitext(f_name)[0]
    f_name = f_name +".obj"
    new_path = os.path.join(current_dir,output_folder,f_name)
    if overwrite == False:
        while (os.path.exists(new_path)):
            i += 1
            num = '(' + str(i) + ')'
            f_name = os.path.splitext(f_name)[0]
            f_name = f_name + num +".obj"
            new_path = os.path.join(current_dir,output_folder,f_name)
        else: 
            o3d.io.write_triangle_mesh(new_path, pcd)
    if overwrite == True:
        o3d.io.write_triangle_mesh(new_path, pcd)
    if verbose:
        print('mesh saved to: ', new_path)
    return new_path

def down_sample(pcd,voxel_size=0.5, preview=True):
    print("Downsample the point cloud with a voxel of ", voxel_size)
    downpcd = pcd.voxel_down_sample(voxel_size)
    if preview:
        o3d.visualization.draw_geometries([downpcd])
    return downpcd

def est_normal(pcd,radius=10, max_nn=30, preview=True):
    print("Recompute the normal of the point cloud")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    if preview:
        o3d.visualization.draw_geometries([pcd],point_show_normal=True)
    return pcd

def mesh_reconstruction(pcd, preview=True,min_density=0.1):
    print('Run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, scale=1.1 )
    if preview:
        o3d.visualization.draw_geometries([mesh])
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    
    return mesh 
        