import pymeshlab
import numpy
import os
def samples_absolute_path():
    path_sample = os.path.dirname(os.path.abspath(__file__))
    path_sample = os.path.dirname(path_sample)
    return path_sample


def test_output_path(folder_name):
    path_sample = samples_absolute_path()
    output_path = os.path.join(path_sample,folder_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path


def custom_mesh_element(s,num,input_path,output_path):
    print('\n')
    #ms.generate_sampling_element
    base_path = os.path.splitext(input_path)[0]
    base_name = os.path.basename(base_path)
    # create a new MeshSet
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    #default_params = ms.filter_parameter_values('generate_sampling_element')
    # sampling : str = 'Vertex' (or int = 0)  Choose what mesh element has to be used for the subsampling
    #'Vertex' = 0
    #'Edge' = 1
    #'Face' = 2
    if s == 'Vertex' or s == 0:
        print('vertex is choosen for this mesh element:', s)
    if s == 'Edge' or s == 1:
        print('edge is choosen for this mesh element:', s)
    if s == 'face' or s == 2:
        print('face is choosen for this mesh element:', s) 
    # samplenum : int = 0,The desired number of elements that must be chosen 
    print('the desired number of element chosen is ', num)
    ms.generate_sampling_element(sampling=s,samplenum=num)
    #print('face is choosen for this mesh element:', default_params['sampling']) 
    #print('the desired number of element chosen is ',default_params['samplenum'])
    output_file = output_path +'/' + base_name + '_' + str(num) +'.obj'
    ms.save_current_mesh(output_path +'/' + base_name + '_' + str(num) +'.obj')
    return output_file

def surface_reconstruction_screened_poisson(input_path,output_path):
#Compute the normals of the vertices of a mesh without exploiting the triangle connectivity,
#useful for dataset with no faces
#k : int = 10,Neighbour num: The number of neighbors used to estimate normals
#smoothiter : int = 0,The number of smoothing iteration done on the p used to estimate and propagate normal
#flipflag : bool = False
#viewpos : numpy.ndarray[numpy.float64[3]] = [0, 0, 0]

    print('\n')
    base_path = os.path.splitext(input_path)[0]
    base_name = os.path.basename(base_path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    default_params = ms.filter_parameter_values('compute_normal_for_point_clouds')
    ms.compute_normal_for_point_clouds(k=10,smoothiter=0,flipflag=False, viewpos = [0,0,1])
    ms.generate_surface_reconstruction_screened_poisson()
    output_file = output_path +'/' + base_name + '_mesh_' +'.obj'
    ms.save_current_mesh(output_path +'/' + base_name + '_mesh_' +'.obj')
    return output_file