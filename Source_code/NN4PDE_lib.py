# this is for linear semi South Kensigtom case in 3D

import torch
import vtk
import copy
import os
import torch.distributed as dist
import numpy as np 
import math

class subdomain_3D:
    def __init__(self, nx, ny, nz, dx, dy, dz, nlevel, rank):
        self.nx = torch.tensor(nx)
        self.ny = torch.tensor(ny)
        self.nz = torch.tensor(nz)
        self.dx = torch.tensor(dx)
        self.dy = torch.tensor(dy)
        self.dz = torch.tensor(dz)

        self.nlevel = nlevel
        self.rank = rank
        self.neig = ['side' for _ in range(6)]
        self.corner_neig = None
        self.corner_node_neig = None

        input_shape = (1, 1, nz, ny, nx)
        # if you want numpy arrays use thses lines instead
        self.values_u = torch.zeros(input_shape)
        self.values_v = torch.zeros(input_shape)
        self.values_w = torch.zeros(input_shape)
        self.values_p = torch.zeros(input_shape)
        self.k_u      = torch.zeros(input_shape)
        self.k_v      = torch.zeros(input_shape)
        self.k_w      = torch.zeros(input_shape)
        self.k_x      = torch.zeros(input_shape)
        self.k_y      = torch.zeros(input_shape)
        self.k_z      = torch.zeros(input_shape)
        self.b        = torch.zeros(input_shape)
        self.k1       = torch.ones(input_shape)*2.0

        input_shape_pad = (1,1,nz+2,ny+2,nx+2)
        self.values_uu  = torch.zeros(input_shape_pad)
        self.values_vv  = torch.zeros(input_shape_pad)
        self.values_ww  = torch.zeros(input_shape_pad)
        self.values_pp  = torch.zeros(input_shape_pad)
        self.b_uu = torch.zeros(input_shape_pad)
        self.b_vv = torch.zeros(input_shape_pad)
        self.b_ww = torch.zeros(input_shape_pad)
        self.k_uu = torch.zeros(input_shape_pad)
        self.k_vv = torch.zeros(input_shape_pad)
        self.k_ww = torch.zeros(input_shape_pad)

        self.halo_u = [
            torch.zeros((ny+2, nx+2)),   # front
            torch.zeros((ny+2, nx+2)),   # back
            torch.zeros((nz+2, ny+2)),   # right
            torch.zeros((nz+2, ny+2)),   # left
            torch.zeros((nz+2, nx+2)),   # top
            torch.zeros((nz+2, nx+2))    # bottom
        ]
        self.halo_v    = copy.deepcopy(self.halo_u)
        self.halo_w    = copy.deepcopy(self.halo_u)
        self.halo_p    = copy.deepcopy(self.halo_u)
        self.halo_c    = copy.deepcopy(self.halo_u)
        self.halo_b_u  = copy.deepcopy(self.halo_u)
        self.halo_b_v  = copy.deepcopy(self.halo_u)
        self.halo_b_w  = copy.deepcopy(self.halo_u)
        self.halo_k_uu = copy.deepcopy(self.halo_u)
        self.halo_k_vv = copy.deepcopy(self.halo_u)
        self.halo_k_ww = copy.deepcopy(self.halo_u)

        self.halo_PGu = [
            torch.zeros((ny, nx)),   # front
            torch.zeros((ny, nx)),   # back
            torch.zeros((nz, ny)),   # right
            torch.zeros((nz, ny)),   # left
            torch.zeros((nz, nx)),   # top
            torch.zeros((nz, nx))    # bottom
        ]
        self.halo_PGv = copy.deepcopy(self.halo_PGu)
        self.halo_PGw = copy.deepcopy(self.halo_PGu)
        self.halo_mgw = []

        self.sigma     = torch.zeros(input_shape)
        self.pad_sigma = torch.zeros((1, 1, nz+2, ny+2, nx+2))

        # PG variables
        self.kmax   = None
        self.m_i    = None
        self.pg_cst = None



    # sends all attrebutes to device
    def to(self, device):
        for attr in self.__dict__:
            if torch.is_tensor(self.__dict__[attr]):
                self.__dict__[attr] = self.__dict__[attr].to(device)
            elif isinstance(self.__dict__[attr], list) and all(torch.is_tensor(t) for t in self.__dict__[attr]):
                self.__dict__[attr] = [t.to(device) for t in self.__dict__[attr]]

        # create w tensors for multigid levels for each subdomains
        zz = self.nz; yy = self.ny; xx = self.nx
        self.w = []
        self.halo_MG = []
        for i in range(self.nlevel-1):
            self.w.append(torch.zeros((1,1,zz,yy,xx),device=device))
            
            self.halo_MG.append([
                torch.zeros((yy+2, xx+2)).to(device),   # front
                torch.zeros((yy+2, xx+2)).to(device),   # back
                torch.zeros((zz+2, yy+2)).to(device),   # right
                torch.zeros((zz+2, yy+2)).to(device),   # left
                torch.zeros((zz+2, xx+2)).to(device),   # top
                torch.zeros((zz+2, xx+2)).to(device)    # bottom
            ])
            zz//=2; yy//=2; xx//=2  
        return self


def generate_subdomains(global_nx, global_ny, global_nz, no_domains_x, no_domains_y, no_domains_z, no_domains, sd_indx_with_lower_res, path):
    sub_nx = global_nx//no_domains_x
    sub_ny = global_ny//no_domains_y
    sub_nz = global_nz//no_domains_z

    nx = [sub_nx]*no_domains; ny = [sub_ny]*no_domains; nz = [sub_nz]*no_domains
    dx = [1.0]*no_domains; dy = [1.0]*no_domains; dz = [1.0]*no_domains

    res_ratios = [0.5 if i in sd_indx_with_lower_res else 1 for i in range(no_domains)]
    dx = [val/res_ratios[i] for i, val in enumerate(dx)]
    dy = dx
    # dz stays the same as we always have 1 sd in z-dir
    nx = [int(val/np.floor_divide(1,res_ratios[i])) for i, val in enumerate(nx)]
    ny = [int(val/np.floor_divide(1,res_ratios[i])) for i, val in enumerate(ny)]

    print('ratios:', res_ratios)
    print('nx    :', nx)
    print('ny    :', ny)
    print('nz    :', nz)
    print('dx    :', dx)

    nlevel = int(math.log(min(min(nx), min(ny), min(nz)), 2)) + 1  
    print('How many levels in multigrid:', nlevel)


    # Defines subdomains
    sd = [subdomain_3D(nx[i], ny[i], nz[i], dx[i], dy[i], dz[i], nlevel, i) for i in range(no_domains)]

    # Set the inner neigs correctly
    set_face_neigs(sd, no_domains_x, no_domains_y)

          
    ################# Only for IBM ######################## Immersed Boundary Method
    mesh = np.load('the mesh file')
    mesh = torch.where(mesh == 0, 1.0e09, 0.0)

    # Split along the y-axis
    subdomains_y = np.array_split(mesh, no_domains_y, axis=1)

    # For each subdomain along the y-axis, split along the x-axis
    subdomains_xy = [np.array_split(subdomain, no_domains_x, axis=2) for subdomain in subdomains_y]

    # save subdomains in a list with corect order starting from lower left
    mesh_list = []
    for j in range(no_domains_y):
        for i in range(no_domains_x):
            mesh_list.append(subdomains_xy[j][i])

    # ----------------------------- save meshes inso sd.sigma
    for ele in range(no_domains):
        sd[ele].sigma[0,0,:,:,:] = mesh_list[ele]
    return sd, dx, nlevel


def get_device():
    # Define the device
    if torch.cuda.is_available():
        num_gpu_devices = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_name(i) for i in range(num_gpu_devices)]

        print(f"Number of available GPU devices: {num_gpu_devices}")
        device = []
        for i, device_name in enumerate(device_names):
            device.append(torch.device(f"cuda:{i}"))
            print(f"GPU {i}: {device_name}, {device[i]}")
            
    else:
        device = 'cpu'
        print("No GPU devices available. Using CPU.")
    return device


def init_process(cuda_indx):
    torch.cuda.set_device(cuda_indx)
    """Initialisation"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12369'
    # Initialisation
    dist.init_process_group(backend='nccl', init_method='env://', world_size=2, rank=cuda_indx)

    
def init_grid_levels(nlevel, dummy):
    w = []
    for i in range(nlevel-1):
        w.append(torch.zeros((1,dummy, dummy, dummy, 1)))
        dummy //= 2
    return w


def save_data(sd,n_out,itime, w):
    # if w means if it is 3D which can be True or False
    if itime % n_out == 0:  
        np.save("result_bluff_body/result_3d_BC/u"+str(itime), arr = sd.values_u[0,:,:,:,0])
        np.save("result_bluff_body/result_3d_BC/v"+str(itime), arr = sd.values_v[0,:,:,:,0])
        np.save("result_bluff_body/result_3d_BC/p"+str(itime), arr = sd.values_p[0,:,:,:,0])
        if w:
            np.save("result_bluff_body/result_3d_BC/w"+str(itime), arr = sd.values_w[0,:,:,:,0])


def generate_CNN_weights(dx):
    # # # ################################### # # #
    # # # ######    Linear Filter      ###### # # #
    # # # ################################### # # #
    bias_initializer = torch.tensor([0.0])
    # Laplacian filters
    pd1 = torch.tensor([[2/26, 3/26, 2/26],
                        [3/26, 6/26, 3/26],
                        [2/26, 3/26, 2/26]])
    pd2 = torch.tensor([[3/26, 6/26, 3/26],
                        [6/26, -88/26, 6/26],
                        [3/26, 6/26, 3/26]])
    pd3 = torch.tensor([[2/26, 3/26, 2/26],
                        [3/26, 6/26, 3/26],
                        [2/26, 3/26, 2/26]])
    w1 = torch.zeros([1, 1, 3, 3, 3])
    wA = torch.zeros([1, 1, 3, 3, 3])
    w1[0, 0, 0,:,:] = pd1
    w1[0, 0, 1,:,:] = pd2
    w1[0, 0, 2,:,:] = pd3
    wA[0, 0, 0,:,:] = -pd1
    wA[0, 0, 1,:,:] = -pd2
    wA[0, 0, 2,:,:] = -pd3
    # Gradient filters
    p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
                             [-0.056, 0.0, 0.056],
                             [-0.014, 0.0, 0.014]])
    p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
                             [-0.22, 0.0, 0.22],
                             [-0.056, 0.0, 0.056]])
    p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
                             [-0.056, 0.0, 0.056],
                             [-0.014, 0.0, 0.014]])
    p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],
                             [0.0, 0.0, 0.0],
                             [-0.014, -0.056, -0.014]])
    p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],
                             [0.0, 0.0, 0.0],
                             [-0.056, -0.22, -0.056]])
    p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],
                             [0.0, 0.0, 0.0],
                             [-0.014, -0.056, -0.014]])
    p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
                             [0.056, 0.22, 0.056],
                             [0.014, 0.056, 0.014]])
    p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]])
    p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
                             [-0.056, -0.22, -0.056],
                             [-0.014, -0.056, -0.014]])
    w2 = torch.zeros([1,1,3,3,3])
    w3 = torch.zeros([1,1,3,3,3])
    w4 = torch.zeros([1,1,3,3,3])
    w2[0,0,0,:,:] = -p_div_x1
    w2[0,0,1,:,:] = -p_div_x2
    w2[0,0,2,:,:] = -p_div_x3
    w3[0,0,0,:,:] = -p_div_y1
    w3[0,0,1,:,:] = -p_div_y2
    w3[0,0,2,:,:] = -p_div_y3
    w4[0,0,0,:,:] = -p_div_z1
    w4[0,0,1,:,:] = -p_div_z2
    w4[0,0,2,:,:] = -p_div_z3
    # Restriction filters
    w_res = torch.zeros([1,1,2,2,2])
    w_res[0,0,:,:,:] = 0.125
    diag = wA[0,0,1,1,1]
    return w1, wA, w2, w3, w4, w_res, bias_initializer, diag


def set_face_neigs(sd, width, height):
    '''
    becasue z direction has only 1 subdomain, here we do not consider it
    it defines all neighbours for each subdomains. 
    side: when it is on the domain bc
    inlet: when it is at the inlet
    outlet: when it is at the outlet
    sd nfo: when there is another sd as a neig
    sd: is a list of all subdomains
    width: number of domains in x direction
    height: number of domains in y direction
    
    # Example: Accessing the neigs of sd[3]
    ele = 1
    for iface in range(6):
        if not isinstance(sd[ele].neig[iface], str):
            print(sd[ele].neig[iface].rank)
        else:
            print(sd[ele].neig[iface])
    neig order:: front, back, left, right, top and bottom
    '''
    for i in range(len(sd)):
        rank = sd[i].rank

        # Check if the current domain is on the bottom row
        if rank - width < 0:
            sd[i].neig[4] = 'side'
        else:
            sd[i].neig[4] = sd[rank - width]

        # Check if the current domain is on the leftmost column
        if rank % width == 0:
            sd[i].neig[2] = 'inlet' # 'left'
        else:
            sd[i].neig[2] = sd[rank - 1]

        # Check if the current domain is on the rightmost column
        if (rank+1) % width == 0:
            sd[i].neig[3] = 'outlet' # 'right'
        else:
            sd[i].neig[3] = sd[rank + 1]

        # Check if the current domain is on the top row
        if rank >= width*(height-1):
            sd[i].neig[5] = 'side'
        else:
            sd[i].neig[5] = sd[rank + width]


def set_corner_neighbors(sd,no_subdomains_x, no_subdomains_y):
    '''
    It works only if num_ele in z-dir is 1. otherwise you should modify it
    it find corner neig of each side. to find corner neig of each corner node, you should use another function
    for 3D lets say the neig for face 0 locates in the front of 3D cube, then, corner neig 0 will be neig 2
    of the face neig 0 which is at its left side. In this case a 3D cube has 4 corner neig only if there is
    no subdomain in z-dir. If there is subdomain in z-dir, then you should modify this function.
    '''
    for ele in range(len(sd)):
         # Calculate the row and column indices from the element number
        i = ele // no_subdomains_x
        j = ele % no_subdomains_x

        sd[ele].corner_neig = ['side'] * 4
        sd[ele].corner_neig[0] = sd[((i - 1) * no_subdomains_x + (j - 1))] if i > 0 and j > 0 else 'side'  # Top-left neighbor
        sd[ele].corner_neig[1] = sd[((i - 1) * no_subdomains_x + (j + 1))] if i > 0 and j < no_subdomains_x - 1 else 'side'  # Top-right neighbor
        sd[ele].corner_neig[2] = sd[((i + 1) * no_subdomains_x + (j - 1))] if i < no_subdomains_y - 1 and j > 0 else 'side'  # Bottom-left neighbor
        sd[ele].corner_neig[3] = sd[((i + 1) * no_subdomains_x + (j + 1))] if i < no_subdomains_y - 1 and j < no_subdomains_x - 1 else 'side'  # Bottom-right neighbor


def describe(sd):
    print(f'------------------------------------- description of sd{sd.rank} ---------------------------------------------------')
    print('nx,ny,nz:         ', sd.nx,sd.ny,sd.nz)
    print('dx, dy, dz:', sd.dx, sd.dy, sd.dz)
    print('nlevel:           ', sd.nlevel)
    print('sd.w shape:       ', np.shape(sd.values_w))
    print('neig:             ', sd.neig)
    print('corner neig:      ', sd.corner_neig)
    print('corner node neig: ', sd.corner_node_neig)
    print(f'----------------------------------------------------------------------------------------------------------------------------')


def generate_vtu_file(points, cells, values, output_file):
    # Create a VTK unstructured grid
    unstructured_grid = vtk.vtkUnstructuredGrid()

    # Set points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    unstructured_grid.SetPoints(vtk_points)

    # Set cell connectivity
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        vtk_cells.InsertNextCell(len(cell), cell)
    unstructured_grid.SetCells(vtk.VTK_QUAD, vtk_cells)

    # Set point values
    vtk_data = vtk.vtkFloatArray()
    for value in values:
        vtk_data.InsertNextValue(value)
    unstructured_grid.GetPointData().SetScalars(vtk_data)

    # Write the VTU file
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    











