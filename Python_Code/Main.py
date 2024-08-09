#  Copyright (C) 2024
#  
#  Amin Nadimy, Boyang Chen, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#  ++++++++++++++++++++++++++++++++++++++++
#  amin.nadimy19@imperial.ac.uk
#  
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

# #-- Import general libraries
import numpy as np 
import time 
import torch
import matplotlib.pyplot as plt
import NN4PDE_lib as an
import Models as md
from torch.multiprocessing import Process, set_start_method

device = an.get_device()

    # # # ################################### # # #
    # # # ######         Model          ##### # # #
    # # # ################################### # # #
def run_AI4SK_L_semi_serial(model, device, sd_list, dt, ntime, diag, epsilon_k, iteration, nlevel, ub, Re, n_out, save_fig):
    # model, sd_list, dt, ntime, diag, epsilon_k, iteration, cuda_indx = args
    # compiled_model = torch.compile(model)
    # dev = sd_list[0].values_u.device
    # cuda_indx = device.index
    torch.cuda.set_device(device.index)
    torch.cuda.init()
    an.init_process(device.index)
    istep = 1
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for itime in range(1,ntime+1):
                if (istep*dt) % n_out <= 0.001:
                    print('Time step:', itime, istep)

                for ele in sd_list:
                    _ = model(ele, dt, iteration, nlevel, ub, Re, diag)
                    if np.max(np.abs(ele.w[0].cpu().detach().numpy())) > 80000.0:
                        print('Not converged !!!!!!', ele.rank)
                        break

                istep +=1
            end = time.time()
        print('istep, time',istep, (end-start))


if __name__ == "__main__":
    # # # ################################### # # #
    # # # ######   Physical parameters ###### # # #
    # # # ################################### # # #
    global_nx =                                                            # Global nodes in x-direction
    global_ny =                                                           # Global nodes in y-direction
    global_nz =                                                             # Global nodes in z-direction
    
    no_domains_x =                                                            # number of subdomains in x-direction
    no_domains_y =                                                            # number of subdomains in y-direction
    no_domains_z =                                                            # number of subdomains in z-direction

    # # # ################################### # # #
    # # # ######  Numerical parameters ###### # # #
    # # # ################################### # # #
    ntime = 100                                                                # Time steps
    Re =                                                                    # Petrov-Galerkin factor
    dt = 
    ub =                                                                   # Inflow velocity
    n_out =                                                                  # Results output
    iteration =                                                               # Number of multigrid iteration
    save_fig = True                                                            # Save results
    epsilon_k =                                                           # Stablisatin factor in Petrov-Galerkin for velocity
    ###############################################
    sd_indx_with_lower_res =                                                 # Subdomain indices with lower resolution
    sd , dx, nlevel = an.generate_subdomains(global_nx, global_ny, global_nz, no_domains_x, no_domains_y, no_domains_z, no_domains, sd_indx_with_lower_res, path)

    sd_list0 =                                            # Subdomains for process 0
    sd_list1 =                                            # Subdomains for process 1
    # # # ################################### # # #
    # # # ######         Model          ##### # # #
    # # # ################################### # # #
    w1, wA, w2, w3, w4, w_res, bias_initializer, diag = an.generate_CNN_weights(dx)

    model = [md.AI4Urban(w1, wA, w2, w3, w4, w_res, bias_initializer).to(device[0]), 
             md.AI4Urban(w1, wA, w2, w3, w4, w_res, bias_initializer).to(device[1])]

    # # # ################################### # # #
    # # # ###### Running the simulation ##### # # #
    # # # ################################### # # #
    # define the processes
    p0 = Process(target=run_AI4SK_L_semi_serial, args=(model[0], device[0], sd_list0, dt, ntime, diag, epsilon_k, iteration, nlevel, ub, Re, n_out, save_fig))
    p1 = Process(target=run_AI4SK_L_semi_serial, args=(model[1], device[1], sd_list1, dt, ntime, diag, epsilon_k, iteration, nlevel, ub, Re, n_out, save_fig))

    print('Process 0 started')
    p0.start()
    print('Process 1 started')
    p1.start()
    print('Processes started')
    p0.join()
    print('Process 0 joined')
    p1.join()
    print('Process 1 joined')

    # args = [(model[0], sd_list0, dt, ntime, diag, epsilon_k, iteration,0), (model[1], sd_list1, dt, ntime, diag, epsilon_k, iteration,1)]
    # with Pool(2) as p:
    #     print('Pool started')
    #     p.map(run_AI4SK_L_semi_serial, args)
    #     print('Pool joined')
