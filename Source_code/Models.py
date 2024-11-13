import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# # # ################################### # # #
# # # #########  AI4Urban MAIN ########## # # #
# # # ################################### # # #
class AI4Urban(nn.Module):
    """docstring for AI4Urban"""
    def __init__(self, w1, wA, w2, w3, w4, w_res, bias_initializer):
        super(AI4Urban, self).__init__()
        # self.arg = arg
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.A = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)  
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)

        self.A.weight.data = wA
        self.res.weight.data = w_res
        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.zadv.weight.data = w4

        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer
###############################################################

    def scale_faces(self, my_face_size, neig_face):
        '''
        it scales neig_face and returns it
        my_face_size :: the size of the current face 
        neig_face    :: output :: info coming from
        ratio        :: my_face_size / neig_face size
        '''
        # my_face_size and neig_ faces are 2D pytorch tensors
        ratio = my_face_size[0] / neig_face.size()[0]
        # Check if the sizes are equal
        if ratio == 1:
            pass
        elif ratio == 2:                       # Check if my_face is twice as large as neig_face
            # Convert tensor to float
            neig_face = neig_face.float()
            # Resize neig_face to fill my_face
            neig_face = F.interpolate(neig_face[None, None, :, :], size=my_face_size, mode='nearest')[0, 0]
        elif ratio == 0.5:                      # Check if my_face is half of neig_face
            # Convert tensor to float
            neig_face = neig_face.float()
            # Apply 2D average pooling to neig_face
            neig_face = F.avg_pool2d(neig_face[None, None, :, :], 2)[0, 0]
        
        neig_face = neig_face.unsqueeze(0).unsqueeze(0)
        neig_face = F.pad(neig_face, (1, 1, 1, 1), mode='reflect')
        neig_face = neig_face.squeeze(0).squeeze(0)
        return neig_face
    

    def scale_faces_k_uvw(self, my_face_size, neig_face):
        '''
        it scales neig_face and returns it
        my_face_size :: the size of the current face 
        neig_face    :: output :: info coming from
        ratio        :: my_face_size / neig_face size
        '''
        # my_face_size and neig_ faces are 2D pytorch tensors
        ratio = my_face_size[0] / neig_face.size()[0]
        # Check if the sizes are equal
        if ratio == 1:
            pass
        elif ratio == 2:                       # Check if my_face is twice as large as neig_face
            # Convert tensor to float
            neig_face = neig_face.float()
            # Resize neig_face to fill my_face
            neig_face = F.interpolate(neig_face[None, None, :, :], size=my_face_size, mode='nearest')[0, 0]
        elif ratio == 0.5:                      # Check if my_face is half of neig_face
            # Convert tensor to float
            neig_face = neig_face.float()
            # Apply 2D average pooling to neig_face
            neig_face = F.avg_pool2d(neig_face[None, None, :, :], 2)[0, 0]
        
        return neig_face
    
        
    def update_halos(self, sd, dt, nlevel):  # AMIN:: to optimise pass in as local vbls instead of global
        # --------------------------------------------------------------------------------- front
        neig0 = sd.neig[0]
        if not isinstance(neig0, str):
            my_face_size = sd.values_u[0,0,0,:,:].size()
            sd.halo_u[0]    = self.scale_faces(my_face_size, neig0.values_u[0,0,0,:,:])
            sd.halo_v[0]   = self.scale_faces(my_face_size, neig0.values_v[0,0,0,:,:])
            sd.halo_w[0]    = self.scale_faces(my_face_size, neig0.values_w[0,0,0,:,:])
            sd.halo_p[0]    = self.scale_faces(my_face_size, neig0.values_p[0,0,0,:,:])
            sd.halo_k_u[0]  = self.scale_faces(my_face_size, neig0.k_u[0,0,0,:,:])
            sd.halo_k_v[0]  = self.scale_faces(my_face_size, neig0.k_v[0,0,0,:,:])
            sd.halo_k_w[0]  = self.scale_faces(my_face_size, neig0.k_w[0,0,0,:,:])
            sd.halo_PGu[0]  = torch.minimum(neig0.k_u[0,0,-1,:,:], neig0.k1[0,0,-1,:,:]) /dt
            sd.halo_PGv[0]  = torch.minimum(neig0.k_v[0,0,-1,:,:], neig0.k1[0,0,-1,:,:]) /dt
            sd.halo_PGw[0]  = torch.minimum(neig0.k_w[0,0,-1,:,:], neig0.k1[0,0,-1,:,:]) /dt
            for ilevel in range(nlevel-1):
                sd.halo_MG[ilevel][0]  = self.scale_faces(sd.w[ilevel][0,0,0,:,:].size(), neig0.w.ilevel[0,0,-1,:,:])
        # --------------------------------------------------------------------------------- back
        neig1 = sd.neig[1]
        if not isinstance(neig1, str):
            my_face_size = sd.values_u[0,0,0,:,:].size()
            sd.halo_u[1]    = self.scale_faces(my_face_size, neig1.values_u[0,0,-1,:,:])
            sd.halo_v[1]    = self.scale_faces(my_face_size, neig1.values_v[0,0,-1,:,:])
            sd.halo_w[1]    = self.scale_faces(my_face_size, neig1.values_w[0,0,-1,:,:])
            sd.halo_p[1]    = self.scale_faces(my_face_size, neig1.values_p[0,0,-1,:,:])
            sd.halo_k_u[1]  = self.scale_faces(my_face_size, neig1.k_u[0,0,-1,:,:])
            sd.halo_k_v[1]  = self.scale_faces(my_face_size, neig1.k_v[0,0,-1,:,:])
            sd.halo_k_w[1]  = self.scale_faces(my_face_size, neig1.k_w[0,0,-1,:,:])
            sd.halo_PGu[1]  = torch.minimum(neig1.k_u[0,0,0,:,:], neig1.k1[0,0,0,:,:]) /dt
            sd.halo_PGv[1]  = torch.minimum(neig1.k_v[0,0,0,:,:], neig1.k1[0,0,0,:,:]) /dt
            sd.halo_PGw[1]  = torch.minimum(neig1.k_w[0,0,0,:,:], neig1.k1[0,0,0,:,:]) /dt
            for ilevel in range(nlevel-1):
                sd.halo_MG[ilevel][1]  = self.scale_faces(sd.w[ilevel][0,0,-1,:,:].size(), neig1.w[ilevel][0,0,0,:,:])
        # --------------------------------------------------------------------------------- left
        neig2 = sd.neig[2]
        if not isinstance(neig2, str):
            my_face_size = sd.values_u[0,0,:,:,0].size()
            if neig2.values_u.device.index == sd.values_u.device.index:
                sd.halo_u[2]    = self.scale_faces(my_face_size, neig2.values_u[0,0,:,:,-1])
                sd.halo_v[2]    = self.scale_faces(my_face_size, neig2.values_v[0,0,:,:,-1])
                sd.halo_w[2]    = self.scale_faces(my_face_size, neig2.values_w[0,0,:,:,-1])
                sd.halo_p[2]    = self.scale_faces(my_face_size, neig2.values_p[0,0,:,:,-1])
                sd.halo_k_u[2]  = self.scale_faces(my_face_size, neig2.k_u[0,0,:,:,-1])
                sd.halo_k_v[2]  = self.scale_faces(my_face_size, neig2.k_v[0,0,:,:,-1])
                sd.halo_k_w[2]  = self.scale_faces(my_face_size, neig2.k_w[0,0,:,:,-1])
                sd.halo_PGu[2]  = torch.minimum(neig2.k_u[0,0,:,:, -1], neig2.k1[0,0,:,:,-1]) /dt
                sd.halo_PGv[2]  = torch.minimum(neig2.k_v[0,0,:,:, -1], neig2.k1[0,0,:,:,-1]) /dt
                sd.halo_PGw[2]  = torch.minimum(neig2.k_w[0,0,:,:, -1], neig2.k1[0,0,:,:,-1]) /dt
                for ilevel in range(nlevel-1):
                    sd.halo_MG[ilevel][2]  = self.scale_faces(sd.w[ilevel][0,0,:,:,-1].size(), neig2.w[ilevel][0,0,:,:,-1])
            else:
                neig_rank = neig2.values_u.device.index
                SendTo = sd.values_u.device

                # sending data to neighbour
                dist.isend(tensor=sd.values_u[0,0,:,:,0].contiguous()                                  , dst=neig_rank ,tag = 2300)
                dist.isend(tensor=sd.values_v[0,0,:,:,0].contiguous()                                  , dst=neig_rank ,tag = 2301)
                dist.isend(tensor=sd.values_w[0,0,:,:,0].contiguous()                                  , dst=neig_rank ,tag = 2302)
                dist.isend(tensor=sd.values_p[0,0,:,:,0].contiguous()                                  , dst=neig_rank ,tag = 2303)
                dist.isend(tensor=sd.k_u[0,0,:,:,0].contiguous()                                       , dst=neig_rank ,tag = 2304)
                dist.isend(tensor=sd.k_v[0,0,:,:,0].contiguous()                                       , dst=neig_rank ,tag = 2305)
                dist.isend(tensor=sd.k_w[0,0,:,:,0].contiguous()                                       , dst=neig_rank ,tag = 2306)
                dist.isend(tensor=sd.b_u[0,0,:,:,0].contiguous()                                       , dst=neig_rank ,tag = 2307)
                dist.isend(tensor=sd.b_v[0,0,:,:,0].contiguous()                                       , dst=neig_rank ,tag = 2308)
                dist.isend(tensor=sd.b_w[0,0,:,:,0].contiguous()                                       , dst=neig_rank ,tag = 2309)
                dist.isend(tensor=(torch.minimum(sd.k_u[0,0,:,:,0], sd.k1[0,0,:,:,0]) /dt).contiguous(), dst=neig_rank ,tag = 2310)
                dist.isend(tensor=(torch.minimum(sd.k_v[0,0,:,:,0], sd.k1[0,0,:,:,0]) /dt).contiguous(), dst=neig_rank ,tag = 2311)
                dist.isend(tensor=(torch.minimum(sd.k_w[0,0,:,:,0], sd.k1[0,0,:,:,0]) /dt).contiguous(), dst=neig_rank ,tag = 2312)
                for ilevel in range(nlevel-1):
                    dist.isend(tensor=sd.w[ilevel][0,0,:,:,0].contiguous()                             , dst=neig_rank ,tag = 2313+ilevel)

                # receiving data from neighbour
                recv_3200 = torch.zeros_like(neig2.values_u[0,0,:,:,-1].contiguous()).to(SendTo)
                dist.recv(tensor=recv_3200.contiguous(), src=neig_rank,tag=3200)
                sd.halo_u[2] = self.scale_faces(my_face_size, recv_3200)

                recv_2301 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2301.contiguous(), src=neig_rank,tag=3201)
                sd.halo_v[2] = self.scale_faces(my_face_size, recv_2301)

                recv_2302 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2302.contiguous(), src=neig_rank,tag=3202)
                sd.halo_w[2] = self.scale_faces(my_face_size, recv_2302)
                
                recv_2303 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2303.contiguous(), src=neig_rank,tag=3203)
                sd.halo_p[2] = self.scale_faces(my_face_size, recv_2303)

                recv_2304 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2304.contiguous(), src=neig_rank,tag=3204)
                sd.halo_k_u[2] = self.scale_faces(my_face_size, recv_2304)

                recv_2305 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2305.contiguous(), src=neig_rank,tag=3205)
                sd.halo_k_v[2] = self.scale_faces(my_face_size, recv_2305)

                recv_2308 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2308.contiguous(), src=neig_rank,tag=3208)
                sd.halo_b_v[2] = self.scale_faces(my_face_size, recv_2308)

                recv_2309 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2309.contiguous(), src=neig_rank,tag=3209)
                sd.halo_b_w[2] = self.scale_faces(my_face_size, recv_2309)

                recv_2310 = torch.zeros_like(recv_3200.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2310.contiguous(), src=neig_rank,tag=3210)
                sd.halo_PGu[2] = self.scale_faces_k_uvw(my_face_size, recv_2310)

                # Create a dictionary to store tensors
                tensors = {}
                for ilevel in range(nlevel-1):
                    # Create a tensor and store it in the dictionary
                    tensors[f"tensor_{3213+ilevel}"] = torch.zeros_like(neig2.w[ilevel][0,0,:,:,-1].contiguous()).to(SendTo)
                    # Use the tensor from the dictionary
                    dist.recv(tensor=tensors[f"tensor_{3213+ilevel}"].contiguous(), src=neig_rank,tag=3213+ilevel)
                    sd.halo_MG[ilevel][2]  = self.scale_faces(sd.w[ilevel][0,0,:,:,0].size(), tensors[f"tensor_{3213+ilevel}"])

        # --------------------------------------------------------------------------------- right
        neig3 = sd.neig[3]
        if not isinstance(neig3, str):
            my_face_size = sd.values_u[0,0,:,:,0].size()
            if neig3.values_u.device.index == sd.values_u.device.index:
                sd.halo_u[3]    = self.scale_faces(my_face_size, neig3.values_u[0,0,:,:,0])
                sd.halo_v[3]    = self.scale_faces(my_face_size, neig3.values_v[0,0,:,:,0])
                sd.halo_w[3]    = self.scale_faces(my_face_size, neig3.values_w[0,0,:,:,0])
                sd.halo_p[3]    = self.scale_faces(my_face_size, neig3.values_p[0,0,:,:,0])
                sd.halo_k_u[3]  = self.scale_faces(my_face_size, neig3.k_u[0,0,:,:,0])
                sd.halo_k_v[3]  = self.scale_faces(my_face_size, neig3.k_v[0,0,:,:,0])
                sd.halo_k_w[3]  = self.scale_faces(my_face_size, neig3.k_w[0,0,:,:,0])
                sd.halo_PGu[3]  = torch.minimum(neig3.k_u[0,0,:,:, 0], neig3.k1[0,0,:,:,0]) /dt
                sd.halo_PGv[3]  = torch.minimum(neig3.k_v[0,0,:,:, 0], neig3.k1[0,0,:,:,0]) /dt
                sd.halo_PGw[3]  = torch.minimum(neig3.k_w[0,0,:,:, 0], neig3.k1[0,0,:,:,0]) /dt
                for ilevel in range(nlevel-1):
                    sd.halo_MG[ilevel][3]  = self.scale_faces(sd.w[ilevel][0,0,:,:,-1].size(), neig3.w[ilevel][0,0,:,:,0])
            else:
                neig_rank = neig3.values_u.device.index
                SendTo = sd.values_u.device

                # receiving data from neighbour
                recv_2300 = torch.zeros_like(neig3.values_u[0,0,:,:,0].contiguous()).to(SendTo)
                dist.recv(tensor=recv_2300.contiguous(), src=neig_rank,tag=2300)
                sd.halo_u[3] = self.scale_faces(my_face_size, recv_2300)

                recv_2301 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2301.contiguous(), src=neig_rank,tag=2301)
                sd.halo_v[3] = self.scale_faces(my_face_size, recv_2301)

                recv_2302 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2302.contiguous(), src=neig_rank,tag=2302)
                sd.halo_w[3] = self.scale_faces(my_face_size, recv_2302)
                
                recv_2303 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2303.contiguous(), src=neig_rank,tag=2303)
                sd.halo_p[3] = self.scale_faces(my_face_size, recv_2303)

                recv_2304 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2304.contiguous(), src=neig_rank,tag=2304)
                sd.halo_k_u[3] = self.scale_faces(my_face_size, recv_2304)

                recv_2305 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2305.contiguous(), src=neig_rank,tag=2305)
                sd.halo_k_v[3] = self.scale_faces(my_face_size, recv_2305)

                recv_2306 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2306.contiguous(), src=neig_rank,tag=2306)
                sd.halo_k_w[3] = self.scale_faces(my_face_size, recv_2306)

                recv_2309 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2309.contiguous(), src=neig_rank,tag=2309)
                sd.halo_b_w[3] = self.scale_faces(my_face_size, recv_2309)

                recv_2310 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2310.contiguous(), src=neig_rank,tag=2310)
                sd.halo_PGu[3] = self.scale_faces_k_uvw(my_face_size, recv_2310)

                recv_2311 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2311.contiguous(), src=neig_rank,tag=2311)
                sd.halo_PGv[3] = self.scale_faces_k_uvw(my_face_size, recv_2311)

                recv_2312 = torch.zeros_like(recv_2300.contiguous()).to(SendTo)
                dist.recv(tensor=recv_2312.contiguous(), src=neig_rank,tag=2312)
                sd.halo_PGw[3] = self.scale_faces_k_uvw(my_face_size, recv_2312)

                # Create a dictionary to store tensors
                tensors = {}
                for ilevel in range(nlevel-1):
                    # Create a tensor and store it in the dictionary
                    tensors[f"tensor_{2313+ilevel}"] = torch.zeros_like(neig3.w[ilevel][0,0,:,:,0].contiguous()).to(SendTo)
                    # Use the tensor from the dictionary
                    dist.recv(tensor=tensors[f"tensor_{2313+ilevel}"].contiguous(), src=neig_rank,tag=2313+ilevel)
                    sd.halo_MG[ilevel][3]  = self.scale_faces(sd.w[ilevel][0,0,:,:,-1].size(), tensors[f"tensor_{2313+ilevel}"])


                # sending data to neighbour
                dist.isend(tensor=sd.values_u[0,0,:,:,-1].contiguous()                                   , dst=neig_rank ,tag = 3200)
                dist.isend(tensor=sd.values_v[0,0,:,:,-1].contiguous()                                   , dst=neig_rank ,tag = 3201)
                dist.isend(tensor=sd.values_w[0,0,:,:,-1].contiguous()                                   , dst=neig_rank ,tag = 3202)
                dist.isend(tensor=sd.values_p[0,0,:,:,-1].contiguous()                                   , dst=neig_rank ,tag = 3203)
                dist.isend(tensor=sd.k_u[0,0,:,:,-1].contiguous()                                        , dst=neig_rank ,tag = 3204)
                dist.isend(tensor=sd.k_v[0,0,:,:,-1].contiguous()                                        , dst=neig_rank ,tag = 3205)
                dist.isend(tensor=sd.k_w[0,0,:,:,-1].contiguous()                                        , dst=neig_rank ,tag = 3206)
                dist.isend(tensor=(torch.minimum(sd.k_u[0,0,:,:,-1], sd.k1[0,0,:,:,-1]) /dt).contiguous(), dst=neig_rank ,tag = 3210)
                dist.isend(tensor=(torch.minimum(sd.k_v[0,0,:,:,-1], sd.k1[0,0,:,:,-1]) /dt).contiguous(), dst=neig_rank ,tag = 3211)
                dist.isend(tensor=(torch.minimum(sd.k_w[0,0,:,:,-1], sd.k1[0,0,:,:,-1]) /dt).contiguous(), dst=neig_rank ,tag = 3212)
                for ilevel in range(nlevel-1):
                    dist.isend(tensor=sd.w[ilevel][0,0,:,:,-1].contiguous()                              , dst=neig_rank ,tag = 3213+ilevel)

        # --------------------------------------------------------------------------------- top
        neig4 = sd.neig[4]
        if not isinstance(neig4, str):
            my_face_size = sd.values_u[0,0,:,-1,:].size()
            if neig4.values_u.device.index == sd.values_u.device.index:
                sd.halo_u[4]    = self.scale_faces(my_face_size, neig4.values_u[0,0,:,-1,:])
                sd.halo_v[4]    = self.scale_faces(my_face_size, neig4.values_v[0,0,:,-1,:])
                sd.halo_w[4]    = self.scale_faces(my_face_size, neig4.values_w[0,0,:,-1,:])
                sd.halo_p[4]    = self.scale_faces(my_face_size, neig4.values_p[0,0,:,-1,:])
                sd.halo_k_u[4]  = self.scale_faces(my_face_size, neig4.k_u[0,0,:,-1,:])
                sd.halo_k_v[4]  = self.scale_faces(my_face_size, neig4.k_v[0,0,:,-1,:])
                sd.halo_k_w[4]  = self.scale_faces(my_face_size, neig4.k_w[0,0,:,-1,:])
                sd.halo_PGu[4]  = torch.minimum(neig4.k_u[0,0,:,-1,:], neig4.k1[0,0,:,-1,:]) /dt
                sd.halo_PGv[4]  = torch.minimum(neig4.k_v[0,0,:,-1,:], neig4.k1[0,0,:,-1,:]) /dt
                sd.halo_PGw[4]  = torch.minimum(neig4.k_w[0,0,:,-1,:], neig4.k1[0,0,:,-1,:]) /dt
                for ilevel in range(nlevel-1):
                    sd.halo_MG[ilevel][4]  = self.scale_faces(sd.w[ilevel][0,0,:,-1,:].size(), neig4.w[ilevel][0,0,:,-1,:])
            else:
                neig_rank = neig4.values_u.device.index
                SendTo = sd.values_u.device

                # sending data to neighbour
                dist.isend(tensor=sd.values_u[0,0,:,0,:].contiguous()                                  , dst=neig_rank ,tag = 4500)
                dist.isend(tensor=sd.values_v[0,0,:,0,:].contiguous()                                  , dst=neig_rank ,tag = 4501)
                dist.isend(tensor=sd.values_w[0,0,:,0,:].contiguous()                                  , dst=neig_rank ,tag = 4502)
                dist.isend(tensor=sd.values_p[0,0,:,0,:].contiguous()                                  , dst=neig_rank ,tag = 4503)
                dist.isend(tensor=sd.k_u[0,0,:,0,:].contiguous()                                       , dst=neig_rank ,tag = 4504)
                dist.isend(tensor=sd.k_v[0,0,:,0,:].contiguous()                                       , dst=neig_rank ,tag = 4505)
                dist.isend(tensor=sd.k_w[0,0,:,0,:].contiguous()                                       , dst=neig_rank ,tag = 4506)
                dist.isend(tensor=(torch.minimum(sd.k_u[0,0,:,0,:], sd.k1[0,0,:,0,:]) /dt).contiguous(), dst=neig_rank ,tag = 4510)
                dist.isend(tensor=(torch.minimum(sd.k_v[0,0,:,0,:], sd.k1[0,0,:,0,:]) /dt).contiguous(), dst=neig_rank ,tag = 4511)
                dist.isend(tensor=(torch.minimum(sd.k_w[0,0,:,0,:], sd.k1[0,0,:,0,:]) /dt).contiguous(), dst=neig_rank ,tag = 4512)
                for ilevel in range(nlevel-1):
                    dist.isend(tensor=sd.w[ilevel][0,0,:,0,:].contiguous()                             , dst=neig_rank ,tag = 4513+ilevel)

                # receiving data from neighbour
                recv_5400 = torch.zeros_like(neig4.values_u[0,0,:,-1,:].contiguous()).to(SendTo)
                dist.recv(tensor=recv_5400.contiguous(), src=neig_rank,tag=5400)
                sd.halo_u[4] = self.scale_faces(my_face_size, recv_5400)

                recv_5401 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5401.contiguous(), src=neig_rank,tag=5401)
                sd.halo_v[4] = self.scale_faces(my_face_size, recv_5401)

                recv_5402 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5402.contiguous(), src=neig_rank,tag=5402)
                sd.halo_w[4] = self.scale_faces(my_face_size, recv_5402)
                
                recv_5403 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5403.contiguous(), src=neig_rank,tag=5403)
                sd.halo_p[4] = self.scale_faces(my_face_size, recv_5403)

                recv_5404 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5404.contiguous(), src=neig_rank,tag=5404)
                sd.halo_k_u[4] = self.scale_faces(my_face_size, recv_5404)

                recv_5405 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5405.contiguous(), src=neig_rank,tag=5405)
                sd.halo_k_v[4] = self.scale_faces(my_face_size, recv_5405)

                recv_5406 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5406.contiguous(), src=neig_rank,tag=5406)
                sd.halo_k_w[4] = self.scale_faces(my_face_size, recv_5406)
                sd.halo_b_v[4] = self.scale_faces(my_face_size, recv_5408)

                recv_5409 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5409.contiguous(), src=neig_rank,tag=5409)
                sd.halo_b_w[4] = self.scale_faces(my_face_size, recv_5409)

                recv_5410 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5410.contiguous(), src=neig_rank,tag=5410)
                sd.halo_PGu[4] = self.scale_faces_k_uvw(my_face_size, recv_5410)

                recv_5411 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5411.contiguous(), src=neig_rank,tag=5411)
                sd.halo_PGv[4] = self.scale_faces_k_uvw(my_face_size, recv_5411)

                recv_5412 = torch.zeros_like(recv_5400.contiguous()).to(SendTo)
                dist.recv(tensor=recv_5412.contiguous(), src=neig_rank,tag=5412)
                sd.halo_PGw[4] = self.scale_faces_k_uvw(my_face_size, recv_5412)

                # Create a dictionary to store tensors
                tensors = {}
                for ilevel in range(nlevel-1):
                    # Create a tensor and store it in the dictionary
                    tensors[f"tensor_{5413+ilevel}"] = torch.zeros_like(neig4.w[ilevel][0,0,:,-1,:].contiguous()).to(SendTo)
                    # Use the tensor from the dictionary
                    dist.recv(tensor=tensors[f"tensor_{5413+ilevel}"].contiguous(), src=neig_rank,tag=5413+ilevel)
                    sd.halo_MG[ilevel][4]  = self.scale_faces(sd.w[ilevel][0,0,:,0,:].size(), tensors[f"tensor_{5413+ilevel}"])

        # --------------------------------------------------------------------------------- bottom
        neig5 = sd.neig[5]
        if not isinstance(neig5, str):
            my_face_size = sd.values_u[0,0,:,0,:].size()
            if neig5.values_u.device.index == sd.values_u.device.index:
                sd.halo_u[5]    = self.scale_faces(my_face_size, neig5.values_u[0,0,:,0,:])
                sd.halo_v[5]    = self.scale_faces(my_face_size, neig5.values_v[0,0,:,0,:])
                sd.halo_w[5]    = self.scale_faces(my_face_size, neig5.values_w[0,0,:,0,:])
                sd.halo_p[5]    = self.scale_faces(my_face_size, neig5.values_p[0,0,:,0,:])
                sd.halo_k_u[5]  = self.scale_faces(my_face_size, neig5.k_u[0,0,:,0,:])
                sd.halo_k_v[5]  = self.scale_faces(my_face_size, neig5.k_v[0,0,:,0,:])
                sd.halo_k_w[5]  = self.scale_faces(my_face_size, neig5.k_w[0,0,:,0,:])
                sd.halo_PGu[5]  = torch.minimum(neig5.k_u[0,0,:,0,:], neig5.k1[0,0,:,0,:]) /dt
                sd.halo_PGv[5]  = torch.minimum(neig5.k_v[0,0,:,0,:], neig5.k1[0,0,:,0,:]) /dt
                sd.halo_PGw[5]  = torch.minimum(neig5.k_w[0,0,:,0,:], neig5.k1[0,0,:,0,:]) /dt
                for ilevel in range(nlevel-1):
                    sd.halo_MG[ilevel][5]  = self.scale_faces(sd.w[ilevel][0,0,:,-1,:].size(), neig5.w[ilevel][0,0,:,0,:])
            else:
                neig_rank = neig5.values_u.device.index
                SendTo = sd.values_u.device

                # receiving data from neighbour
                recv_4500 = torch.zeros_like(neig5.values_u[0,0,:,0,:].contiguous()).to(SendTo)
                dist.recv(tensor=recv_4500.contiguous(), src=neig_rank,tag=4500)
                sd.halo_u[5] = self.scale_faces(my_face_size, recv_4500)

                recv_4501 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4501.contiguous(), src=neig_rank,tag=4501)
                sd.halo_v[5] = self.scale_faces(my_face_size, recv_4501)

                recv_4502 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4502.contiguous(), src=neig_rank,tag=4502)
                sd.halo_w[5] = self.scale_faces(my_face_size, recv_4502)
                
                recv_4503 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4503.contiguous(), src=neig_rank,tag=4503)
                sd.halo_p[5] = self.scale_faces(my_face_size, recv_4503)

                recv_4504 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4504.contiguous(), src=neig_rank,tag=4504)
                sd.halo_k_u[5] = self.scale_faces(my_face_size, recv_4504)

                recv_4505 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4505.contiguous(), src=neig_rank,tag=4505)
                sd.halo_k_v[5] = self.scale_faces(my_face_size, recv_4505)

                recv_4508 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4508.contiguous(), src=neig_rank,tag=4508)
                sd.halo_b_v[5] = self.scale_faces(my_face_size, recv_4508)

                recv_4509 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4509.contiguous(), src=neig_rank,tag=4509)
                sd.halo_b_w[5] = self.scale_faces(my_face_size, recv_4509)

                recv_4510 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4510.contiguous(), src=neig_rank,tag=4510)
                sd.halo_PGu[5] = self.scale_faces_k_uvw(my_face_size, recv_4510)

                recv_4511 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4511.contiguous(), src=neig_rank,tag=4511)
                sd.halo_PGv[5] = self.scale_faces_k_uvw(my_face_size, recv_4511)

                recv_4512 = torch.zeros_like(recv_4500.contiguous()).to(SendTo)
                dist.recv(tensor=recv_4512.contiguous(), src=neig_rank,tag=4512)
                sd.halo_PGw[5] = self.scale_faces_k_uvw(my_face_size, recv_4512)

                # Create a dictionary to store tensors
                tensors = {}
                for ilevel in range(nlevel-1):
                    # Create a tensor and store it in the dictionary
                    tensors[f"tensor_{4513+ilevel}"] = torch.zeros_like(neig5.w[ilevel][0,0,:,0,:].contiguous()).to(SendTo)
                    # Use the tensor from the dictionary
                    dist.recv(tensor=tensors[f"tensor_{4513+ilevel}"].contiguous(), src=neig_rank,tag=4513+ilevel)
                    sd.halo_MG[ilevel][5]  = self.scale_faces(sd.w[ilevel][0,0,:,-1,:].size(), tensors[f"tensor_{4513+ilevel}"])

                # sending data to neighbour
                dist.isend(tensor=sd.values_u[0,0,:,-1,:].contiguous()                                   , dst=neig_rank ,tag = 5400)
                dist.isend(tensor=sd.values_v[0,0,:,-1,:].contiguous()                                   , dst=neig_rank ,tag = 5401)
                dist.isend(tensor=sd.values_w[0,0,:,-1,:].contiguous()                                   , dst=neig_rank ,tag = 5402)
                dist.isend(tensor=sd.values_p[0,0,:,-1,:].contiguous()                                   , dst=neig_rank ,tag = 5403)
                dist.isend(tensor=sd.k_u[0,0,:,-1,:].contiguous()                                        , dst=neig_rank ,tag = 5404)
                dist.isend(tensor=sd.k_v[0,0,:,-1,:].contiguous()                                        , dst=neig_rank ,tag = 5405)
                dist.isend(tensor=sd.k_w[0,0,:,-1,:].contiguous()                                        , dst=neig_rank ,tag = 5406)
                dist.isend(tensor=(torch.minimum(sd.k_u[0,0,:,-1,:], sd.k1[0,0,:,-1,:]) /dt).contiguous(), dst=neig_rank ,tag = 5410)
                dist.isend(tensor=(torch.minimum(sd.k_v[0,0,:,-1,:], sd.k1[0,0,:,-1,:]) /dt).contiguous(), dst=neig_rank ,tag = 5411)
                dist.isend(tensor=(torch.minimum(sd.k_w[0,0,:,-1,:], sd.k1[0,0,:,-1,:]) /dt).contiguous(), dst=neig_rank ,tag = 5412)
                for ilevel in range(nlevel-1):
                    dist.isend(tensor=sd.w[ilevel][0,0,:,-1,:].contiguous()                              , dst=neig_rank ,tag = 5413+ilevel)
        return
    
    
    def boundary_condition_u(self, values_u, values_uu, halo, sd, ub):
        values_uu[0,0,1:-1,1:-1,1:-1] = values_u[0,0,:,:,:] 
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            values_uu[0,0,0,:,:] = values_uu[0,0,1,:,:] 
        else:
            values_uu[0,0,0,:,:] = halo[0]
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):
            values_uu[0,0,-1,:,:] = values_uu[0,0,-2,:,:]
        else:
            values_uu[0,0,-1,:,:] = halo[1]
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            values_uu[0,0,:,:,0].fill_(ub)
        else:
            values_uu[0,0,:,:,0] = halo[2]
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            values_uu[0,0,:,:,-1].fill_(ub)
        else:
            values_uu[0,0,:,:,-1] = halo[3]
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            values_uu[0,0,:,0,:] = values_uu[0,0,:,1,:]
        else:
            values_uu[0,0,:,0,:] = halo[4]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            values_uu[0,0,:,-1,:] = values_uu[0,0,:,-2,:]
        else:
            values_uu[0,0,:,-1,:] = halo[5]
        return
    

    def boundary_condition_v(self, values_v, values_vv, halo, sd):
        values_vv[0,0,1:-1,1:-1,1:-1] = values_v[0,0,:,:,:]
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            values_vv[0,0,0,:,:].fill_(0.0)
        else:
            values_vv[0,0,0,:,:] = halo[0]
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):
            values_vv[0,0,-1,:,:] = values_vv[0,0,-2,:,:]
        else:
            values_vv[0,0,-1,:,:] = halo[1]
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            values_vv[0,0,:,:,0].fill_(0.0)
        else:
            values_vv[0,0,:,:,0] = halo[2]
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            values_vv[0,0,:,:,-1].fill_(0.0)
        else:
            values_vv[0,0,:,:,-1] = halo[3]
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            values_vv[0,0,:,0,:].fill_(0.0)
        else:
            values_vv[0,0,:,0,:] = halo[4]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            values_vv[0,0,:,-1,:].fill_(0.0)
        else:
            values_vv[0,0,:,-1,:] = halo[5]
        return


    def boundary_condition_w(self, values_w, values_ww, halo, sd):
        values_ww[0,0,1:-1,1:-1,1:-1] = values_w[0,0,:,:,:]
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            values_ww[0,0,0,:,:].fill_(0.0)
        else:
            values_ww[0,0,0,:,:] = halo[0]
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):
            values_ww[0,0,-1,:,:].fill_(0.0)
        else:
            values_ww[0,0,-1,:,:] = halo[1]
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            values_ww[0,0,:,:,0].fill_(0.0)
        else:
            values_ww[0,0,:,:,0] = halo[2]
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            values_ww[0,0,:,:,-1].fill_(0.0)
        else:
            values_ww[0,0,:,:,-1] = halo[3]
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            values_ww[0,0,:,0,:].fill_(0.0)
        else:
            values_ww[0,0,:,0,:] = halo[4]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            values_ww[0,0,:,-1,:].fill_(0.0)
        else:
            values_ww[0,0,:,-1,:] = halo[5]
        return
    
    
    def boundary_condition_p(self, values_p, values_pp, halo, sd):  # need check this boundary condition for the real case 
        values_pp[0,0,1:-1,1:-1,1:-1] = values_p[0,0,:,:,:]
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            values_pp[0,0,0,:,:] = values_pp[0,0,1,:,:]
        else:
            values_pp[0,0,0,:,:] = halo[0]
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):
            values_pp[0,0,-1,:,:] = values_pp[0,0,-2,:,:]
        else:
            values_pp[0,0,-1,:,:] = halo[1]
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            values_pp[0,0,:,:,0] =  values_pp[0,0,:,:,1] 
        else:
            values_pp[0,0,:,:,0] = halo[2]
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            values_pp[0,0,:,:,-1].fill_(0.0)
        else:
            values_pp[0,0,:,:,-1] = halo[3]
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            values_pp[0,0,:,0,:] = values_pp[0,0,:,1,:]
        else:
            values_pp[0,0,:,0,:] = halo[4]
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            values_pp[0,0,:,-1,:] = values_pp[0,0,:,-2,:]
        else:
            values_pp[0,0,:,-1,:] = halo[5]
        return values_pp
    
    
    def boundary_condition_k_u(self, sd):   # Amin:: if results are not correct. uncomment /(1+dt*sd.sigma)
        sd.k_uu[0,0,1:-1,1:-1,1:-1] = sd.k_u[0,0,:,:,:]
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            sd.k_uu[0,0,0,:,:].fill_(0.0)
        else:
            # sd.k_uu[0,0,0,:,:]   = self.scale_faces(sd.k_uu[0,0,0,1:-1,1:-1].size(), torch.minimum(sd.neig[0].k_u[0,0,-1,:,:], sd.neig[0].k1[0,0,-1,:,:]) /dt)# (1+dt*sd.sigma))
            sd.k_uu[0,0,0,:,:]   = self.scale_faces(sd.k_uu[0,0,0,:,:].size(), sd.halo_PGu[0])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):
            sd.k_uu[0,0,-1,:,:].fill_(0.0)
        else:
            # sd.k_uu[0,0,-1, :,:] = self.scale_faces(sd.k_uu[0,0,-1, 1:-1,1:-1].size(), torch.minimum(sd.neig[1].k_u[0,0,0,:,:], sd.neig[1].k1[0,0,0,:,:]) /dt)# (1+dt*sd.sigma))
            sd.k_uu[0,0,-1, :,:] = self.scale_faces(sd.k_uu[0,0,-1, :,:].size(), sd.halo_PGu[1])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            sd.k_uu[0,0,:,:,0].fill_(0.0)
        else:
            # sd.k_uu[0,0, :,:,0]  = self.scale_faces(sd.k_uu[0,0, 1:-1,1:-1,0].size(), torch.minimum(sd.neig[2].k_u[0,0,:,:, -1], sd.neig[2].k1[0,0,:,:,-1]) /dt)# (1+dt*sd.sigma))
            sd.k_uu[0,0, :,:,0]  = self.scale_faces(sd.k_uu[0,0, :,:,0].size(), sd.halo_PGu[2])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            sd.k_uu[0,0,:,:,-1].fill_(0.0) 
        else:
            # sd.k_uu[0,0, :,:,-1] = self.scale_faces(sd.k_uu[0,0, 1:-1,1:-1,-1].size(), torch.minimum(sd.neig[3].k_u[0,0,:,:, 0], sd.neig[3].k1[0,0,:,:,0]) /dt)# (1+dt*sd.sigma))
            sd.k_uu[0,0, :,:,-1] = self.scale_faces(sd.k_uu[0,0, :,:,-1].size(), sd.halo_PGu[3])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            sd.k_uu[0,0,:,0,:].fill_(0.0)
        else:
            # sd.k_uu[0,0,:,0,:]   = self.scale_faces(sd.k_uu[0,0,1:-1,0,1:-1].size(), torch.minimum(sd.neig[4].k_u[0,0,:,-1,:], sd.neig[4].k1[0,0,:,-1,:]) /dt)# (1+dt*sd.sigma))
            sd.k_uu[0,0,:,0,:]   = self.scale_faces(sd.k_uu[0,0,:,0,:].size(), sd.halo_PGu[4])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            sd.k_uu[0,0,:,-1,:].fill_(0.0)
        else:
            # sd.k_uu[0,0,:, -1,:] = self.scale_faces(sd.k_uu[0,0,1:-1, -1,1:-1].size(), torch.minimum(sd.neig[5].k_u[0,0,:,0,:], sd.neig[5].k1[0,0,:,0,:]) /dt)# (1+dt*sd.sigma))
            sd.k_uu[0,0,:, -1,:] = self.scale_faces(sd.k_uu[0,0,:, -1,:].size(), sd.halo_PGu[5])# (1+dt*sd.sigma))
        return


    def boundary_condition_k_v(self, sd):
        sd.k_vv[0,0,1:-1,1:-1,1:-1] = sd.k_v[0,0,:,:,:]
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            sd.k_vv[0,0,0,:,:].fill_(0.0)
        else:
            # sd.k_vv[0,0,0,:,:] = self.scale_faces(sd.k_vv[0,0,0,1:-1,1:-1].size(), torch.minimum(sd.neig[0].k_v[0,0,-1,:,:], sd.neig[0].k1[0,0,-1,:,:]) /dt)# (1+dt*sd.sigma))
            sd.k_vv[0,0,0,:,:] = self.scale_faces(sd.k_vv[0,0,0,:,:].size(), sd.halo_PGv[0])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):
            sd.k_vv[0,0,-1,:,:].fill_(0.0)
        else:
            # sd.k_vv[0,0,-1, :,:] = self.scale_faces(sd.k_vv[0,0,-1, 1:-1,1:-1].size(), torch.minimum(sd.neig[1].k_v[0,0,0,:,:], sd.neig[1].k1[0,0,0,:,:]) /dt)# (1+dt*sd.sigma))
            sd.k_vv[0,0,-1, :,:] = self.scale_faces(sd.k_vv[0,0,-1, :,:].size(), sd.halo_PGv[1])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            sd.k_vv[0,0,:,:,0].fill_(0.0)
        else:
            # sd.k_vv[0,0, :,:,0] = self.scale_faces(sd.k_vv[0,0, 1:-1,1:-1,0].size(), torch.minimum(sd.neig[2].k_v[0,0,:,:, -1], sd.neig[2].k1[0,0,:,:,-1]) /dt)# (1+dt*sd.sigma))
            sd.k_vv[0,0, :,:,0] = self.scale_faces(sd.k_vv[0,0, :,:,0].size(), sd.halo_PGv[2])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            sd.k_vv[0,0,:,:,-1].fill_(0.0) 
        else:
            # sd.k_vv[0,0, :,:,-1] = self.scale_faces(sd.k_vv[0,0, 1:-1,1:-1,-1].size(), torch.minimum(sd.neig[3].k_v[0,0,:,:, 0], sd.neig[3].k1[0,0,:,:,0]) /dt)# (1+dt*sd.sigma))
            sd.k_vv[0,0, :,:,-1] = self.scale_faces(sd.k_vv[0,0, :,:,-1].size(), sd.halo_PGv[3])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            sd.k_vv[0,0,:,0,:].fill_(0.0)
        else:
            # sd.k_vv[0,0,:,0,:] = self.scale_faces(sd.k_vv[0,0,1:-1,0,1:-1].size(), torch.minimum(sd.neig[4].k_v[0,0,:,-1,:], sd.neig[4].k1[0,0,:,-1,:]) /dt)# (1+dt*sd.sigma))
            sd.k_vv[0,0,:,0,:] = self.scale_faces(sd.k_vv[0,0,:,0,:].size(), sd.halo_PGv[4])# (1+dt*sd.sigma))
        # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            sd.k_vv[0,0,:,-1,:].fill_(0.0)
        else:
            # sd.k_vv[0,0,:, -1,:] = self.scale_faces(sd.k_vv[0,0,1:-1, -1,1:-1].size(), torch.minimum(sd.neig[5].k_v[0,0,:,0,:], sd.neig[5].k1[0,0,:,0,:]) /dt)# (1+dt*sd.sigma))
            sd.k_vv[0,0,:,-1,:] = self.scale_faces(sd.k_vv[0,0,:,-1,:].size(), sd.halo_PGv[5])# (1+dt*sd.sigma))
        return
    
    
    def boundary_condition_cw(self, sd, w, ilevel):
        ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        # -------------------------------------------------------------------------------- front
        if isinstance(sd.neig[0], str):
            ww[0,0,0,:,:].fill_(0.0)
        else:
            ww[0,0,0,:,:] = sd.halo_MG[ilevel][0]
        # -------------------------------------------------------------------------------- back
        if isinstance(sd.neig[1], str):    
            ww[0,0,-1,:,:].fill_(0.0)
        else:
            ww[0,0,-1,:,:] = sd.halo_MG[ilevel][1]
        # -------------------------------------------------------------------------------- left
        if isinstance(sd.neig[2], str):
            ww[0,0,:,:,0].fill_(0.0)
        else:
            ww[0,0,:,:,0] = sd.halo_MG[ilevel][2]
        # -------------------------------------------------------------------------------- right
        if isinstance(sd.neig[3], str):
            ww[0,0,:,:,-1].fill_(0.0)
        else:
            ww[0,0,:,:,-1] = sd.halo_MG[ilevel][3]
        # -------------------------------------------------------------------------------- top
        if isinstance(sd.neig[4], str):
            ww[0,0,:,0,:].fill_(0.0)
        else:
            ww[0,0,:,0,:] = sd.halo_MG[ilevel][4]
            # -------------------------------------------------------------------------------- bottom
        if isinstance(sd.neig[5], str):
            ww[0,0,:,-1,:].fill_(0.0)
        else:
            ww[0,0,:,-1,:] = sd.halo_MG[ilevel][5]
        return ww
    

    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        values_u = values_u / (1+dt*sigma) 
        values_v = values_v / (1+dt*sigma) 
        values_w = values_w / (1+dt*sigma) 
        return values_u, values_v, values_w
    

    def V_cycle_MG(self, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, sd):
        # ratio = int(max(nx, ny, nz) / min(nx, ny, nz))
        # ratio_x = int(sd.nx/sd.nz)
        # ratio_y = int(sd.ny/sd.nz)
        b = -(self.xadv(values_uu) + self.yadv(values_vv) + self.zadv(values_ww)) / dt
        for MG in range(iteration):
            sd.w[0].fill_(0.0)# = torch.zeros((1,1,2,2*ratio_y,2*ratio_x), device=device)
            r = self.A(self.boundary_condition_p(values_p, values_pp, sd.halo_p, sd)) - b 
            r_s = []  
            r_s.append(r)
            for i in range(1,nlevel-1):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-1)):
                sd.w[i] = sd.w[i] - self.A(self.boundary_condition_cw(sd, sd.w[i],i)) / diag + r_s[i] / diag
                sd.w[i-1] = self.prol(sd.w[i])  
            values_p = values_p - sd.w[0]
            values_p = values_p - self.A(self.boundary_condition_p(values_p, values_pp, sd.halo_p, sd)) / diag + b / diag
        return values_p, r


    def PG_vector(self, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w, sd, dx, dy, dz):
        sd.k_u = 0.1 * dx * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_u) / \
            (1e-03 + (torch.abs(ADx_u) * dx**-3 + torch.abs(ADy_u) * dx**-3 + torch.abs(ADz_u) * dx**-3) / 3)

        sd.k_v = 0.1 * dy * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_v) / \
            (1e-03 + (torch.abs(ADx_v) * dx**-3 + torch.abs(ADy_v) * dx**-3 + torch.abs(ADz_v) * dx**-3) / 3)

        sd.k_w = 0.1 * dz * torch.abs(1/3 * dx**-3 * (torch.abs(values_u) * dx + torch.abs(values_v) * dy + torch.abs(values_w) * dz) * AD2_w) / \
            (1e-03 + (torch.abs(ADx_w) * dx**-3 + torch.abs(ADy_w) * dx**-3 + torch.abs(ADz_w) * dx**-3) / 3)

        self.boundary_condition_k_u(sd)     # **************************** halo update -> k_uu ****************************
        self.boundary_condition_k_v(sd)     # **************************** halo update -> k_vv ****************************
        self.boundary_condition_k_w(sd)     # **************************** halo update -> k_ww ****************************

        k_x = 0.5 * (sd.k_u * AD2_u + self.diff(values_uu * sd.k_uu) - values_u * self.diff(sd.k_uu))
        k_y = 0.5 * (sd.k_v * AD2_v + self.diff(values_vv * sd.k_vv) - values_v * self.diff(sd.k_vv))
        k_z = 0.5 * (sd.k_w * AD2_w + self.diff(values_ww * sd.k_ww) - values_w * self.diff(sd.k_ww))
        return k_x, k_y, k_z


    def forward(self, sd, dt, iteration, nlevel, ub, Re, diag):
        self.update_halos(sd, dt, nlevel)
        [sd.values_u, sd.values_v, sd.values_w] = self.solid_body(sd.values_u, sd.values_v, sd.values_w, sd.sigma, dt)
        # Padding velocity vectors 
        self.boundary_condition_u(sd.values_u, sd.values_uu, sd.halo_u, sd, ub) # ****************************** halo update -> values_uu ************************  
        self.boundary_condition_v(sd.values_v, sd.values_vv, sd.halo_v, sd) # ****************************** halo update -> values_vv ************************ 
        self.boundary_condition_w(sd.values_w, sd.values_ww, sd.halo_w, sd) # ****************************** halo update -> values_ww ************************ 
        sd.values_pp = self.boundary_condition_p(sd.values_p, sd.values_pp, sd.halo_p, sd) # ****************************** halo update -> values_pp ************************ 

        Grapx_p = self.xadv(sd.values_pp) * dt ; Grapy_p = self.yadv(sd.values_pp) * dt ; Grapz_p = self.zadv(sd.values_pp) * dt  
        ADx_u   = self.xadv(sd.values_uu) ; ADy_u = self.yadv(sd.values_uu) ; ADz_u = self.zadv(sd.values_uu)
        ADx_v   = self.xadv(sd.values_vv) ; ADy_v = self.yadv(sd.values_vv) ; ADz_v = self.zadv(sd.values_vv)
        ADx_w   = self.xadv(sd.values_ww) ; ADy_w = self.yadv(sd.values_ww) ; ADz_w = self.zadv(sd.values_ww)
        AD2_u   = self.diff(sd.values_uu) ; AD2_v = self.diff(sd.values_vv) ; AD2_w = self.diff(sd.values_ww)
        # First step for solving uvw
        [k_x,k_y,k_z] = self.PG_vector(sd.values_uu, sd.values_vv, sd.values_ww, sd.values_u, sd.values_v, sd.values_w, sd.k1, sd.k_uu, sd.k_vv, sd.k_ww, 
                                       ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w, sd, sd.dx, sd.dy, sd.dz)

        sd.b_u = sd.values_u + 0.5 * (Re * k_x * dt - sd.values_u * ADx_u * dt - sd.values_v * ADy_u * dt - sd.values_w * ADz_u * dt) - Grapx_p
        sd.b_v = sd.values_v + 0.5 * (Re * k_y * dt - sd.values_u * ADx_v * dt - sd.values_v * ADy_v * dt - sd.values_w * ADz_v * dt) - Grapy_p
        sd.b_w = sd.values_w + 0.5 * (Re * k_z * dt - sd.values_u * ADx_w * dt - sd.values_v * ADy_w * dt - sd.values_w * ADz_w * dt) - Grapz_p
        
        [sd.b_u, sd.b_v, sd.b_w] = self.solid_body(sd.b_u, sd.b_v, sd.b_w, sd.sigma, dt)
        # Padding velocity vectors 
        self.boundary_condition_u(sd.b_u,sd.b_uu, sd.halo_b_u, sd, ub) # ****************************** halo update -> b_uu ************************ 
        self.boundary_condition_v(sd.b_v,sd.b_vv, sd.halo_b_v, sd)     # ****************************** halo update -> b_vv ************************ 
        self.boundary_condition_w(sd.b_w,sd.b_ww, sd.halo_b_w, sd)     # ****************************** halo update -> b_ww ************************ 

        ADx_u = self.xadv(sd.b_uu) ; ADy_u = self.yadv(sd.b_uu) ; ADz_u = self.zadv(sd.b_uu)
        ADx_v = self.xadv(sd.b_vv) ; ADy_v = self.yadv(sd.b_vv) ; ADz_v = self.zadv(sd.b_vv)
        ADx_w = self.xadv(sd.b_ww) ; ADy_w = self.yadv(sd.b_ww) ; ADz_w = self.zadv(sd.b_ww)
        AD2_u = self.diff(sd.b_uu) ; AD2_v = self.diff(sd.b_vv) ; AD2_w = self.diff(sd.b_ww)

        [k_x,k_y,k_z] = self.PG_vector(sd.b_uu, sd.b_vv, sd.b_ww, sd.b_u, sd.b_v, sd.b_w, sd.k1, sd.k_uu, sd.k_vv, sd.k_ww, 
                                       ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w, AD2_u, AD2_v, AD2_w, sd, sd.dx, sd.dy, sd.dz)   
        # Second step for solving uvw   
        sd.values_u = sd.values_u + Re * k_x * dt - sd.b_u * ADx_u * dt - sd.b_v * ADy_u * dt - sd.b_w * ADz_u * dt - Grapx_p  
        sd.values_v = sd.values_v + Re * k_y * dt - sd.b_u * ADx_v * dt - sd.b_v * ADy_v * dt - sd.b_w * ADz_v * dt - Grapy_p  
        sd.values_w = sd.values_w + Re * k_z * dt - sd.b_u * ADx_w * dt - sd.b_v * ADy_w * dt - sd.b_w * ADz_w * dt - Grapz_p
        
        [sd.values_u, sd.values_v, sd.values_w] = self.solid_body(sd.values_u, sd.values_v, sd.values_w, sd.sigma, dt)
        # pressure
        self.boundary_condition_u(sd.values_u,sd.values_uu, sd.halo_u, sd, ub) # ****************************** halo update -> values_uu ************************ 
        self.boundary_condition_v(sd.values_v,sd.values_vv, sd.halo_v, sd)     # ****************************** halo update -> values_vv ************************ 
        self.boundary_condition_w(sd.values_w,sd.values_ww, sd.halo_w, sd)     # ****************************** halo update -> values_ww ************************
        [sd.values_p, r] = self.V_cycle_MG(sd.values_uu, sd.values_vv, sd.values_ww, sd.values_p, sd.values_pp, iteration, diag, dt, nlevel, sd)
        # Pressure gradient correction    
        sd.values_pp = self.boundary_condition_p(sd.values_p, sd.values_pp, sd.halo_p, sd) # ****************************** halo update -> values_pp ************************       
        sd.values_u  = sd.values_u - self.xadv(sd.values_pp) * dt ; sd.values_v = sd.values_v - self.yadv(sd.values_pp) * dt ; sd.values_w = sd.values_w - self.zadv(sd.values_pp) * dt      
        # Solid body
        [sd.values_u, sd.values_v, sd.values_w] = self.solid_body(sd.values_u, sd.values_v, sd.values_w, sd.sigma, dt)
        return

