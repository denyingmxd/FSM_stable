# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry_util import Projection


class ViewRendering(nn.Module):
    """
    Class for rendering images from given camera parameters and pixel wise depth information
    """
    def __init__(self, cfg, rank):
        super().__init__()
        self.read_config(cfg)
        self.rank = rank
        self.project = self.init_project_imgs(rank)      
            
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_project_imgs(self, rank):
        project_imgs = {}
        project_imgs = Projection(
                self.batch_size, self.height, self.width, rank)
        return project_imgs    
    
    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w)
        return mean, torch.sqrt(var + 1e-16)     

    def get_ls_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()
        b,c,h,w = src_img.shape
        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            mask_one = mask[0][0].view(-1)
            if torch.any(mask_sum == 0):
                return warp_img
            kbs = []
            masked_src_img = mask * src_img
            masked_warp_img = mask * warp_img
            for i in range(3):
                masked_src_img_slice = masked_src_img[0][i].view(-1)
                masked_warp_img_slice = masked_warp_img[0][i].view(-1)
                masked_src_img_slice = masked_src_img_slice[mask_one>0]
                masked_warp_img_slice = masked_warp_img_slice[mask_one>0]

                XX = torch.stack((torch.ones_like(masked_warp_img_slice), masked_warp_img_slice ), 1 )
                kb = torch.linalg.lstsq(XX, masked_src_img_slice).solution.view(-1,1)
                kbs.append(kb)

        ls_warps = []
        for i in range(3):
            warp_img_slice = warp_img[0][i].view(-1)
            warp_img_slice_mono = torch.stack( ( torch.ones_like( warp_img_slice), warp_img_slice) , 1 )
            warp_img_slice_ls = warp_img_slice_mono.mm(kbs[i])

            ls_warps.append(warp_img_slice_ls.view(b,1,h,w))
        norm_warp = torch.cat(ls_warps,dim=1)

        return norm_warp * warp_mask.float()

    def get_ls2_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()
        b, c, h, w = src_img.shape
        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1, 3, 1, 1)

            mask_sum = mask.sum(dim=(-3, -2, -1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            masked_src_img = mask * src_img
            masked_warp_img = mask * warp_img

            masked_src_img_vector = masked_src_img[mask]
            masked_warp_img_vector = masked_warp_img[mask]

            XX = torch.stack((torch.ones_like(masked_warp_img_vector), masked_warp_img_vector), 1)
            kb = torch.linalg.lstsq(XX, masked_src_img_vector).solution.view(-1, 1)




        warp_img_slice = warp_img.view(-1)
        warp_img_slice_mono = torch.stack((torch.ones_like(warp_img_slice), warp_img_slice), 1)
        warp_img_slice_ls = warp_img_slice_mono.mm(kb)

        norm_warp = warp_img_slice_ls.view(b, 3, h, w)

        return norm_warp * warp_mask.float()

    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T, scale=0, temporal_border=False):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        """
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K)
        if temporal_border:
            img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear',
                                        padding_mode='border', align_corners=True)
        else:
            img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, 
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped

    def get_virtual_depth(self, src_depth, src_mask, src_invK, src_K, tar_depth, tar_invK, tar_K, T, min_depth, max_depth, scale=0):
        """
        This function backward-warp source depth into the target coordinate.
        This function backward-warp source depth into the target coordinate.
        src -> target
        """
        # transform source depth
        b, _, h, w = src_depth.size()
        src_points = self.project.backproject(src_invK, src_depth)
        src_points_warped = torch.matmul(T[:, :3, :], src_points)
        src_depth_warped = src_points_warped.reshape(b, 3, h, w)[:, 2:3, :, :]

        # reconstruct depth: backward-warp source depth to the target coordinate
        pix_coords = self.project(tar_depth, torch.inverse(T), tar_invK, src_K)
        depth_warped = F.grid_sample(src_depth_warped, pix_coords, mode='bilinear',
                                        padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_depth = torch.isnan(depth_warped)
        depth_warped[inf_depth] = 2.0
        inf_regions = torch.isnan(mask_warped)
        mask_warped[inf_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, pix_coords < -1).sum(dim=1, keepdim=True) > 0

        # range handling
        valid_depth_min = (depth_warped > min_depth)
        depth_warped[~valid_depth_min] = min_depth
        valid_depth_max = (depth_warped < max_depth)
        depth_warped[~valid_depth_max] = max_depth
        return depth_warped, (~invalid_mask).float() * mask_warped * valid_depth_min * valid_depth_max
        
    def forward(self, inputs, outputs, cam, spt_rel_poses):
        # predict images for each scale(default = scale 0 only)
        source_scale = 0
        
        # ref inputs
        ref_color = inputs['color', 0, source_scale][:,cam, ...]        
        ref_mask = inputs['mask'][:, cam, ...]
        ref_K = inputs[('K', source_scale)][:,cam, ...]
        ref_invK = inputs[('inv_K', source_scale)][:,cam, ...]  
        
        # output
        target_view = outputs[('cam', cam)]
        
        for scale in self.scales:           
            ref_depth = target_view[('depth', scale)]          
            for frame_id in self.frame_ids[1:]:                
                # for temporal learning
                T = target_view[('cam_T_cam', 0, frame_id)]
                src_color = inputs['color', frame_id, source_scale][:, cam, ...] 
                src_mask = inputs['mask'][:, cam, ...]
                warped_img, warped_mask = self.get_virtual_image(
                    src_color, 
                    src_mask, 
                    ref_depth, 
                    ref_invK, 
                    ref_K, 
                    T, 
                    source_scale,
                    temporal_border=self.temporal_border
                )
                
                target_view[('color', frame_id, scale)] = warped_img
                target_view[('color_mask', frame_id, scale)] = warped_mask            

            # spatio-temporal learning
            if self.spatio or self.spatio_temporal:
                for frame_id in self.frame_ids:
                    overlap_img = torch.zeros_like(ref_color)
                    overlap_mask = torch.zeros_like(ref_mask)
                    use_depth_consistency = hasattr(self, 'spatial_depth_consistency_loss_weight')
                    if use_depth_consistency:
                        overlap_depth = torch.zeros_like(ref_depth)

                    for aaa,cur_index in enumerate(self.rel_cam_list[cam]):
                        # for partial surround view training
                        if cur_index >= self.num_cams: 
                            continue

                        src_color = inputs['color', frame_id, source_scale][:, cur_index, ...]
                        src_mask = inputs['mask'][:, cur_index, ...]
                        src_K = inputs[('K', source_scale)][:, cur_index, ...]                        
                        
                        # rel_pose = rel_pose_dict[(frame_id, cur_index)]
                        rel_pose = spt_rel_poses[frame_id + 1, cam, cur_index].unsqueeze(0)

                        warped_img, warped_mask = self.get_virtual_image(
                            src_color, 
                            src_mask, 
                            ref_depth, 
                            ref_invK, 
                            src_K,
                            rel_pose, 
                            source_scale,
                        )


                        if self.intensity_align:
                            warped_img = self.get_norm_image_single(
                                ref_color, 
                                ref_mask,
                                warped_img, 
                                warped_mask
                            )

                        if self.intensity_align_ls:
                            warped_img = self.get_ls_image_single(
                                ref_color,
                                ref_mask,
                                warped_img,
                                warped_mask
                            )

                        if self.intensity_align_ls2:
                            warped_img = self.get_ls2_image_single(
                                ref_color,
                                ref_mask,
                                warped_img,
                                warped_mask
                            )

                        target_view[('overlap', frame_id, scale,cur_index)] = warped_img
                        target_view[('overlap_mask', frame_id, scale,cur_index)] = warped_mask

                        # assuming no overlap between warped images
                        overlap_img = overlap_img + warped_img
                        overlap_mask = overlap_mask + warped_mask

                        if use_depth_consistency:
                            if frame_id==0:
                                if self.spatial_depth_consistency_type=='pre':
                                    src_depth = outputs[('cam', cur_index)][('depth', scale)]
                                    src_invK = inputs[('inv_K', source_scale)][:, cur_index, ...]
                                    src_depth_tar_view = self.project.transform_depth(src_depth,torch.linalg.inv(rel_pose),src_invK,ref_K)[:,2:,]
                                    warped_depth, warped_mask = self.get_virtual_image(
                                        src_depth_tar_view,
                                        src_mask,
                                        ref_depth,
                                        ref_invK,
                                        src_K,
                                        rel_pose,
                                        source_scale
                                    )

                                elif self.spatial_depth_consistency_type=='forward':
                                    src_depth = outputs[('cam', cur_index)][('depth', scale)]
                                    src_invK = inputs[('inv_K', source_scale)][:, cur_index, ...]
                                    warped_depth = self.project.get_unnormed_projects(src_depth,torch.linalg.inv(rel_pose),src_invK,ref_K)
                                elif self.spatial_depth_consistency_type=='wrong':
                                    src_depth = outputs[('cam', cur_index)][('depth', scale)]

                                    warped_depth, warped_mask = self.get_virtual_image(
                                        src_depth,
                                        src_mask,
                                        ref_depth,
                                        ref_invK,
                                        src_K,
                                        rel_pose,
                                        source_scale
                                    )
                                else:
                                    raise NotImplementedError


                                overlap_depth = overlap_depth+warped_depth



                    
                    target_view[('overlap', frame_id, scale)] = overlap_img
                    target_view[('overlap_mask', frame_id, scale)] = overlap_mask

                    if use_depth_consistency:
                        target_view[('overlap_depth', frame_id, scale)] = overlap_depth

                    # depth augmentation at a novel view
            if self.aug_depth:
                tform_depth = []
                tform_mask = []

                aug_ext = inputs['extrinsics_aug'][:, cam, ...]
                aug_ext_inv = torch.inverse(aug_ext)                
                aug_K, aug_invK = ref_K, ref_invK
                aug_depth = target_view[('depth', scale, 'aug')]

                for i, curr_index in enumerate(self.rel_cam_list[cam] + [cam]):
                    # for partial surround view training
                    if curr_index >= self.num_cams: 
                        continue

                    src_ext = inputs['extrinsics'][:, curr_index, ...]                        
                    
                    src_depth = outputs[('cam', curr_index)][('depth', scale)]
                    src_mask = inputs['mask'][:, curr_index, ...]                
                    src_invK = inputs[('inv_K', source_scale)][:,curr_index, ...]
                    src_K = inputs[('K', source_scale)][:,curr_index, ...]

                    # current view to the novel view
                    rel_pose = torch.matmul(aug_ext_inv, src_ext)
                    warp_depth, warp_mask = self.get_virtual_depth(
                        src_depth, 
                        src_mask, 
                        src_invK, 
                        src_K,
                        aug_depth, 
                        aug_invK, 
                        aug_K, 
                        rel_pose,
                        self.min_depth,
                        self.max_depth
                    )

                    tform_depth.append(warp_depth)
                    tform_mask.append(warp_mask)

                target_view[('tform_depth', scale)] = tform_depth
                target_view[('tform_depth_mask', scale)] = tform_mask

        outputs[('cam', cam)] = target_view