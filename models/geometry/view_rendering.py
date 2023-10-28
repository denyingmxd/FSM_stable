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
    
    def get_mean_std_multi_cam(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features.
        """
        sum_dim = list(range(len(mask.shape) - 3, len(mask.shape)))
        mask_num = mask.sum(dim=sum_dim, keepdim=True)
        mean = (feature * mask).sum(dim=sum_dim, keepdim=True) / (mask_num + 1e-8)
        var = (((feature - mean) * mask) ** 2).sum(dim=sum_dim, keepdim=True) / (mask_num + 1e-8)
        return mean, torch.sqrt(var + 1e-16)

    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mask_num = mask.sum(dim=(1, 2, 3), keepdim=True)
        mean = (feature * mask).sum(dim=(1, 2, 3), keepdim=True) / (mask_num + 1e-8)
        var = (((feature - mean) * mask) ** 2).sum(dim=(1, 2, 3), keepdim=True) / (mask_num + 1e-8)
        return mean, torch.sqrt(var + 1e-16)
    
    def get_norm_image_multi_cam(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()

            s_mean, s_std = self.get_mean_std_multi_cam(src_img, mask)
            w_mean, w_std = self.get_mean_std_multi_cam(warp_img, mask)

        norm_warp = (warp_img - w_mean) / w_std * s_std + s_mean
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

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / w_std * s_std + s_mean
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

    def get_virtual_image_multi_cam(self, color, mask, depth, invK, K, cam_T_cam, scale=0, img_warp_pad_mode='zeros'):
        """
                This function warps source image to target image using backprojection and reprojection process.
                """
        # do reconstruction for target from source
        pix_coords = self.project.forward_temp(depth, cam_T_cam, invK, K)
        pix_coords = pix_coords.view(-1, *pix_coords.shape[-3:])

        img_warped = F.grid_sample(color.squeeze(0), pix_coords, mode='bilinear', padding_mode=img_warp_pad_mode, align_corners=True)
        mask_warped = F.grid_sample(mask.squeeze(0), pix_coords, mode='nearest', padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords_dims = len(pix_coords.shape)
        pix_coords = pix_coords.permute(*range(pix_coords_dims - 3), pix_coords_dims - 1, pix_coords_dims - 3,
                                        pix_coords_dims - 2)
        invalid_mask = torch.logical_or(pix_coords > 1,
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        mask_warped = ((~invalid_mask).float() * mask_warped)

        return img_warped.unsqueeze(0), mask_warped.unsqueeze(0)

    def pred_all_cam_imgs(self, inputs, outputs, spt_rel_poses):
        scale = 0
        ref_depth = torch.concat([outputs[('cam', i)][('depth', scale)] for i in range(6)], dim=0).unsqueeze(0)  # 1,6,1,384,640
        outputs[('depth_multi_cam', scale)] = ref_depth
        ref_mask = inputs['mask']
        ref_invK = inputs[('inv_K', scale)]  # 1,6,4,4
        ref_K = inputs[('K', scale)]
        spatio_left_order = [1, 3, 0, 5, 2, 4]
        spatio_right_order = [2, 0, 4, 1, 5, 3]

        # 1.temporal
        for frame_id in self.frame_ids[1:]:
            spt_rel_pose = spt_rel_poses[('temporal', frame_id)]
            warped_img, warped_mask = self.get_virtual_image_multi_cam(inputs['color', frame_id, scale], inputs['mask'],
                                                                  ref_depth, ref_invK, inputs[('K', scale)],
                                                                  spt_rel_pose,
                                                                  scale, img_warp_pad_mode='border')
            self.build_cam_outputs(outputs, warped_img, warped_mask, 'color', frame_id)

        # 2.spatio and temporal
        for frame_id in self.frame_ids:
            overlap_img = torch.zeros_like(inputs['color', frame_id, scale])
            overlap_mask = torch.zeros_like(inputs['mask'])
            use_depth_consistency = hasattr(self, 'spatial_depth_consistency_loss_weight')
            overlap_depth = torch.zeros_like(ref_depth) if use_depth_consistency else None
            for direction in ['left', 'right']:
                spt_rel_pose = spt_rel_poses[('spatio', direction)] if frame_id == 0 else spt_rel_poses[
                    ('spatio_temporal', direction, frame_id)]
                order = spatio_left_order if direction == 'left' else spatio_right_order
                src_color = inputs['color', frame_id, scale][:, order, ...]
                src_mask = ref_mask[:, order, ...]
                src_K = ref_K[:, order, ...]
                warped_img, warped_mask = self.get_virtual_image_multi_cam(src_color, src_mask, ref_depth, ref_invK, src_K,
                                                                      spt_rel_pose, scale)
                if self.intensity_align:
                    warped_img = self.get_norm_image_multi_cam(inputs['color', frame_id, scale], inputs['mask'],
                                                            warped_img, warped_mask)
                overlap_img += warped_img
                overlap_mask += warped_mask

                if use_depth_consistency and frame_id == 0:
                    src_depth = ref_depth[:, order, ...]
                    src_invK = ref_invK[:, order, ...]
                    src_depth_tar_view = self.project.transform_depth_multi_cam(src_depth,
                                                                                torch.linalg.inv(spt_rel_pose),
                                                                                src_invK,
                                                                                ref_K)[:, :, 2:]
                    warped_depth, warped_mask = self.get_virtual_image_multi_cam(src_depth_tar_view, src_mask, ref_depth,
                                                                       ref_invK, src_K, spt_rel_pose, scale)
                    overlap_depth = overlap_depth + warped_depth

            self.build_cam_outputs(outputs, overlap_img, overlap_mask, 'overlap', frame_id, overlap_depth=overlap_depth)

    def build_cam_outputs(self, outputs, warped_img, warped_mask, warp_mode, frame_id, overlap_depth=None):
        outputs[(warp_mode, frame_id, 0)] = warped_img
        outputs[(warp_mode + '_mask', frame_id, 0)] = warped_mask
        if overlap_depth is not None:
            outputs[(warp_mode + '_depth', frame_id, 0)] = overlap_depth

    def forward(self, inputs, outputs, cam, rel_pose_dict):
        # predict images for each scale(default = scale 0 only)
        scale = source_scale = 0
        
        # ref inputs
        ref_color = inputs['color', 0, source_scale][:,cam, ...]        
        ref_mask = inputs['mask'][:, cam, ...]
        ref_K = inputs[('K', source_scale)][:,cam, ...]
        ref_invK = inputs[('inv_K', source_scale)][:,cam, ...]  
        
        # output
        target_view = outputs[('cam', cam)]

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

            if self.temporal_intensity_align:
                warped_img = self.get_norm_image_single(
                    ref_color,
                    ref_mask,
                    warped_img,
                    warped_mask
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
                    if self.with_eq:
                        src_color = inputs['color_eq', frame_id, source_scale][:, cur_index, ...]
                    else:
                        src_color = inputs['color', frame_id, source_scale][:, cur_index, ...]
                    src_mask = inputs['mask'][:, cur_index, ...]
                    src_K = inputs[('K', source_scale)][:, cur_index, ...]

                    rel_pose = rel_pose_dict[(frame_id, cur_index)]

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

        outputs[('cam', cam)] = target_view