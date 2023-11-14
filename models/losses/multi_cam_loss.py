# Copyright (c) 2023 42dot. All rights reserved.
import torch
from pytorch3d.transforms import matrix_to_euler_angles
import matplotlib.pyplot as plt
import numpy as np
from .loss_util import (compute_photometric_loss, compute_masked_loss,
                        compute_photometric_loss_multi_cam, compute_masked_loss_multi_cam)

from .single_cam_loss import SingleCamLoss
from ..geometry.geometry_util import Projection

import kornia

_EPSILON = 0.00001
class MultiCamLoss(SingleCamLoss):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """
    def __init__(self, cfg, rank):
        super(MultiCamLoss, self).__init__(cfg, rank)

    def compute_spatio_loss_multi_cam(self, inputs, outputs, scale=0):
        """
        This function computes spatial loss.
        """
        # self occlusion mask * overlap region mask
        spatio_mask = inputs['mask'] * outputs[('overlap_mask', 0, scale)]  # 1,6,1,384,640
        loss_args = {
            'pred': outputs[('overlap', 0, scale)],
            'target': inputs['color', 0, 0]
        }

        if hasattr(self,'intensity_type'):
            if self.intensity_type==1:
                loss_args = {
                    'pred': kornia.color.rgb_to_yuv(outputs[('overlap', 0, scale)]),
                    'target':  kornia.color.rgb_to_yuv(inputs['color', 0, 0])
                }
            elif self.intensity_type==2:
                loss_args = {
                    'pred': kornia.color.rgb_to_yuv(outputs[('overlap', 0, scale)])[:,:,0:1,:],
                    'target':  kornia.color.rgb_to_yuv(inputs['color', 0, 0][:,:,0:1,:])
                }
            elif self.intensity_type==3:
                loss_args = {
                    'pred': kornia.color.rgb_to_yuv(outputs[('overlap', 0, scale)])[:,:,2:3,:],
                    'target':  kornia.color.rgb_to_yuv(inputs['color', 0, 0][:,:,2:3,:])
                }

        spatio_loss = compute_photometric_loss_multi_cam(**loss_args)  # 1,6,1,384,640

        outputs[('overlap_mask', 0, scale)] = spatio_mask
        outputs[('sp_loss', 0, scale)] = spatio_loss

        return compute_masked_loss_multi_cam(spatio_loss, spatio_mask)


    def compute_spatio_tempo_loss_multi_cam(self, inputs, outputs, scale=0, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = inputs['mask'] * outputs[('overlap_mask', frame_id, scale)]  # 1,6,1,384,640
            # pred_mask = pred_mask * reproj_loss_mask  # 1,6,1,384,640
            pred_mask = pred_mask  # 1,6,1,384,640

            loss_args = {
                'pred': outputs[('overlap', frame_id, scale)],
                'target': inputs['color', 0, 0]
            }
            if hasattr(self, 'intensity_type'):
                if self.intensity_type == 1:
                    loss_args = {
                        'pred': kornia.color.rgb_to_yuv(outputs[('overlap', frame_id, scale)]),
                        'target': kornia.color.rgb_to_yuv(inputs['color', 0, 0])
                    }
                elif self.intensity_type == 2:
                    loss_args = {
                        'pred': kornia.color.rgb_to_yuv(outputs[('overlap', frame_id, scale)])[:, :, 0:1, :],
                        'target': kornia.color.rgb_to_yuv(inputs['color', 0, 0][:, :, 0:1, :])
                    }
                elif self.intensity_type == 3:
                    loss_args = {
                        'pred': kornia.color.rgb_to_yuv(outputs[('overlap', frame_id, scale)])[:, :, 2:3, :],
                        'target': kornia.color.rgb_to_yuv(inputs['color', 0, 0][:, :, 2:3, :])
                    }
            spatio_tempo_losses.append(compute_photometric_loss_multi_cam(**loss_args))
            spatio_tempo_masks.append(pred_mask)

        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 2)  # 1,6,2,384,640
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 2)  # 1,6,2,384,640
        spatio_tempo_losses[spatio_tempo_masks == 0] = 999
        spatio_tempo_loss, reprojection_loss_min_index = torch.min(spatio_tempo_losses, dim=2, keepdim=True)  # 1,6,1,384,640
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=2, keepdim=True)  # 1,6,1,384,640


        outputs[('stp_loss', 0, scale)] = spatio_tempo_loss*spatio_tempo_mask
        return compute_masked_loss_multi_cam(spatio_tempo_loss, spatio_tempo_mask)

    # def calculate_

    def compute_ground_two_view_loss_multi_cam(self, inputs, outputs, scale=0, reproj_loss_mask=None):
        import open3d as o3d
        b, n, c, h, w = outputs[('overlap_depth', 0, scale)].shape
        project_imgs = Projection(b, h, w, self.rank)
        overlap_depth = outputs[('overlap_depth', 0, scale)]
        cur_depth =  outputs[('depth_multi_cam', scale)]
        bp_invK = inputs[('inv_K',0)]#1,6,4,4
        cur_ground_mask = (inputs['ground'] * inputs['mask']).view(b,n,c,h*w)
        overlap_ground_mask = (inputs['ground'] * inputs['mask'] * outputs[('overlap_mask', 0, scale)]).view(b,n,c,h*w)
        cur_points_3d = project_imgs.backproject_multi_cam(bp_invK, cur_depth)[:,:,:3,:]
        overlap_points_3d = project_imgs.backproject_multi_cam(bp_invK, overlap_depth)[:,:,:3,:]
        ground_two_view_loss = 0.
        valid_cams = 0
        for b in range(b):
            for i in range(n):
                cur_ground_mask_i = (cur_ground_mask[b,i:i+1,:]==1).view(-1)
                overlap_ground_mask_i = (overlap_ground_mask[b,i:i+1,:]==1).view(-1)
                cur_points_3d_i = (cur_points_3d[b,i:i+1,:]).permute(2,1,0).squeeze(-1)
                overlap_points_3d_i = (overlap_points_3d[b,i:i+1,:]).permute(2,1,0).squeeze(-1)
                if overlap_ground_mask_i.sum()>1000:
                    valid_cams+=1
                    overlap_ground_points = overlap_points_3d_i[overlap_ground_mask_i]
                    cur_ground_points = cur_points_3d_i[cur_ground_mask_i]

                    # overlap_valid_color = inputs[('color',0,0)][0][i].permute(1, 2, 0).reshape(-1, 3)[overlap_ground_mask_i]
                    # cur_valid_color = inputs[('color',0,0)][0][i].permute(1, 2, 0).reshape(-1, 3)[cur_ground_mask_i]
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(cur_ground_points.detach().cpu())
                    # pcd.colors = o3d.utility.Vector3dVector(cur_valid_color.detach().cpu())
                    # pcd2 = o3d.geometry.PointCloud()
                    # pcd2.points = o3d.utility.Vector3dVector(overlap_ground_points.detach().cpu())
                    # pcd2.colors = o3d.utility.Vector3dVector(overlap_valid_color.detach().cpu())
                    #
                    # o3d.visualization.draw_geometries([pcd,pcd2])
                    # print(123)

                    # cur_indexes = torch.arange(0,len(cur_ground_points),1)
                    # overlap_indexes = torch.arange(0,len(overlap_ground_points),1)

                    cur_selected_indexes = torch.randperm(len(cur_ground_points))[:200]
                    cur_selected_points = cur_ground_points[cur_selected_indexes]
                    cur_selected_points_groups = cur_selected_points.split(100,dim=0)
                    group_a,group_b = cur_selected_points_groups

                    overlap_selected_indexes = torch.randperm(len(overlap_ground_points))[:200]
                    overlap_selected_points = overlap_ground_points[overlap_selected_indexes]
                    overlap_selected_points_groups = overlap_selected_points.split(100, dim=0)
                    group_c, group_d = overlap_selected_points_groups

        #
                    line1 = group_b - group_a
                    line2 = group_c - group_a
                    line3 = group_d - group_a
                    plane_loss = torch.mean(torch.abs((torch.cross(line1,line2)*line3).sum(dim=1)))

                    ground_two_view_loss += plane_loss

        return ground_two_view_loss/valid_cams


    def compute_spatial_depth_consistency_loss_multi_cam(self, inputs, outputs, scale=0, reproj_loss_mask=None):
        spatio_mask = inputs['mask'] * outputs[('overlap_mask', 0, scale)]  # 1,6,1,384,640
        loss_args = {
            'pred': outputs[('overlap_depth', 0, scale)],
            'target': outputs[('depth_multi_cam', scale)]
        }
        spatio_depth_consistency_loss = torch.abs(loss_args['pred']-loss_args['target'])  # 1,6,1,384,640

        depth_con_mask = spatio_mask  # 1,6,1,384,640
        outputs[('depth_consistency_loss',0,scale)] = spatio_depth_consistency_loss * depth_con_mask
        return compute_masked_loss_multi_cam(spatio_depth_consistency_loss, depth_con_mask)

    def compute_spatial_normal_consistency_loss_multi_cam(self, inputs, outputs, scale=0, reproj_loss_mask=None):
        spatio_mask = inputs['mask'] * outputs[('overlap_mask', 0, scale)]  # 1,6,1,384,640
        loss_args = {
            'pred': outputs[('overlap_depth', 0, scale)],
            'target': outputs[('depth_multi_cam', scale)]
        }
        for k,v in loss_args.items():
            b,n,c,h,w = v.shape
            v = v.view(b*n,c,h,w)
            v = kornia.geometry.depth_to_normals(v,inputs[('K',0)].reshape(-1,4,4)[:,:3,:3])
            loss_args[k] = v.view(b,n,-1,h,w)

        spatio_normal_consistency_loss = torch.abs(loss_args['pred']-loss_args['target'])  # 1,6,1,384,640

        normal_con_mask = spatio_mask  # 1,6,1,384,640
        outputs[('overlap_normal', 0, scale)] = loss_args['pred']
        outputs[('normal_multi_cam', scale)] = loss_args['target']
        outputs[('normal_consistency_loss',0,scale)] = spatio_normal_consistency_loss * normal_con_mask
        return compute_masked_loss_multi_cam(spatio_normal_consistency_loss, normal_con_mask)





    def forward_multi_cam(self, inputs, outputs):
        loss_dict = {}
        scale = 0
        cam_loss = 0.
        reprojection_loss = self.compute_reproj_loss_multi_cam(inputs, outputs)
        smooth_loss = self.compute_smooth_loss_multi_cam(inputs, outputs)
        spatio_loss = self.compute_spatio_loss_multi_cam(inputs, outputs)
        spatio_tempo_loss = self.compute_spatio_tempo_loss_multi_cam(inputs, outputs,
                                                           reproj_loss_mask=outputs[('reproj_mask', scale)])
        if hasattr(self, 'spatial_depth_consistency_loss_weight'):
            spatial_depth_consistency_loss = self.compute_spatial_depth_consistency_loss_multi_cam(inputs, outputs,
                                                                reproj_loss_mask=outputs[('reproj_mask', scale)])
        if hasattr(self, 'spatial_normal_consistency_loss_weight'):
            spatial_normal_consistency_loss = self.compute_spatial_normal_consistency_loss_multi_cam(inputs, outputs,
                                                                reproj_loss_mask=outputs[('reproj_mask', scale)])
        if hasattr(self, 'ground_two_view_loss_weight'):
            ground_two_view_loss = self.compute_ground_two_view_loss_multi_cam(inputs, outputs,
                                                                reproj_loss_mask=outputs[('reproj_mask', scale)])
        cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
        cam_loss += reprojection_loss

        cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss
        if hasattr(self, 'spatial_depth_consistency_loss_weight'):
            cam_loss += self.spatial_depth_consistency_loss_weight * spatial_depth_consistency_loss
        if hasattr(self, 'spatial_normal_consistency_loss_weight'):
            cam_loss += self.spatial_normal_consistency_loss_weight * spatial_normal_consistency_loss
        if hasattr(self, 'ground_two_view_loss_weight'):
            cam_loss += self.ground_two_view_loss_weight * ground_two_view_loss

        loss_dict['reproj_loss'] = reprojection_loss.item()
        loss_dict['spatio_loss'] = spatio_loss.item()
        loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
        loss_dict['smooth'] = smooth_loss.item()
        if hasattr(self, 'spatial_depth_consistency_loss_weight'):
            loss_dict['spatial_depth_consistency_loss'] = spatial_depth_consistency_loss.item() * self.spatial_depth_consistency_loss_weight
        if hasattr(self, 'spatial_normal_consistency_loss_weight'):
            loss_dict['spatial_normal_consistency_loss'] = spatial_normal_consistency_loss.item() * self.spatial_normal_consistency_loss_weight
        if hasattr(self, 'ground_two_view_loss_weight'):
            loss_dict['ground_two_view_loss'] = ground_two_view_loss.item() * self.ground_two_view_loss_weight
        self.get_logs_multi_cam(loss_dict, outputs)
        cam_loss /= len(self.scales)
        return cam_loss, loss_dict


