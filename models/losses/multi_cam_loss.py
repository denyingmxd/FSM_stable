# Copyright (c) 2023 42dot. All rights reserved.
import torch
from pytorch3d.transforms import matrix_to_euler_angles
import matplotlib.pyplot as plt

from .loss_util import (compute_photometric_loss, compute_masked_loss,
                        compute_photometric_loss_multi_cam, compute_masked_loss_multi_cam)

from .single_cam_loss import SingleCamLoss


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
        spatio_loss = compute_photometric_loss_multi_cam(**loss_args)  # 1,6,1,384,640

        outputs[('overlap_mask', 0, scale)] = spatio_mask
        outputs[('sp_loss', 0, scale)] = spatio_loss
        return compute_masked_loss_multi_cam(spatio_loss, spatio_mask)

    def compute_spatio_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        """
        This function computes spatial loss.
        """
        # self occlusion mask * overlap region mask

        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]


        if hasattr(self,'vidar_mask') and self.vidar_mask=='vertical':
            b, c, h, w = spatio_mask.shape
            vidar_horizontal_mask = torch.ones_like(spatio_mask)
            vidar_horizontal_mask[:, :, :int(0.15 * h), :] = 0
            vidar_horizontal_mask[:, :, int(0.75 * h):, :] = 0
            spatio_mask *= vidar_horizontal_mask
        if hasattr(self,'vidar_mask') and self.vidar_mask=='horizontal':
            b, c, h, w = spatio_mask.shape
            vidar_vertical_mask = torch.ones_like(spatio_mask)
            vidar_vertical_mask[:, :, :, :int(0.15 * w)] = 0
            vidar_vertical_mask[:, :, :, int(0.75 * w):] = 0
            spatio_mask *= vidar_vertical_mask
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color', 0, 0][:, cam, ...]
        }
        spatio_loss = compute_photometric_loss(**loss_args)

        target_view[('overlap_mask', 0, scale)] = spatio_mask
        target_view[('sp_loss', 0, scale)] = spatio_loss
        return compute_masked_loss(spatio_loss, spatio_mask)

    def compute_spatio_tempo_loss_multi_cam(self, inputs, outputs, scale=0, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = inputs['mask'] * outputs[('overlap_mask', frame_id, scale)]  # 1,6,1,384,640
            pred_mask = pred_mask * reproj_loss_mask  # 1,6,1,384,640

            loss_args = {
                'pred': outputs[('overlap', frame_id, scale)],
                'target': inputs['color', 0, 0]
            }

            spatio_tempo_losses.append(compute_photometric_loss_multi_cam(**loss_args))
            spatio_tempo_masks.append(pred_mask)

        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 2)  # 1,6,2,384,640
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 2)  # 1,6,2,384,640

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, reprojection_loss_min_index = torch.min(spatio_tempo_losses, dim=2, keepdim=True)  # 1,6,1,384,640
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=2, keepdim=True)  # 1,6,1,384,640

        return compute_masked_loss_multi_cam(spatio_tempo_loss, spatio_tempo_mask)

    def compute_spatio_tempo_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]  # 1,1,384,640
            pred_mask = pred_mask * reproj_loss_mask  # 1,1,384,640

            if hasattr(self, 'vidar_mask') and self.vidar_mask == 'vertical':
                b, c, h, w = pred_mask.shape
                vidar_horizontal_mask = torch.ones_like(pred_mask)
                vidar_horizontal_mask[:, :, :int(0.15 * h), :] = 0
                vidar_horizontal_mask[:, :, int(0.75 * h):, :] = 0
                pred_mask *= vidar_horizontal_mask
            if hasattr(self, 'vidar_mask') and self.vidar_mask == 'horizontal':
                b, c, h, w = pred_mask.shape
                vidar_vertical_mask = torch.ones_like(pred_mask)
                vidar_vertical_mask[:, :, :, :int(0.15 * w)] = 0
                vidar_vertical_mask[:, :, :, int(0.75 * w):] = 0
                pred_mask *= vidar_vertical_mask


            loss_args = {
                'pred': target_view[('overlap', frame_id, scale)],
                'target': inputs['color', 0, 0][:, cam, ...]
            }

            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)

        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)  # 1,2,384,640
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)  # 1,2,384,640

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, reprojection_loss_min_index = torch.min(spatio_tempo_losses, dim=1, keepdim=True)  # 1,1,384,640
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)  # 1,1,384,640

        target_view[('overlap_mask', -1, scale)] = spatio_tempo_mask * target_view[('overlap_mask', -1, scale)]
        target_view[('overlap_mask', 1, scale)] = spatio_tempo_mask * target_view[('overlap_mask', 1, scale)]

        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask)

    def compute_spatial_depth_consistency_loss_multi_cam(self, inputs, outputs, scale=0, reproj_loss_mask=None):
        spatio_mask = inputs['mask'] * outputs[('overlap_mask', 0, scale)]  # 1,6,1,384,640
        loss_args = {
            'pred': outputs[('overlap_depth', 0, scale)],
            'target': outputs[('depth_multi_cam', scale)]
        }
        spatio_depth_consistency_loss = torch.abs(loss_args['pred']-loss_args['target'])  # 1,6,1,384,640

        depth_con_mask = spatio_mask * (loss_args['pred'] > 0)  # 1,6,1,384,640
        return compute_masked_loss_multi_cam(spatio_depth_consistency_loss, depth_con_mask)

    def compute_spatial_depth_consistency_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None):
        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]  # 1,1,384,640
        loss_args = {
            'pred': target_view[('overlap_depth', 0, scale)],
            'target': target_view[('depth', scale)]
        }
        spatio_depth_consistency_loss = torch.abs(loss_args['pred']-loss_args['target'])  # 1,1,384,640

        depth_con_mask = spatio_mask * (loss_args['pred'] > 0)  # 1,1,384,640
        if hasattr(self, 'spatial_depth_consistency_margin'):
            margin = self.spatial_depth_consistency_margin
            depth_con_mask*=spatio_depth_consistency_loss<margin
        return compute_masked_loss(spatio_depth_consistency_loss, depth_con_mask)


    def compute_sp_tp_recon_con_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None):
        sp_tp_recon_con_loss = 0
        if hasattr(self,'sptp_recon_con_type'):
            if self.sptp_recon_con_type=='combine':
                pred_mask = ref_mask * target_view[('overlap_mask', -1, scale)]
                pred_mask = pred_mask * reproj_loss_mask
                pred_mask = pred_mask * target_view[('overlap_mask', 0, scale)]
                pred_mask = pred_mask * target_view[('overlap_mask', 1, scale)]

                loss_args = {
                    'pred': target_view[('stp_reproj_combined', scale)],
                    'target': target_view[('overlap', 0, scale)],
                }
                local_loss = compute_photometric_loss(**loss_args)

                sp_tp_recon_con_loss = sp_tp_recon_con_loss + compute_masked_loss(local_loss, pred_mask)
                target_view[('sp_tp_recon_con_loss', scale, -1)] = local_loss * pred_mask
                target_view[('sp_tp_recon_con_loss', scale, 1)] = local_loss * pred_mask
                return sp_tp_recon_con_loss
        else:
            for frame_id in self.frame_ids[1:]:
                pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]
                pred_mask = pred_mask * reproj_loss_mask
                pred_mask = pred_mask * target_view[('overlap_mask', 0, scale)]

                loss_args = {
                    'pred': target_view[('overlap', frame_id, scale)],
                    'target': target_view[('overlap', 0, scale)],
                }
                local_loss = compute_photometric_loss(**loss_args)

                sp_tp_recon_con_loss = sp_tp_recon_con_loss+ compute_masked_loss(local_loss,pred_mask)
                target_view[('sp_tp_recon_con_loss', scale,frame_id)] = local_loss*pred_mask
            return sp_tp_recon_con_loss/len(self.frame_ids[1:])



    def compute_pose_con_loss(self, inputs, outputs, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """
        ref_output = outputs[('cam', 0)]
        ref_ext = inputs['extrinsics'][:, 0, ...]
        ref_ext_inv = inputs['extrinsics_inv'][:, 0, ...]

        cur_output = outputs[('cam', cam)]
        cur_ext = inputs['extrinsics'][:, cam, ...]
        cur_ext_inv = inputs['extrinsics_inv'][:, cam, ...]

        trans_loss = 0.
        angle_loss = 0.

        for frame_id in self.frame_ids[1:]:
            ref_T = ref_output[('cam_T_cam', 0, frame_id)]
            cur_T = cur_output[('cam_T_cam', 0, frame_id)]

            cur_T_aligned = ref_ext_inv@cur_ext@cur_T@cur_ext_inv@ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:,:3,:3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:,:3,:3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:,:3,3] - cur_T_aligned[:,:3,3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff

        pose_loss = (trans_loss + 10 * angle_loss) / len(self.frame_ids[1:])
        return pose_loss

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
        cam_loss += reprojection_loss
        cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
        cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss
        if hasattr(self, 'spatial_depth_consistency_loss_weight'):
            cam_loss += self.spatial_depth_consistency_loss_weight * spatial_depth_consistency_loss

        loss_dict['reproj_loss'] = reprojection_loss.item()
        loss_dict['spatio_loss'] = spatio_loss.item()
        loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
        loss_dict['smooth'] = smooth_loss.item()
        if hasattr(self, 'spatial_depth_consistency_loss_weight'):
            loss_dict['spatial_depth_consistency_loss'] = spatial_depth_consistency_loss.item() * self.spatial_depth_consistency_loss_weight

        self.get_logs_multi_cam(loss_dict, outputs)
        cam_loss /= len(self.scales)
        return cam_loss, loss_dict


    def forward(self, inputs, outputs, cam):
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }

            reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)

            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
            spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs)

            kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
            spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)


            if self.pose_model == 'fsm' and cam != 0:  # ignore
                pose_loss = self.compute_pose_con_loss(inputs, outputs, **kargs)
                cam_loss += self.pose_con_coeff * pose_loss
            else:
                pose_loss = 0

            if hasattr(self, 'spatial_depth_consistency_loss_weight'):
                spatial_depth_consistency_loss = self.compute_spatial_depth_consistency_loss(inputs,target_view,**kargs)
            if hasattr(self, 'sp_tp_recon_con_loss_weight'):  # ignore
                sp_tp_recon_con_loss = self.compute_sp_tp_recon_con_loss(inputs, target_view,**kargs)
            if hasattr(self, 'spatial_depth_aug_smoothness'):  # ignore
                spatial_depth_aug_smooth_loss = self.compute_spatial_depth_aug_smooth_loss(inputs,target_view,**kargs)

            cam_loss += reprojection_loss
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss

            if hasattr(self, 'spatial_depth_consistency_loss_weight') :
                cam_loss += self.spatial_depth_consistency_loss_weight * spatial_depth_consistency_loss
            if hasattr(self, 'sp_tp_recon_con_loss_weight'):
                cam_loss += self.sp_tp_recon_con_loss_weight * sp_tp_recon_con_loss
            if hasattr(self,'spatial_depth_aug_smoothness') :
                cam_loss += self.spatial_depth_aug_smoothness * spatial_depth_aug_smooth_loss

            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['reproj_loss'] = reprojection_loss.item()
                loss_dict['spatio_loss'] = spatio_loss.item()
                loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
                loss_dict['smooth'] = smooth_loss.item()
                # loss_dict['reproj_loss'] = reprojection_loss
                # loss_dict['spatio_loss'] = spatio_loss
                # loss_dict['spatio_tempo_loss'] = spatio_tempo_loss
                # loss_dict['smooth'] = smooth_loss

                if self.pose_model == 'fsm' and cam != 0:
                    loss_dict['pose'] = pose_loss.item()
                if hasattr(self, 'spatial_depth_consistency_loss_weight') :
                    loss_dict['spatial_depth_consistency_loss'] = spatial_depth_consistency_loss.item() * self.spatial_depth_consistency_loss_weight
                if hasattr(self, 'sp_tp_recon_con_loss_weight'):
                    loss_dict['sp_tp_recon_con_loss'] = sp_tp_recon_con_loss.item() * self.sp_tp_recon_con_loss_weight
                if hasattr(self, 'spatial_depth_aug_smoothness'):
                    loss_dict['spatial_depth_aug_smooth_loss'] = spatial_depth_aug_smooth_loss.item() * self.spatial_depth_aug_smoothness
                # log statistics
                self.get_logs(loss_dict, target_view, cam)

        cam_loss /= len(self.scales)
        return cam_loss, loss_dict