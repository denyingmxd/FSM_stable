# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn.functional as F
    

def compute_auto_masks(reprojection_loss, identity_reprojection_loss):
    """ 
    This function computes auto mask using reprojection loss and identity reprojection loss.
    """
    if identity_reprojection_loss is None:
        # without using auto(identity loss) mask
        reprojection_loss_mask = torch.ones_like(reprojection_loss)
    else:
        # using auto(identity loss) mask
        losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=1)  # 1,2,384,640
        idxs = torch.argmin(losses, dim=1, keepdim=True)  # 1,1,384,640
        reprojection_loss_mask = (idxs == 0).float()
    return reprojection_loss_mask

def compute_auto_masks_multi_cam(reprojection_loss, identity_reprojection_loss):
    """
    This function computes auto mask using reprojection loss and identity reprojection loss.
    """
    if identity_reprojection_loss is None:
        # without using auto(identity loss) mask
        reprojection_loss_mask = torch.ones_like(reprojection_loss)
    else:
        # using auto(identity loss) mask
        losses = torch.cat([reprojection_loss, identity_reprojection_loss], dim=2)  # 1,6,2,384,640
        idxs = torch.argmin(losses, dim=2, keepdim=True)  # 1,6,1,384,640
        reprojection_loss_mask = (idxs == 0).float()
    return reprojection_loss_mask


def compute_masked_loss(loss, mask):    
    """
    This function masks losses while avoiding zero division.
    """    
    return (loss * mask).sum() / (mask.sum() + 1e-8)

def compute_edg_smooth_loss_multi_cam(rgb, disp_map):
    """
    This function calculates edge-aware smoothness.
    """
    grad_rgb_x = (rgb[..., :-1] - rgb[..., 1:]).abs().mean(2, True)  # 1,6,1,384,639
    grad_rgb_y = (rgb[..., :-1, :] - rgb[..., 1:, :]).abs().mean(2, True)  # 1,6,1,383,640

    grad_disp_x = (disp_map[..., :-1] - disp_map[..., 1:]).abs()  # 1,6,1,384,639
    grad_disp_y = (disp_map[..., :-1, :] - disp_map[..., 1:, :]).abs()  # 1,6,1,383,640

    grad_disp_x *= (-1.0 * grad_rgb_x).exp()  # 1,6,1,384,639
    grad_disp_y *= (-1.0 * grad_rgb_y).exp()  # 1,6,1,383,640
    return grad_disp_x.mean() + grad_disp_y.mean()

def compute_edg_smooth_loss(rgb, disp_map):
    """
    This function calculates edge-aware smoothness.
    """
    grad_rgb_x = (rgb[:, :, :, :-1] - rgb[:, :, :, 1:]).abs().mean(1, True)  # 1,1,384,639
    grad_rgb_y = (rgb[:, :, :-1, :] - rgb[:, :, 1:, :]).abs().mean(1, True)  # 1,1,383,640

    grad_disp_x = (disp_map[:, :, :, :-1] - disp_map[:, :, :, 1:]).abs()  # 1,1,384,639
    grad_disp_y = (disp_map[:, :, :-1, :] - disp_map[:, :, 1:, :]).abs()  # 1,1,383,640

    grad_disp_x *= (-1.0 * grad_rgb_x).exp()  # 1,1,384,639
    grad_disp_y *= (-1.0 * grad_rgb_y).exp()  # 1,1,383,640
    return grad_disp_x.mean() + grad_disp_y.mean()
    # return grad_disp_x, grad_disp_y


def compute_ssim_loss_multi_cam(pred, target):
    """
    This function calculates SSIM loss between predicted image and target image.
    """
    B, D, C, H, W = pred.shape
    ref_pad = torch.nn.ReflectionPad2d(1)
    pred = ref_pad(pred.view(-1, C, H, W))
    target = ref_pad(target.view(-1, C, H, W))

    mu_pred = F.avg_pool2d(pred, kernel_size = 3, stride = 1)
    mu_target = F.avg_pool2d(target, kernel_size = 3, stride = 1)

    musq_pred = mu_pred.pow(2)
    musq_target = mu_target.pow(2)
    mu_pred_target = mu_pred*mu_target

    sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size = 3, stride = 1)-musq_pred
    sigma_target = F.avg_pool2d(target.pow(2), kernel_size = 3, stride = 1)-musq_target
    sigma_pred_target = F.avg_pool2d(pred*target, kernel_size = 3, stride = 1)-mu_pred_target

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu_pred_target + C1)*(2*sigma_pred_target + C2)) \
                    /((musq_pred + musq_target + C1)*(sigma_pred + sigma_target + C2)+1e-8)
    return torch.clamp((1-ssim_map)/2, 0, 1).view(B, D, C, H, W)

def compute_ssim_loss(pred, target):
    """
    This function calculates SSIM loss between predicted image and target image.
    """
    ref_pad = torch.nn.ReflectionPad2d(1)
    pred = ref_pad(pred)  # 1,3,386,642
    target = ref_pad(target)  # 1,3,386,642

    mu_pred = F.avg_pool2d(pred, kernel_size = 3, stride = 1)  # 1,3,384,640
    mu_target = F.avg_pool2d(target, kernel_size = 3, stride = 1)  # 1,3,384,640

    musq_pred = mu_pred.pow(2)  # 1,3,384,640
    musq_target = mu_target.pow(2)  # 1,3,384,640
    mu_pred_target = mu_pred*mu_target  # 1,3,384,640

    sigma_pred = F.avg_pool2d(pred.pow(2), kernel_size = 3, stride = 1)-musq_pred  # 1,3,384,640
    sigma_target = F.avg_pool2d(target.pow(2), kernel_size = 3, stride = 1)-musq_target  # 1,3,384,640
    sigma_pred_target = F.avg_pool2d(pred*target, kernel_size = 3, stride = 1)-mu_pred_target  # 1,3,384,640

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu_pred_target + C1)*(2*sigma_pred_target + C2)) \
                    /((musq_pred + musq_target + C1)*(sigma_pred + sigma_target + C2)+1e-8)  # 1,3,384,640
    return torch.clamp((1-ssim_map)/2, 0, 1)


def compute_photometric_loss_multi_cam(pred=None, target=None):
    """
    This function calculates photometric reconstruction loss (0.85*SSIM + 0.15*L1)
    """
    abs_diff = torch.abs(target - pred)  # 1,6,3,384,640
    l1_loss = abs_diff.mean(2, True)  # 1,6,1,384,640
    ssim_loss = compute_ssim_loss_multi_cam(pred, target).mean(2, True)  # 1,6,1,384,640
    rep_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return rep_loss

def compute_photometric_loss(pred=None, target=None):
    """
    This function calculates photometric reconstruction loss (0.85*SSIM + 0.15*L1)
    """
    abs_diff = torch.abs(target - pred)  # 1,3,384,640
    l1_loss = abs_diff.mean(1, True)  # 1,1,384,640
    ssim_loss = compute_ssim_loss(pred, target).mean(1, True)  # 1,1,384,640
    rep_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    return rep_loss
