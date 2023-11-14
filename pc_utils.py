import open3d
import os
import numpy as np
import torch
import torch.nn.functional as F
import struct
import torch.nn as nn
def get_virtual_image_multi_cam(color, mask, depth, invK, K, cam_T_cam, scale=0, img_warp_pad_mode='zeros',
                                cam_points=None,project=None):
    """
            This function warps source image to target image using backprojection and reprojection process.
            """
    # do reconstruction for target from source
    if cam_points is None:
        pix_coords =project.forward_multi_cam(depth, cam_T_cam, invK, K)
    else:
        pix_coords = project.reproject_multi_cam(K, cam_points, cam_T_cam)
    pix_coords = pix_coords.view(-1, *pix_coords.shape[-3:])

    img_warped = F.grid_sample(color.squeeze(0), pix_coords, mode='bilinear', padding_mode=img_warp_pad_mode,
                               align_corners=True)
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

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))

    fid.close()


class Projection(nn.Module):
    """
    This class computes projection and reprojection function.
    """

    def __init__(self, batch_size, height, width, device):
        super().__init__()
        self.batch_size = batch_size
        self.width = width
        self.height = height

        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points.to(device)

        self.to_homo = torch.ones([batch_size, 1, width * height]).to(device)
        self.homo_points = torch.cat([img_points, self.to_homo], 1)

    def backproject(self, invK, depth):
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(self.batch_size, 1, -1)

        points3D = torch.matmul(invK[:, :3, :3], self.homo_points)
        points3D = depth * points3D
        return torch.cat([points3D, self.to_homo], 1)

    def reproject(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        # project points
        points2D = (K @ T)[:, :3, :] @ points3D
        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :] / (points2D[:, 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(self.batch_size, 2, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        norm_points2D[..., 0] /= self.width - 1
        norm_points2D[..., 1] /= self.height - 1
        norm_points2D = (norm_points2D - 0.5) * 2
        return norm_points2D

    def reproject_unnormed(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """

        # project points
        points2D = (K @ T)[:, :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D[:, :2, :] /= (points2D[:, 2:, :] + 1e-7)
        norm_points2D = points2D
        norm_points2D = norm_points2D.view(self.batch_size, 3, self.height, self.width)
        norm_points2D = norm_points2D.permute(0, 2, 3, 1)

        bs = points2D.shape[0]
        aaaas = []
        for b in range(bs):
            local_norm_points2D = norm_points2D[b].reshape(-1, 3)
            zz = local_norm_points2D[:, 2:]
            # local_norm_points2D = local_norm_points2D.detach().clone()
            local_norm_points2D[:, 0] = torch.round(local_norm_points2D[:, 0]) - 1
            local_norm_points2D[:, 1] = torch.round(local_norm_points2D[:, 1]) - 1
            val_inds = (local_norm_points2D[:, 0] >= 0) & (local_norm_points2D[:, 1] >= 0)
            val_inds = val_inds & (local_norm_points2D[:, 0] < self.width) & (local_norm_points2D[:, 1] < self.height)
            local_norm_points2D = local_norm_points2D[val_inds, :]
            zz = zz[val_inds, :]
            aaa = torch.zeros((self.height, self.width), device=points3D.device, dtype=points3D.dtype)
            aaa[local_norm_points2D[:, 1].long(), local_norm_points2D[:, 0].long()] = zz[:, 0]

            aaaas.append(aaa.unsqueeze(0))
        aaaas = torch.stack(aaaas)

        return aaaas

    def reproject_transform(self, K, points3D, T):
        points2D = (K @ T)[:, :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D = points2D
        points2D = points2D.view(self.batch_size, 3, self.height, self.width)
        return points2D

    def reproject_transform_multi_cam(self, K, points3D, T):
        points2D = (K @ T)[..., :3, :] @ points3D

        # normalize projected points for grid sample function
        points2D = points2D.view(*points2D.shape[:-1], self.height, self.width)
        return points2D

    def forward(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject(rp_K, cam_points, T)
        return pix_coords

    def backproject_multi_cam(self, invK, depth):
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(*depth.shape[:-2], -1)  # 1, 6, 1, 245760
        points3D = torch.matmul(invK[..., :3, :3], self.homo_points.unsqueeze(0))  # 1, 6, 3, 245760
        points3D = depth * points3D  # 1, 6, 3, 245760
        return torch.cat([points3D, self.to_homo.unsqueeze(0).expand(-1, 6, -1, -1)], 2)

    def reproject_multi_cam(self, K, points3D, T):
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        """
        # project points
        points2D = (K @ T)[..., :3, :] @ points3D
        # normalize projected points for grid sample function
        norm_points2D = points2D[..., :2, :] / (points2D[..., 2:, :] + 1e-7)
        norm_points2D = norm_points2D.view(*norm_points2D.shape[:-1], self.height, self.width)
        norm_points2D_dims = len(norm_points2D.shape)
        norm_points2D = norm_points2D.permute(*range(norm_points2D_dims - 3), norm_points2D_dims - 2,
                                              norm_points2D_dims - 1, norm_points2D_dims - 3)
        norm_points2D[..., 0] /= self.width - 1
        norm_points2D[..., 1] /= self.height - 1
        norm_points2D.sub_(0.5).mul_(2)
        return norm_points2D

    def forward_multi_cam(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject_multi_cam(bp_invK, depth)  # 1, 6, 4, 245760

        pix_coords = self.reproject_multi_cam(rp_K, cam_points, T)
        return pix_coords

    def get_unnormed_projects(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject_unnormed(rp_K, cam_points, T)
        return pix_coords

    def transform_depth(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)

        pix_coords = self.reproject_transform(rp_K, cam_points, T)
        return pix_coords

    def transform_depth_multi_cam(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject_multi_cam(bp_invK, depth)

        pix_coords = self.reproject_transform_multi_cam(rp_K, cam_points, T)
        return pix_coords
