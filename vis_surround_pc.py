import pickle
import open3d
import numpy as np
import matplotlib.pyplot as plt
import torch
import io
from  pc_utils import *
import open3d as o3d
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open('./example.pkl','rb') as f:
    # data = pickle.load(f)
    contents = CPU_Unpickler(f).load()

rgbs = contents[('color',0,0)][0]
rgbs = torch.nn.UpsamplingBilinear2d((1216,1936))(rgbs)



gt_depths= contents['my_depth'][0]
h,w = gt_depths.shape[-2:]
predicted_depths = torch.stack([contents[('cam',i)][('depth',0)][0] for i in range(6)],axis=0)
predicted_depths = torch.nn.UpsamplingBilinear2d((1216,1936))(predicted_depths)
org_intrinsics = contents['intrinsics_org'][0]
org_extrinsics = contents['extrinsics'][0]
ref_invK = torch.inverse(org_intrinsics)
inv_extrinsics = torch.inverse(org_extrinsics)
# for i in range(6):
#     plt.imshow(rgbs[i].permute((1,2,0)))
#     gt_nonzeros = torch.nonzero(gt_depths[i][0],as_tuple=True)
#     plt.scatter(gt_nonzeros[1],gt_nonzeros[0],c = gt_depths[i][0][gt_nonzeros],s=1)
#     plt.show()
#     plt.imshow(predicted_depths[i].permute((1,2,0)));plt.show()
    # exit()
projection =  Projection(1 , h ,w ,torch.device('cpu'))
points = []
colors = []
used_depth = predicted_depths
# used_depth = gt_depths
for i in range(6):
    plt.imshow(rgbs[i].permute((1, 2, 0)))
    # gt_nonzeros = torch.nonzero(gt_depths[i][0],as_tuple=True)
    # plt.scatter(gt_nonzeros[1],gt_nonzeros[0],c = gt_depths[i][0][gt_nonzeros],s=1)
    plt.show()
    # plt.imshow(predicted_depths[i].permute((1,2,0)));plt.show()

    cam_points = projection.backproject(ref_invK[i:i+1],used_depth[i:i+1])
    T = org_extrinsics[i:i+1]
    cam_points =T[:, :4, :] @ cam_points

    valid_pos =(used_depth[i].view(-1)>0) & (gt_depths[i].view(-1)>0) & (gt_depths[i].view(-1)<50)
    valid_cam_points = cam_points[valid_pos.repeat(1,4,1)].reshape(1,4,-1)[0].permute(1,0)
    valid_color = rgbs[i].permute(1,2,0).reshape(-1,3)[valid_pos]
    points.extend(valid_cam_points)
    colors.extend(valid_color)

points = np.stack(points)
colors = np.stack(colors)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# pcd.points = o3d.utility.Vector3dVector(points[:, :3][points[:,2]<0])
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3][points[:,2]<0])

o3d.visualization.draw_geometries([pcd])


# o3d.io.write_point_cloud('all_pred.ply',pcd)




