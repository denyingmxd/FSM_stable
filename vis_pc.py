import open3d
import numpy as np

def processing(disp, K, T, image):
    depth = 1.0 / disp.squeeze()  # h, w
    height, width = depth.shape
    img = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(
        width, height), interpolation=cv2.INTER_LINEAR)

    fx, fy, cx, cy = K
    intrinsics_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])
    intrinsics_matrix_homo = np.zeros((3, 4), dtype=np.float32)
    intrinsics_matrix_homo[:3, :3] = intrinsics_matrix
    intrinsics_matrix_homo[2, 3] = 1

    pose = pose_quaternion_to_matrix(T)

    points = []
    rgbs = []
    for v in range(height):
        for u in range(width):
            d = depth[v, u]

            if d == 0: continue

            x_cam = (u - cx) / fx * d
            y_cam = (v - cy) / fy * d
            z_cam = d
            points.append([x_cam, y_cam, z_cam])
            rgbs.append(img[v, u, :])

    rgbs = np.array(rgbs)
    points = np.array(points)
    ones = np.ones((points.shape[0], 1))
    points = np.hstack((points, ones))  # homo

    pc_world = (pose @ points.T).T

    return points, pc_world, intrinsics_matrix_homo, pose, rgbs


cam_npy = np.load(cam_npz)
img = cv2.imread(cam_img)
disp = cam_npy['disp_up']
K = cam_npy['intrinsics']
T = cam_npy['pose']

points, pc_world, k, t, rgbs = processing(disp=disp, K=K, T=T, image=img)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_world[:, :3])
pcd.colors = o3d.utility.Vector3dVector(rgbs / 255.0)
o3d.visualization.draw_geometries([pcd])