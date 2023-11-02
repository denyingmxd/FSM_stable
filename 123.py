# import os
# import torch
# import pickle
# mode='train'
# with open('/data/laiyan/airs/SurroundDepth/datasets/ddad/{}.txt'.format(mode), 'r') as f:
#     # self.filenames = list(set(f.readlines()))
#     vf_filenames = f.readlines()
#     vf_filenames = [r.strip() for r in vf_filenames]
# with open('/data/laiyan/ssd/ddad/meta_data/info_{}.pkl'.format(mode), 'rb') as f:
#     info = pickle.load(f)
# with open('./train_r3d3.txt', 'r') as f:
#     # self.filenames = list(set(f.readlines()))
#     r3d3_filenames = f.readlines()
#     r3d3_filenames = [r.strip() for r in r3d3_filenames]
# vf_scenes = [v['scene_name'] for k,v in info.items()]
#
#
# clean_vf_filenames = []
# for filename in vf_filenames:
#     k,v = filename, info[filename]
#     if v['scene_name'] not in r3d3_filenames:
#         continue
#     clean_vf_filenames.append(filename)
#
# with open('/data/laiyan/codes/FSM_stable/dataset/ddad/train_r3d3.txt','w') as f:
#     for line in clean_vf_filenames:
#         f.write(f"{line}\n")

import os
import numpy as np
import pickle
from PIL import Image
cams = ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']

data = {}

data['rgb'] = {}
for cam in cams:
    data['rgb'][cam] = np.array(Image.open('/data/kye/data/SSSDE/DDAD/DDAD/000000/rgb_1216_1936/{}/15621787638931470.jpg'.format(cam)))

data['depth'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/projected/depth_1216_1936/lidar/CAMERA_01/15621787638931470.npz")
data['extrinsics'] = {}
data['extrinsics']['CAMERA_01'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/extrinsics/CAMERA_01/15621787638931470.npy")
data['extrinsics']['CAMERA_05'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/extrinsics/CAMERA_05/15621787638931470.npy")
data['extrinsics']['CAMERA_06'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/extrinsics/CAMERA_06/15621787638931470.npy")
data['extrinsics']['CAMERA_07'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/extrinsics/CAMERA_07/15621787638931470.npy")
data['extrinsics']['CAMERA_08'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/extrinsics/CAMERA_08/15621787638931470.npy")
data['extrinsics']['CAMERA_09'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/extrinsics/CAMERA_09/15621787638931470.npy")
data['intrinsics'] = {}
data['intrinsics']['CAMERA_01'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/intrinsics_1216_1936/CAMERA_01/15621787638931470.npy")
data['intrinsics']['CAMERA_05'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/intrinsics_1216_1936/CAMERA_05/15621787638931470.npy")
data['intrinsics']['CAMERA_06'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/intrinsics_1216_1936/CAMERA_06/15621787638931470.npy")
data['intrinsics']['CAMERA_07'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/intrinsics_1216_1936/CAMERA_07/15621787638931470.npy")
data['intrinsics']['CAMERA_08'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/intrinsics_1216_1936/CAMERA_08/15621787638931470.npy")
data['intrinsics']['CAMERA_09'] = np.load("/data/kye/data/SSSDE/DDAD/DDAD/000000/intrinsics_1216_1936/CAMERA_09/15621787638931470.npy")





