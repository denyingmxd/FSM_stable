import matplotlib.pyplot as plt
from mmseg.apis import MMSegInferencer
import numpy as np
import torch
import os
cameras =  ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']

inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024',
                             weights="/data/laiyan/codes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth")

ddad_dir = '/data/laiyan/ssd/ddad/raw_data/'
# inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
start = 150
end = 200
for i in range(start,end):
    scene = str(i).zfill(6)
    scene_dir = os.path.join(ddad_dir, scene)
    rgb_dir = os.path.join(scene_dir, 'rgb')
    save_dir = os.path.join(scene_dir, 'deeplabv3plus_results')
    for cam in cameras:
        print(scene,cam)
        cam = cam.upper()
        cam_dir = os.path.join(rgb_dir,cam)
        cam_seg_dir =  os.path.join(save_dir,cam)
        os.makedirs(cam_seg_dir, exist_ok=True)
        for file in os.listdir(cam_dir):
            if file.endswith('jpg'):
                data_path = os.path.join(cam_dir,file)
                result = inferencer(data_path, show=False,return_datasamples=True)
                pred_sem_seg = result.pred_sem_seg.data.detach().cpu()
                save_path = data_path.replace('rgb','deeplabv3plus_results').replace('jpg','npz')
                np.savez_compressed(save_path, pred_sem_seg)
            # exit()


