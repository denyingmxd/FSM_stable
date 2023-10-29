import os
import torch
import pickle
mode='train'
with open('/data/laiyan/airs/SurroundDepth/datasets/ddad/{}.txt'.format(mode), 'r') as f:
    # self.filenames = list(set(f.readlines()))
    vf_filenames = f.readlines()
    vf_filenames = [r.strip() for r in vf_filenames]
with open('/data/laiyan/ssd/ddad/meta_data/info_{}.pkl'.format(mode), 'rb') as f:
    info = pickle.load(f)
with open('./train_r3d3.txt', 'r') as f:
    # self.filenames = list(set(f.readlines()))
    r3d3_filenames = f.readlines()
    r3d3_filenames = [r.strip() for r in r3d3_filenames]
vf_scenes = [v['scene_name'] for k,v in info.items()]


clean_vf_filenames = []
for filename in vf_filenames:
    k,v = filename, info[filename]
    if v['scene_name'] not in r3d3_filenames:
        continue
    clean_vf_filenames.append(filename)

with open('/data/laiyan/codes/FSM_stable/dataset/ddad/train_r3d3.txt','w') as f:
    for line in clean_vf_filenames:
        f.write(f"{line}\n")







