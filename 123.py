import os
import torch
import pickle
mode='train'
with open('/data/laiyan/airs/SurroundDepth/datasets/ddad/{}.txt'.format(mode), 'r') as f:
    # self.filenames = list(set(f.readlines()))
    filenames = f.readlines()
with open('/data/laiyan/ssd/ddad/meta_data/info_{}.pkl'.format(mode), 'rb') as f:
    info = pickle.load(f)
with open('./train_r3d3.txt', 'r') as f:
    # self.filenames = list(set(f.readlines()))
    r3d3_filenames = f.readlines()
    r3d3_filenames = [r.strip() for r in r3d3_filenames]
vf_scenes = [v['scene_name'] for k,v in info.items()]
set_vf_scenes = (set(vf_scenes))

set_r3d3_scenes = (set(r3d3_filenames))
diff = list(set_vf_scenes-set_r3d3_scenes)
diff.sort()
print(diff)




