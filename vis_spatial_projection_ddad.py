import time
import torch
import cv2
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from layers import *
from PIL import Image, ImageOps
import torch.nn.functional as F
import pickle
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
fpath = "/data/laiyan/airs/SurroundDepth/datasets/ddad/{}.txt"
train_filenames = readlines(fpath.format("val"))
cams = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
data_root = '/data/laiyan/ssd/ddad/raw_data/'
rgb_path = '/data/laiyan/ssd/ddad/raw_data/'
cur_path = os.path.dirname(os.path.realpath(__file__))
mask_path = os.path.join(cur_path, 'dataset/ddad_mask')
depth_path = '/data/laiyan/ssd/ddad/depth/'
with open("/data/laiyan/airs/VFDepth/dataset/ddad_mask/mask_idx_dict.pkl", 'rb') as f:
    mask_pkl = pickle.load(f)


def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_rgb_pil(line,scene_id,cam):
    full_color_path = os.path.join(rgb_path, scene_id, 'rgb',
                                   cam, line + '.jpg')
    color = pil_loader(full_color_path)
    return color

def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))
    with open(fname, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def show_one_cam(sample,src_cam,scene_id):

    src_color = get_rgb_pil(sample,scene_id,src_cam)
    plt.imshow(src_color)
    plt.show()

    mask_idx = mask_pkl[int(scene_id)]
    mask = mask_loader_scene(mask_path, mask_idx, src_cam)
    plt.imshow(mask)
    plt.show()

    # src_color_eq = ImageOps.equalize(src_color, mask = None)
    aaa=time.time()
    src_color_eq = ImageOps.equalize(src_color, mask = None)
    # src_color_eq = cv2.equalizeHist(src_color)
    print(time.time()-aaa)
    plt.imshow(src_color_eq)
    plt.show()
    return




if __name__ == '__main__':
    sample = "1568863909528155"
    scene_id = '000192'

    show_one_cam(sample,cams[3],scene_id)
