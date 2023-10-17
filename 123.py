from dgp.datasets import SynchronizedSceneDataset

# Load synchronized pairs of camera and lidar frames.
dataset = SynchronizedSceneDataset("/data/laiyan/datasets/DDAD/ddad_train_val/ddad.json",
    datum_names=('lidar', 'CAMERA_01', 'CAMERA_05'),
    generate_depth_from_datum='lidar',
    split='val'
    )

# Iterate through the dataset.
for sample in dataset:
    print(1)
  # Each sample contains a list of the requested datums.
  #   lidar, camera_01, camera_05 = sample[0:3]
  #   point_cloud = lidar['point_cloud'] # Nx3 numpy.ndarray
  #   image_01 = camera_01['rgb']  # PIL.Image
  #   depth_01 = camera_01['depth'] # (H,W) numpy.ndarray, generated from 'lidar'