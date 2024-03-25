import cv2
import numpy as np
import json
import math
from PIL import Image, ImageDraw
import open3d as o3d
from ast import literal_eval
from matplotlib import pyplot as plt


class Frame:
    cam_name_arr = ['cam_l', 'cam_r']
    T_Cam_To_Base = {
        'cam_l': np.array([[0, 0, -1, 0.25], [-1, 0, 0, 0.1], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32),
        'cam_r': np.array([[0, 0, -1, 0.25], [-1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    }
    floor_class = ['BACKGROUND', 'floor', 'floor_decal']

    grid_size = 200
    grid_resolution = 0.05

    def __init__(self, folder='/home/kilox/workspace/isaac_dev/_out_sdrec3', frame_id='0000'):
        data = {'frame_id': frame_id}

        for cam_name in self.cam_name_arr:
            cam_info = {'came_name': cam_name}

            data_folder = f'{folder}/RenderProduct_{cam_name}'
            cam_info['img_path'] = f'{data_folder}/rgb/rgb_{frame_id}.png'
            cam_info['img'] = np.array(Image.open(f'{data_folder}/rgb/rgb_{frame_id}.png'))
            cam_info['wh'] = cam_info['img'].shape[:2]
            width, height = cam_info['wh']

            # draw = ImageDraw.Draw(img)

            with open(f'{data_folder}/camera_params/camera_params_{frame_id}.json') as f:
                camera_params = json.load(f)
                cam_info['img_param'] = camera_params

                cam_info['proj_mat'] = np.array(camera_params["cameraProjection"]).reshape(4, 4)
                cam_info['view_mat'] = np.array(camera_params["cameraViewTransform"]).reshape(4, 4)

            # 加载深度
            cam_info['depth'] = np.load(f'{data_folder}/distance_to_camera/distance_to_camera_{frame_id}.npy')
            depth = cam_info['depth']

            # 加载semantic_segmentation
            seg_img = np.array(
                Image.open(f'{data_folder}/semantic_segmentation/semantic_segmentation_{frame_id}.png'))

            with open(f'{data_folder}/semantic_segmentation/semantic_segmentation_labels_{frame_id}.json') as f:
                label = json.load(f)

            ## 将segmantation分成可行驶与不可行驶
            seg_filted = np.ones([width, height])
            for k, v in label.items():
                if v['class'] in self.floor_class:
                    k = literal_eval(k)
                    mask = seg_img[:, :] == k
                    mask = mask[:, :, 0]
                    seg_filted[mask] = 0
            cam_info['seg'] = seg_filted

            # parse points
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))

            pix = np.concatenate([xx[:, :, None], yy[:, :, None]], axis=2, dtype=np.float32)

            pix[:, :, 1] = (height - pix[:, :, 1]) / height
            pix[:, :, 0] = pix[:, :, 0] / width
            homo_pix = np.pad(pix, ((0, 0), (0, 0), (0, 2)), constant_values=1.0)
            depth_tmp = np.repeat(depth[:, :, None], 4, axis=2)
            homo_pix = homo_pix * -depth_tmp

            homo_pix[:, :, 2] = 0.01

            homo_pix = homo_pix @ np.linalg.inv(cam_info['proj_mat'])

            # homo_pix = np.reshape(homo_pix, [-1, 4])

            homo_pix[:, :, -1] = 1

            # 转换到车体坐标系
            homo_pix = homo_pix @ self.T_Cam_To_Base[cam_name].transpose()

            # 添加语义
            homo_pix[:, :, 3] = seg_filted

            # reshape 成点的形式
            homo_pix = np.reshape(homo_pix, [-1, 4])

            cam_info['points'] = homo_pix

            data[cam_name] = cam_info

        self.data = data

    def viz_points(self):
        total_pts = np.concatenate([self.data[cam_name]['points'] for cam_name in self.cam_name_arr], axis=0,
                                   dtype=np.float64)

        pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
        pcd_o3d.points = o3d.utility.Vector3dVector(total_pts[:, :3])  # set pcd_np as the point cloud points

        # Visualize:
        o3d.visualization.draw_geometries([pcd_o3d])

    def base_seg(self):
        total_pts = np.concatenate([self.data[cam_name]['points'] for cam_name in self.cam_name_arr], axis=0,
                                   dtype=np.float64)

        # 转换到BEV
        # 0:unknow, 1:ground 2:obstacle

        grid = np.ones((self.grid_size, self.grid_size), dtype=np.float32) * 0.5
        half_size = self.grid_size * self.grid_resolution / 2

        for pt in total_pts:
            cx, cy = round((half_size - pt[1]) / self.grid_resolution), round(
                (half_size - pt[0]) / self.grid_resolution)
            if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size:
                continue
            # free point
            if pt[3] == 0:
                if grid[cy, cx] == 0:
                    continue

                grid[cy, cx] = 1

            # obs point
            if pt[3] == 1:
                grid[cy, cx] = 0

        self.data['bev_map'] = grid
