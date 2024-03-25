import numpy as np
import json
from PIL import Image, ImageDraw

data_folder = '_out_sdrec6'

img = Image.open(f'{data_folder}/rgb_0000.png')
draw = ImageDraw.Draw(img)

width, height = img.size

with open(f'{data_folder}/camera_params_0000.json') as f:
    camera_params = json.load(f)
    print(camera_params)

proj_mat = np.array(camera_params["cameraProjection"]).reshape(4, 4)
view_mat = np.array(camera_params["cameraViewTransform"]).reshape(4, 4)

# print(view_proj_mat)

bbox3d_arr_raw = np.load(f'{data_folder}/bounding_box_3d_0000.npy')


def draw_line(pt0, pt1):
    draw.line((pt0[0], pt0[1], pt1[0], pt1[1]), fill="black", width=2)


valid_cnt = 0
for i, bbox3d_raw in enumerate(bbox3d_arr_raw):
    print(bbox3d_raw)

    x_min = bbox3d_raw["x_min"]
    y_min = bbox3d_raw["y_min"]
    z_min = bbox3d_raw["z_min"]
    x_max = bbox3d_raw["x_max"]
    y_max = bbox3d_raw["y_max"]
    z_max = bbox3d_raw["z_max"]

    trans = bbox3d_raw['transform']
    print(f'trans is f{trans}')

    pts = []
    pts.append([x_min, y_min, z_min])
    pts.append([x_min, y_min, z_max])
    pts.append([x_min, y_max, z_max])
    pts.append([x_min, y_max, z_min])

    pts.append([x_max, y_min, z_min])
    pts.append([x_max, y_min, z_max])
    pts.append([x_max, y_max, z_max])
    pts.append([x_max, y_max, z_min])

    corners = np.array(pts)

    # project
    points_homo = np.pad(corners, ((0, 0), (0, 1)), constant_values=1.0)

    temp = points_homo @ trans @ view_mat

    tf_points = points_homo @ trans @ view_mat @ proj_mat

    tf_points = tf_points / (tf_points[..., -1:])
    corners_2d = 0.5 * (tf_points[..., :2] + 1)
    corners_2d *= np.array([[width, height]])

    # draw
    radius = 5
    corners_2d[:, 1] = height - corners_2d[:, 1]

    bbox_invalid = False
    for pt in corners_2d:
        if pt[0] < 0 or pt[0] > width or pt[1] < 0 or pt[1] > height:
            bbox_invalid = True
    if bbox_invalid:
        continue

    for point in corners_2d:
        xy = (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)
        draw.ellipse(xy, fill=(255, 0, 0), outline=(0, 0, 0))

    draw_line(corners_2d[0], corners_2d[3])
    draw_line(corners_2d[3], corners_2d[7])
    draw_line(corners_2d[7], corners_2d[4])
    draw_line(corners_2d[4], corners_2d[0])

    draw_line(corners_2d[0], corners_2d[1])
    draw_line(corners_2d[3], corners_2d[2])
    draw_line(corners_2d[7], corners_2d[6])
    draw_line(corners_2d[4], corners_2d[5])

    draw_line(corners_2d[1], corners_2d[2])
    draw_line(corners_2d[2], corners_2d[6])
    draw_line(corners_2d[6], corners_2d[5])
    draw_line(corners_2d[5], corners_2d[1])

    valid_cnt += 1
    # if valid_cnt > 6:
    #     break

print(valid_cnt)
img.show()
