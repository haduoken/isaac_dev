import omni.replicator.core as rep
from PIL import Image, ImageDraw
import omni.syntheticdata as sd
import numpy as np


def world_to_image_pinhole(world_points, camera_params):
    # Project corners to image space (assumes pinhole camera model)
    proj_mat = camera_params["cameraProjection"].reshape(4, 4)
    view_mat = camera_params["cameraViewTransform"].reshape(4, 4)
    view_proj_mat = np.dot(view_mat, proj_mat)

    world_points_homo = np.pad(world_points, ((0, 0), (0, 1)), constant_values=1.0)
    tf_points = np.dot(world_points_homo, view_proj_mat)
    tf_points = tf_points / (tf_points[..., -1:])
    return 0.5 * (tf_points[..., :2] + 1)


def draw_points(img, image_points, radius=5):
    width, height = img.size
    draw = ImageDraw.Draw(img)
    image_points[:, 1] = height - image_points[:, 1]
    for point in image_points:
        xy = (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius)
        draw.ellipse(xy, fill=(255, 0, 0), outline=(0, 0, 0))


class CustomWriter(rep.Writer):
    def __init__(self, output_dir):
        self.frame_id = 0
        self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators = ["bounding_box_3d", "rgb", "camera_params"]

    def write(self, data):
        img = Image.fromarray(data["rgb"])
        render_product = [k for k in data.keys() if k.startswith("rp_")][0]  # assumes a single viewport attached
        width, height = data[render_product]["resolution"]
        bbox3ds = data["bounding_box_3d"]["data"]

        # Get 3D BBOX corners
        corners_3d = sd.helpers.get_bbox_3d_corners(bbox3ds)
        corners_3d = corners_3d.reshape(-1, 3)

        # Project to image space
        corners_2d = world_to_image_pinhole(corners_3d, data["camera_params"])
        corners_2d *= np.array([[width, height]])

        # Draw corners on image
        draw_points(img, corners_2d)
        self.backend.write_image(f"{self.frame_id}.png", img)
        self.frame_id += 1


rep.WriterRegistry.register(CustomWriter)

camera = rep.create.camera(position=(0, 0, 1000))
rp = rep.create.render_product(camera, (1024, 512))
writer = rep.WriterRegistry.get("CustomWriter")
writer.initialize(output_dir="_out")
writer.attach(rp)

# TEST
with rep.trigger.on_frame(num_frames=10):
    rep.create.cube(semantics=[("class", "cube")])
    with rep.create.cube(count=1, position=rep.distribution.uniform((-200, -200, -200), (200, 200, 200)),
                         semantics=[("class", "cube")]):
        rep.randomizer.rotation()
