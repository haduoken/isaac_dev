__copyright__ = "Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import io
import json
from typing import List

import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry, orchestrator
from omni.replicator.core.bindings._omni_replicator_core import Schema_omni_replicator_extinfo_1_0
from omni.replicator.core.scripts.utils import skeleton_data_utils
from omni.syntheticdata.scripts.SyntheticData import SyntheticData

# from .tools import colorize_normals

__version__ = "0.0.1"


class CustomWriterSep(Writer):
    """Basic writer capable of writing built-in annotator groundtruth.

    Attributes:
        output_dir:
            Output directory string that indicates the directory to save the results.
        s3_bucket:
            The S3 Bucket name to write to. If not provided, disk backend will be used instead. Default: None.
            This backend requires that AWS credentials are set up in ~/.aws/credentials.
            See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration
        s3_region:
            If provided, this is the region the S3 bucket will be set to. Default: us-east-1
        s3_endpoint:
            If provided, this endpoint URL will be used instead of the default.
        semantic_types:
            List of semantic types to consider when filtering annotator data. Default: ["class"]
        rgb:
            Boolean value that indicates whether the rgb annotator will be activated
            and the data will be written or not. Default: False.
        bounding_box_2d_tight:
            Boolean value that indicates whether the bounding_box_2d_tight annotator will be activated
            and the data will be written or not. Default: False.
        bounding_box_2d_loose:
            Boolean value that indicates whether the bounding_box_2d_loose annotator will be activated
            and the data will be written or not. Default: False.
        semantic_segmentation:
            Boolean value that indicates whether the semantic_segmentation annotator will be activated
            and the data will be written or not. Default: False.
        instance_id_segmentation:
            Boolean value that indicates whether the instance_id_segmentation annotator will be activated
            and the data will be written or not. Default: False.
        instance_segmentation:
            Boolean value that indicates whether the instance_segmentation annotator will be activated
            and the data will be written or not. Default: False.
        distance_to_camera:
            Boolean value that indicates whether the distance_to_camera annotator will be activated
            and the data will be written or not. Default: False.
        distance_to_image_plane:
            Boolean value that indicates whether the distance_to_image_plane annotator will be activated
            and the data will be written or not. Default: False.
        bounding_box_3d:
            Boolean value that indicates whether the bounding_box_3d annotator will be activated
            and the data will be written or not. Default: False.
        occlusion:
            Boolean value that indicates whether the occlusion annotator will be activated
            and the data will be written or not. Default: False.
        normals:
            Boolean value that indicates whether the normals annotator will be activated
            and the data will be written or not. Default: False.
        motion_vectors:
            Boolean value that indicates whether the motion_vectors annotator will be activated
            and the data will be written or not. Default: False.
        camera_params:
            Boolean value that indicates whether the camera_params annotator will be activated
            and the data will be written or not. Default: False.
        pointcloud:
            Boolean value that indicates whether the pointcloud annotator will be activated
            and the data will be written or not. Default: False.
        pointcloud_include_unlabelled:
            If ``True``, pointcloud annotator will capture any prim in the camera's perspective, not matter if it has
            semantics or not. If ``False``, only prims with semantics will be captured.
            Defaults to ``False``.
        image_output_format:
            String that indicates the format of saved RGB images. Default: "png"
        colorize_semantic_segmentation:
            If ``True``, semantic segmentation is converted to an image where semantic IDs are mapped to colors
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            Defaults to ``True``.
        colorize_instance_id_segmentation:
            If True, instance id segmentation is converted to an image where instance IDs are mapped to colors.
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            Defaults to ``True``.
        colorize_instance_segmentation:
            If True, instance segmentation is converted to an image where instance are mapped to colors.
            and saved as a uint8 4 channel PNG image. If ``False``, the output is saved as a uint32 PNG image.
            Defaults to ``True``.
        frame_padding:
            Pad the frame number with leading zeroes.  Default: 4
        semantic_filter_predicate:
            A string specifying a semantic filter predicate as a disjunctive normal form of semantic type, labels.

            Examples :
                "typeA : labelA & !labelB | labelC , typeB: labelA ; typeC: labelD"
                "typeA : * ; * : labelA"

    Example:
        >>> import omni.replicator.core as rep
        >>> camera = rep.create.camera()
        >>> render_product = rep.create.render_product(camera, (1024, 1024))
        >>> writer = rep.WriterRegistry.get("BasicWriter")
        >>> import carb
        >>> tmp_dir = carb.tokens.get_tokens_interface().resolve("${temp}/rgb")
        >>> writer.initialize(output_dir=tmp_dir, rgb=True)
        >>> writer.attach([render_product])
        >>> rep.orchestrator.run()
    """

    def __init__(
            self,
            output_dir: str,
            s3_bucket: str = None,
            s3_region: str = None,
            s3_endpoint: str = None,
            semantic_types: List[str] = None,
            rgb: bool = False,
            bounding_box_2d_tight: bool = False,
            bounding_box_2d_loose: bool = False,
            semantic_segmentation: bool = False,
            instance_id_segmentation: bool = False,
            instance_segmentation: bool = False,
            # background_rand: bool = False,
            distance_to_camera: bool = False,
            distance_to_image_plane: bool = False,
            bounding_box_3d: bool = False,
            occlusion: bool = False,
            normals: bool = False,
            motion_vectors: bool = False,
            camera_params: bool = False,
            pointcloud: bool = False,
            pointcloud_include_unlabelled: bool = False,
            image_output_format: str = "png",
            colorize_semantic_segmentation: bool = True,
            colorize_instance_id_segmentation: bool = True,
            colorize_instance_segmentation: bool = True,
            skeleton_data: bool = False,
            frame_padding: int = 4,
            semantic_filter_predicate: str = None,
    ):
        self._output_dir = output_dir
        if s3_bucket:
            self.backend = BackendDispatch(
                {
                    "use_s3": True,
                    "paths": {
                        "out_dir": output_dir,
                        "s3_bucket": s3_bucket,
                        "s3_region": s3_region,
                        "s3_endpoint_url": s3_endpoint,
                    },
                }
            )
        else:
            self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._backend = self.backend  # Kept for backwards compatibility
        self._output_dir = self.backend.output_dir
        self._frame_id = 0
        self._sequence_id = 0
        self._image_output_format = image_output_format
        self._output_data_format = {}
        self.annotators = []
        self.version = __version__
        self._frame_padding = frame_padding
        self._telemetry = Schema_omni_replicator_extinfo_1_0()

        self.colorize_semantic_segmentation = colorize_semantic_segmentation
        self.colorize_instance_id_segmentation = colorize_instance_id_segmentation
        self.colorize_instance_segmentation = colorize_instance_segmentation

        # Specify the semantic types that will be included in output
        if semantic_types is not None:
            if semantic_filter_predicate is None:
                semantic_filter_predicate = ":*; ".join(semantic_types) + ":*"
            else:
                raise ValueError(
                    "`semantic_types` and `semantic_filter_predicate` are mutually exclusive. Please choose only one."
                )
        elif semantic_filter_predicate is None:
            semantic_filter_predicate = "class:*"

        # Set the global semantic filter predicate
        if semantic_filter_predicate is not None:
            SyntheticData.Get().set_instance_mapping_semantic_filter(semantic_filter_predicate)

        # RGB
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Bounding Box 2D
        if bounding_box_2d_tight:
            self.annotators.append("bounding_box_2d_tight_fast")

        if bounding_box_2d_loose:
            self.annotators.append("bounding_box_2d_loose_fast")

        # Semantic Segmentation
        if semantic_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": colorize_semantic_segmentation}
                )
            )

        # Instance Segmentation
        if instance_id_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_id_segmentation_fast", init_params={"colorize": colorize_instance_id_segmentation}
                )
            )

        # Instance Segmentation
        if instance_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_segmentation_fast", init_params={"colorize": colorize_instance_segmentation}
                )
            )

        # # Background Rand
        # if background_rand:
        #     self.annotators.append(AnnotatorRegistry.get_annotator("background_rand", init_params={"colorize": True}))

        # Depth
        if distance_to_camera:
            self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_camera"))

        if distance_to_image_plane:
            self.annotators.append(AnnotatorRegistry.get_annotator("distance_to_image_plane"))

        # Bounding Box 3D
        if bounding_box_3d:
            self.annotators.append("bounding_box_3d_fast")

        # Motion Vectors
        if motion_vectors:
            self.annotators.append(AnnotatorRegistry.get_annotator("motion_vectors"))

        # Occlusion
        if occlusion:
            self.annotators.append(AnnotatorRegistry.get_annotator("occlusion"))

        # Normals
        if normals:
            self.annotators.append(AnnotatorRegistry.get_annotator("normals"))

        # Camera Params
        if camera_params:
            self.annotators.append(AnnotatorRegistry.get_annotator("camera_params"))

        # Pointcloud
        if pointcloud:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "pointcloud", init_params={"includeUnlabelled": pointcloud_include_unlabelled}
                )
            )

        # Skeleton Data
        if skeleton_data:
            self.annotators.append(
                AnnotatorRegistry.get_annotator("skeleton_data", init_params={"useSkelJoints": False})
            )

        backend_type = "S3" if s3_bucket else "Disk"
        self._telemetry.basicwriter_sendEvent(
            self.version,
            backend_type,
            rgb,
            bounding_box_2d_tight,
            bounding_box_2d_loose,
            semantic_segmentation,
            instance_id_segmentation,
            instance_segmentation,
            distance_to_camera,
            distance_to_image_plane,
            bounding_box_3d,
            occlusion,
            normals,
            motion_vectors,
            camera_params,
            pointcloud,
            pointcloud_include_unlabelled,
            image_output_format,
            colorize_semantic_segmentation,
            colorize_instance_id_segmentation,
            colorize_instance_segmentation,
            skeleton_data,
        )

        self.frame_cnt = 0

    def write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        # Check for on_time triggers
        # For each on_time trigger, prefix the output frame number with the trigger counts

        # reduce the FPS to 1/5
        self.frame_cnt += 1
        if self.frame_cnt % 5 != 0:
            return

        sequence_id = ""
        for trigger_name, call_count in data["trigger_outputs"].items():
            if "on_time" in trigger_name:
                sequence_id = f"{call_count}_{sequence_id}"
        if sequence_id != self._sequence_id:
            self._frame_id = 0
            self._sequence_id = sequence_id

        for annotator in data.keys():
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            # multiple render_products
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                self._write_rgb(data, render_product_path, annotator)

            if annotator.startswith("Aug"):
                if isinstance(data[annotator], dict):
                    data[annotator] = data[annotator]["data"]
                if multi_render_prod:
                    render_product_path += f"{annotator}/"
                self._write_rgb(data, render_product_path, annotator)

            if annotator.startswith("normals"):
                if multi_render_prod:
                    render_product_path += "normals/"
                self._write_normals(data, render_product_path, annotator)

            if annotator.startswith("distance_to_camera"):
                if multi_render_prod:
                    render_product_path += "distance_to_camera/"
                self._write_distance_to_camera(data, render_product_path, annotator)

            if annotator.startswith("distance_to_image_plane"):
                if multi_render_prod:
                    render_product_path += "distance_to_image_plane/"
                self._write_distance_to_image_plane(data, render_product_path, annotator)

            if annotator.startswith("semantic_segmentation"):
                if multi_render_prod:
                    render_product_path += "semantic_segmentation/"
                self._write_semantic_segmentation(data, render_product_path, annotator)

            if annotator.startswith("instance_id_segmentation"):
                if multi_render_prod:
                    render_product_path += "instance_id_segmentation/"
                self._write_instance_id_segmentation(data, render_product_path, annotator)

            if annotator.startswith("instance_segmentation"):
                if multi_render_prod:
                    render_product_path += "instance_segmentation/"
                self._write_instance_segmentation(data, render_product_path, annotator)

            if annotator.startswith("motion_vectors"):
                if multi_render_prod:
                    render_product_path += "motion_vectors/"
                self._write_motion_vectors(data, render_product_path, annotator)

            if annotator.startswith("occlusion"):
                if multi_render_prod:
                    render_product_path += "occlusion/"
                self._write_occlusion(data, render_product_path, annotator)

            if annotator.startswith("bounding_box_3d"):
                if multi_render_prod:
                    render_product_path += "bounding_box_3d/"
                self._write_bounding_box_data(data, "3d", render_product_path, annotator)

            if annotator.startswith("bounding_box_2d_loose"):
                if multi_render_prod:
                    render_product_path += "bounding_box_2d_loose/"
                self._write_bounding_box_data(data, "2d_loose", render_product_path, annotator)

            if annotator.startswith("bounding_box_2d_tight"):
                if multi_render_prod:
                    render_product_path += "bounding_box_2d_tight/"
                self._write_bounding_box_data(data, "2d_tight", render_product_path, annotator)

            if annotator.startswith("camera_params"):
                if multi_render_prod:
                    render_product_path += "camera_params/"
                self._write_camera_params(data, render_product_path, annotator)

            if annotator.startswith("pointcloud"):
                if multi_render_prod:
                    render_product_path += "pointcloud/"
                self._write_pointcloud(data, render_product_path, annotator)

            if annotator.startswith("skeleton_data"):
                if multi_render_prod:
                    render_product_path += "skeleton_data/"
                self._write_skeleton(data, render_product_path, annotator)

        self._frame_id += 1

    def _write_rgb(self, data: dict, render_product_path: str, annotator: str):
        file_path = f"{render_product_path}rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        self._backend.write_image(file_path, data[annotator])

    def _write_normals(self, data: dict, render_product_path: str, annotator: str):
        normals_data = data[annotator]
        file_path = f"{render_product_path}normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        # colorized_normals_data = colorize_normals(normals_data)
        # self._backend.write_image(file_path, colorized_normals_data)

    def _write_distance_to_camera(self, data: dict, render_product_path: str, annotator: str):
        dist_to_cam_data = data[annotator]
        file_path = (
            f"{render_product_path}distance_to_camera_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.write_array(file_path, dist_to_cam_data)

    def _write_distance_to_image_plane(self, data: dict, render_product_path: str, annotator: str):
        dis_to_img_plane_data = data[annotator]
        file_path = f"{render_product_path}distance_to_image_plane_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.write_array(file_path, dis_to_img_plane_data)

    def _write_semantic_segmentation(self, data: dict, render_product_path: str, annotator: str):
        semantic_seg_data = data[annotator]["data"]
        height, width = semantic_seg_data.shape[:2]

        file_path = (
            f"{render_product_path}semantic_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        )
        if self.colorize_semantic_segmentation:
            semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, semantic_seg_data)
        else:
            semantic_seg_data = semantic_seg_data.view(np.uint32).reshape(height, width)
            self._backend.write_image(file_path, semantic_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}semantic_segmentation_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_instance_id_segmentation(self, data: dict, render_product_path: str, annotator: str):
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = f"{render_product_path}instance_id_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_instance_id_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, instance_seg_data)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.write_image(file_path, instance_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}instance_id_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_instance_segmentation(self, data: dict, render_product_path: str, annotator: str):
        instance_seg_data = data[annotator]["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = (
            f"{render_product_path}instance_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        )
        if self.colorize_instance_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.write_image(file_path, instance_seg_data)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.write_image(file_path, instance_seg_data)

        id_to_labels = data[annotator]["info"]["idToLabels"]
        file_path = f"{render_product_path}instance_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_labels.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

        id_to_semantics = data[annotator]["info"]["idToSemantics"]
        file_path = f"{render_product_path}instance_segmentation_semantics_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps({str(k): v for k, v in id_to_semantics.items()}).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_motion_vectors(self, data: dict, render_product_path: str, annotator: str):
        motion_vec_data = data[annotator]
        file_path = (
            f"{render_product_path}motion_vectors_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.write_array(file_path, motion_vec_data)

    def _write_occlusion(self, data: dict, render_product_path: str, annotator: str):
        occlusion_data = data[annotator]

        file_path = f"{render_product_path}occlusion_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.write_array(file_path, occlusion_data)

    def _write_bounding_box_data(self, data: dict, bbox_type: str, render_product_path: str, annotator: str):
        bbox_data = data[annotator]["data"]
        id_to_labels = data[annotator]["info"]["idToLabels"]
        prim_paths = data[annotator]["info"]["primPaths"]

        file_path = f"{render_product_path}bounding_box_{bbox_type}_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.write_array(file_path, bbox_data)

        labels_file_path = f"{render_product_path}bounding_box_{bbox_type}_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(id_to_labels).encode())
        self._backend.write_blob(labels_file_path, buf.getvalue())

        labels_file_path = f"{render_product_path}bounding_box_{bbox_type}_prim_paths_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        buf = io.BytesIO()
        buf.write(json.dumps(prim_paths).encode())
        self._backend.write_blob(labels_file_path, buf.getvalue())

    def _write_camera_params(self, data: dict, render_product_path: str, annotator: str):
        camera_data = data[annotator]
        serializable_data = {}

        for key, val in camera_data.items():
            if isinstance(val, np.ndarray):
                serializable_data[key] = val.tolist()
            else:
                serializable_data[key] = val

        file_path = (
            f"{render_product_path}camera_params_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        )
        buf = io.BytesIO()
        buf.write(json.dumps(serializable_data).encode())
        self._backend.write_blob(file_path, buf.getvalue())

    def _write_pointcloud(self, data: dict, render_product_path: str, annotator: str):
        pointcloud_data = data[annotator]["data"]
        pointcloud_rgb = data[annotator]["info"]["pointRgb"].reshape(-1, 4)
        pointcloud_normals = data[annotator]["info"]["pointNormals"].reshape(-1, 4)
        pointcloud_semantic = data[annotator]["info"]["pointSemantic"]
        pointcloud_instance = data[annotator]["info"]["pointInstance"]

        file_path = f"{render_product_path}pointcloud_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.write_array(file_path, pointcloud_data)

        rgb_file_path = (
            f"{render_product_path}pointcloud_rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.write_array(rgb_file_path, pointcloud_rgb)

        normals_file_path = (
            f"{render_product_path}pointcloud_normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.write_array(normals_file_path, pointcloud_normals)

        semancit_file_path = (
            f"{render_product_path}pointcloud_semantic_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.write_array(semancit_file_path, pointcloud_semantic)

        instance_file_path = (
            f"{render_product_path}pointcloud_instance_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.write_array(instance_file_path, pointcloud_instance)

    def _write_skeleton(self, data: dict, render_product_path: str, annotator: str):
        # "skeletonData" is deprecated
        # skeleton = json.loads(data[annotator]["skeletonData"])

        skeleton_dict = {}

        skel_name = data[annotator]["skelName"]
        skel_path = data[annotator]["skelPath"]
        asset_path = data[annotator]["assetPath"]
        animation_variant = data[annotator]["animationVariant"]
        skeleton_parents = skeleton_data_utils.get_skeleton_parents(
            data[annotator]["numSkeletons"], data[annotator]["skeletonParents"], data[annotator]["skeletonParentsSizes"]
        )
        rest_global_translations = skeleton_data_utils.get_rest_global_translations(
            data[annotator]["numSkeletons"],
            data[annotator]["restGlobalTranslations"],
            data[annotator]["restGlobalTranslationsSizes"],
        )
        rest_local_translations = skeleton_data_utils.get_rest_local_translations(
            data[annotator]["numSkeletons"],
            data[annotator]["restLocalTranslations"],
            data[annotator]["restLocalTranslationsSizes"],
        )
        rest_local_rotations = skeleton_data_utils.get_rest_local_rotations(
            data[annotator]["numSkeletons"],
            data[annotator]["restLocalRotations"],
            data[annotator]["restLocalRotationsSizes"],
        )
        global_translations = skeleton_data_utils.get_global_translations(
            data[annotator]["numSkeletons"],
            data[annotator]["globalTranslations"],
            data[annotator]["globalTranslationsSizes"],
        )
        local_rotations = skeleton_data_utils.get_local_rotations(
            data[annotator]["numSkeletons"], data[annotator]["localRotations"], data[annotator]["localRotationsSizes"]
        )
        translations_2d = skeleton_data_utils.get_translations_2d(
            data[annotator]["numSkeletons"], data[annotator]["translations2d"], data[annotator]["translations2dSizes"]
        )
        skeleton_joints = skeleton_data_utils.get_skeleton_joints(data[annotator]["skeletonJoints"])
        joint_occlusions = skeleton_data_utils.get_joint_occlusions(
            data[annotator]["numSkeletons"], data[annotator]["jointOcclusions"], data[annotator]["jointOcclusionsSizes"]
        )
        occlusion_types = skeleton_data_utils.get_occlusion_types(
            data[annotator]["numSkeletons"], data[annotator]["occlusionTypes"], data[annotator]["occlusionTypesSizes"]
        )
        in_view = data[annotator]["inView"]

        for skel_num in range(data[annotator]["numSkeletons"]):
            skeleton_dict[f"skeleton_{skel_num}"] = {}
            skeleton_dict[f"skeleton_{skel_num}"]["skel_name"] = skel_name[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["skel_path"] = skel_path[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["asset_path"] = asset_path[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["animation_variant"] = animation_variant[skel_num]
            skeleton_dict[f"skeleton_{skel_num}"]["skeleton_parents"] = (
                skeleton_parents[skel_num].tolist() if skeleton_parents else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_global_translations"] = (
                rest_global_translations[skel_num].tolist() if rest_global_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_local_translations"] = (
                rest_local_translations[skel_num].tolist() if rest_local_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["rest_local_rotations"] = (
                rest_local_rotations[skel_num].tolist() if rest_local_rotations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["global_translations"] = (
                global_translations[skel_num].tolist() if global_translations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["local_rotations"] = (
                local_rotations[skel_num].tolist() if local_rotations else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["translations_2d"] = (
                translations_2d[skel_num].tolist() if translations_2d else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["skeleton_joints"] = (
                skeleton_joints[skel_num] if skeleton_joints else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["joint_occlusions"] = (
                joint_occlusions[skel_num].tolist() if joint_occlusions else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["occlusion_types"] = (
                occlusion_types[skel_num] if occlusion_types else []
            )
            skeleton_dict[f"skeleton_{skel_num}"]["in_view"] = bool(in_view[skel_num]) if in_view.any() else False

        file_path = f"{render_product_path}skeleton_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"

        buf = io.BytesIO()
        buf.write(json.dumps(skeleton_dict).encode())
        self.backend.write_blob(file_path, buf.getvalue())


WriterRegistry.register(CustomWriterSep)
