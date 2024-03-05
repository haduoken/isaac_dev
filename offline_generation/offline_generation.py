# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate offline synthetic dataset
"""

import argparse
import json
import math
import os
import random

import carb
import yaml
from omni.isaac.kit import SimulationApp

# Default config (will be updated/extended by any other passed config arguments)
config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "resolution": [1024, 1024],
    "rt_subframes": 2,
    "num_frames": 20,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "scope_name": "/MyScope",
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "_out_offline_generation",
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "Forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "TrafficCone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "Pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "Cardbox",
    },
}

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    print("File exist")
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    print(f"File {args.config} does not exist, will use default config")
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# If there are specific writer parameters in the input config file make sure they are not mixed with the default ones
if "writer_config" in args_config:
    config["writer_config"].clear()

# Update the default config dictionay with any new parameters or values from the config file
config.update(args_config)

# Create the simulation app with the given launch_config
simulation_app = SimulationApp(launch_config=config["launch_config"])

import offline_generation_utils

# Late import of runtime modules (the SimulationApp needs to be created before loading the modules)
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.utils import prims
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Gf

# Get server path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

# Open the given environment in a new stage
print(f"Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage{config['env_url']}, closing application..")
    simulation_app.close()

# Disable capture on play and async rendering
carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
carb.settings.get_settings().set("/app/asyncRendering", False)

if config["clear_previous_semantics"]:
    stage = get_current_stage()
    offline_generation_utils.remove_previous_semantics(stage)

# Spawn a new forklift at a random pose
forklift_prim = prims.create_prim(
    prim_path="/World/Forklift",
    position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
    usd_path=assets_root_path + config["forklift"]["url"],
    semantic_label=config["forklift"]["class"],
)

# Spawn the pallet in front of the forklift with a random offset on the Y (pallet's forward) axis
forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, random.uniform(-1.2, -1.8), 0))
pallet_pos_gf = (pallet_offset_tf * forklift_tf).ExtractTranslation()
forklift_quat_gf = forklift_tf.ExtractRotationQuat()
forklift_quat_xyzw = (forklift_quat_gf.GetReal(), *forklift_quat_gf.GetImaginary())

pallet_prim = prims.create_prim(
    prim_path="/World/Pallet",
    position=pallet_pos_gf,
    orientation=forklift_quat_xyzw,
    usd_path=assets_root_path + config["pallet"]["url"],
    semantic_label=config["pallet"]["class"],
)

# Register randomizers graphs
offline_generation_utils.register_scatter_boxes(pallet_prim, assets_root_path, config)
offline_generation_utils.register_cone_placement(forklift_prim, assets_root_path, config)
offline_generation_utils.register_lights_placement(forklift_prim, pallet_prim)

# Spawn a camera in the driver's location looking at the pallet
foklift_pos_gf = forklift_tf.ExtractTranslation()
driver_cam_pos_gf = foklift_pos_gf + Gf.Vec3d(0.0, 0.0, 1.9)

driver_cam = rep.create.camera(
    focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name="DriverCam"
)

# Camera looking at the pallet
pallet_cam = rep.create.camera(name="PalletCam")

# Camera looking at the forklift from a top view with large min clipping to see the scene through the ceiling
top_view_cam = rep.create.camera(clipping_range=(6.0, 1000000.0), name="TopCam")

# Generate graph nodes to be triggered every frame
with rep.trigger.on_frame():
    rep.randomizer.scatter_boxes()
    rep.randomizer.place_cones()
    rep.randomizer.randomize_lights()

    pallet_cam_min = (pallet_pos_gf[0] - 2, pallet_pos_gf[1] - 2, 2)
    pallet_cam_max = (pallet_pos_gf[0] + 2, pallet_pos_gf[1] + 2, 4)
    with pallet_cam:
        rep.modify.pose(
            position=rep.distribution.uniform(pallet_cam_min, pallet_cam_max),
            look_at=str(pallet_prim.GetPrimPath()),
        )

    driver_cam_min = (driver_cam_pos_gf[0], driver_cam_pos_gf[1], driver_cam_pos_gf[2] - 0.25)
    driver_cam_max = (driver_cam_pos_gf[0], driver_cam_pos_gf[1], driver_cam_pos_gf[2] + 0.25)
    with driver_cam:
        rep.modify.pose(
            position=rep.distribution.uniform(driver_cam_min, driver_cam_max),
            look_at=str(pallet_prim.GetPrimPath()),
        )

# Generate graph nodes to be triggered only at the given interval
with rep.trigger.on_frame(interval=4):
    top_view_cam_min = (foklift_pos_gf[0], foklift_pos_gf[1], 9)
    top_view_cam_max = (foklift_pos_gf[0], foklift_pos_gf[1], 11)
    with top_view_cam:
        rep.modify.pose(
            position=rep.distribution.uniform(top_view_cam_min, top_view_cam_max),
            rotation=rep.distribution.uniform((0, -90, -30), (0, -90, 30)),
        )

# If output directory is relative, set it relative to the current working directory
if config["writer_config"]["output_dir"] and not os.path.isabs(config["writer_config"]["output_dir"]):
    config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), config["writer_config"]["output_dir"])
print(f"Output directory={config['writer_config']['output_dir']}")

# Setup the writer
writer = rep.WriterRegistry.get(config["writer"])
writer.initialize(**config["writer_config"])
forklift_rp = rep.create.render_product(top_view_cam, config["resolution"], name="TopView")
driver_rp = rep.create.render_product(driver_cam, config["resolution"], name="DriverView")
pallet_rp = rep.create.render_product(pallet_cam, config["resolution"], name="PalletView")
writer.attach([forklift_rp, driver_rp, pallet_rp])

# Run a simulation before generating data
offline_generation_utils.simulate_falling_objects(forklift_prim, assets_root_path, config)

# Increase subframes if materials are not loaded on time or ghosting rendering artifacts appear on quickly moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
if config["rt_subframes"] > 1:
    rep.settings.carb_settings("/omni/replicator/RTSubframes", config["rt_subframes"])
else:
    carb.log_warn("RTSubframes is set to 1, consider increasing it if materials are not loaded on time")

# Run the SDG
rep.orchestrator.run_until_complete(num_frames=config["num_frames"])

simulation_app.close()
