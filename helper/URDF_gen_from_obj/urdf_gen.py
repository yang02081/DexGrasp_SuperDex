"""
urdf_gen.py

Purpose:
This script processes 3D model data in the following steps:
1. Find `.obj` files in the source folder.
2. Copy each `.obj` file to the destination folder and rename it to `top.obj`.
3. Copy a file named `cube.obj` into the same destination folder and rename it to `bottom.obj`.
4. Generate two types of URDF (Unified Robot Description Format) files from the processed `.obj` files:
    - URDFs with a free base (`fixed_base=False`).
    - URDFs with a fixed base (`fixed_base=True`).
5. Move all generated URDF files into the project's `rsc/` resource directory.

Input:
- `source_folder`: Directory path containing the original `.obj` models.
- `cube.obj`: A file named `cube.obj` used as the bottom model for all generated URDFs.
  This file should exist in the script's current working directory.
- `export_meshes_shapenet.py`: Provides the `get_obj_data` function for extracting mesh data from the model path.
- `urdf_affordpose.py`: Provides the `create_urdf_obj_new` and `write_urdf_file` functions for generating and writing URDF files.

Output:
- `destination_parent_folder`: A temporary directory containing the processed `top.obj` and `bottom.obj` files organized by the original `.obj` structure.
- `output_directory`: A temporary directory containing the generated URDF files and related resources for each processed `.obj` file.

"""
import os
import shutil
import argparse
import sys
import csv

import numpy as np



# Resolve defaults from the repository rather than from an author's workstation.
repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
default_source_folder = os.path.join(repository_root, "rsc", "ycb_crossDex")

parser = argparse.ArgumentParser(description="Generate URDF resources from OBJ meshes.")
parser.add_argument("--source-folder", default=default_source_folder)
parser.add_argument("--output-folder", default=os.path.join(repository_root, "rsc", "ycb_crossDex_output"))
parser.add_argument("--temp-folder", default=os.path.join(repository_root, "rsc", "ycb_crossDex_temp"))
parser.add_argument("--cube-obj", default=os.path.join(os.path.dirname(__file__), "cube.obj"))
args = parser.parse_args()

from export_meshes_shapenet import get_obj_data
from urdf_affordpose import create_urdf_obj, write_urdf_file, create_urdf_obj_new, create_urdf_obj_free

source_folder = os.path.abspath(args.source_folder)
destination_parent_folder = os.path.abspath(args.output_folder)
temp_diretory = os.path.abspath(args.temp_folder)
cube_obj_path = os.path.abspath(args.cube_obj)

# Create the intermediate directory if it does not exist
if not os.path.exists(temp_diretory):
    os.makedirs(temp_diretory)
if not os.path.exists(destination_parent_folder):
    os.makedirs(destination_parent_folder)

dat_ids_list = []  # Initialize the data ID list

print("Step 1: Format OBJ files and generate top.obj / bottom.obj")
# Iterate over each .obj file in the source folder
for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.endswith('.obj'):
            obj_path = os.path.join(root, filename)
            
            obj_base_name = os.path.splitext(filename)[0]


            # Build the destination folder path: temp_diretory/obj_name
            destination_folder = os.path.join(temp_diretory, obj_base_name)

            # Create the destination folder if it does not exist
            # if not os.path.exists(destination_folder):
            os.makedirs(destination_folder,exist_ok=True)
            # else:
            #     print(f"pass  {obj_base_name}: exists")
            #     continue

            # Target file path, renamed to top.obj
            top_obj_path = os.path.join(destination_folder, 'top.obj')

            # Copy the .obj file to the destination folder and rename it to top.obj
            shutil.copyfile(obj_path, top_obj_path)

            # Copy cube.obj to the destination folder and rename it to bottom.obj
            bottom_obj_path = os.path.join(destination_folder, 'bottom.obj')
            shutil.copyfile(cube_obj_path, bottom_obj_path)
            print(f"obj:{filename} process finished")



no_bottom = True
decimated = False

print("\nStep 2: Generate URDF files")
# Iterate over all `top.obj` files in the input directory
# This section generates URDF files for each `top.obj` (free-base and fixed-base variants)
for root, dirs, files in os.walk(temp_diretory):
    for filename in files:
        if filename.endswith("top.obj"):
            # Extract the object name from the path
            # Example: <temporary-directory>/objA/top.obj
            obj_base_name = root.split(os.sep)[-1]


            # Create the output directory for the current object, e.g. rsc/obj_base_name/
            label_dir = os.path.join(destination_parent_folder, obj_base_name)
            os.makedirs(label_dir,exist_ok=True)

            # Load mesh data
            mesh_data = get_obj_data(model_path=root, out_path=label_dir, decimated=decimated)

            # Generate a URDF with a free base
            root1 = create_urdf_obj_new(obj_name=obj_base_name, mesh_data=mesh_data, pose_limits=[0, 0.001], fixed_base=False, no_bottom=no_bottom)
            write_urdf_file(root=root1, out_path=label_dir, file_name=obj_base_name)

            # Generate a URDF with a fixed base
            root2 = create_urdf_obj_new(obj_name=obj_base_name, mesh_data=mesh_data, pose_limits=[0, 0.001], fixed_base=True, no_bottom=no_bottom)
            fixed_base_obj_name = obj_base_name + "_fixed_base"  # Append a suffix to the fixed-base URDF filename
            write_urdf_file(root=root2, out_path=label_dir, file_name=fixed_base_obj_name)


# Clean up intermediate files
shutil.rmtree(temp_diretory)
