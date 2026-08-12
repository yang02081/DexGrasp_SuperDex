import os
import glob
import argparse

# --- Configuration ---
# Project root
parser = argparse.ArgumentParser(description="Convert Inspire hand GLB meshes and update its URDF.")
parser.add_argument(
    "--project-root",
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
args = parser.parse_args()
project_root = os.path.abspath(args.project_root)

from bs4 import BeautifulSoup
import trimesh

# Inspire hand meshes directory (where the GLB files live)
meshes_dir = os.path.join(project_root, "rsc", "inspire_hand", "meshes", "visual")

# Path to inspire_hand_right_envusing.urdf
urdf_file_path = os.path.join(project_root, "rsc", "inspire_hand", "inspire_hand_right_envusing.urdf")


# --- GLB to OBJ conversion ---
print(f"Searching for .glb files in '{meshes_dir}' and converting them to .obj...")
converted_files = []
for glb_path in glob.glob(os.path.join(meshes_dir, "*.glb")):
    try:
        mesh = trimesh.load(glb_path)
        obj_path = glb_path.replace(".glb", ".obj")
        mesh.export(obj_path)
        print(f"Converted successfully: '{os.path.basename(glb_path)}' -> '{os.path.basename(obj_path)}'")
        converted_files.append((os.path.basename(glb_path), os.path.basename(obj_path)))
    except Exception as e:
        print(f"Failed to convert '{os.path.basename(glb_path)}': {e}")

if not converted_files:
    print("No .glb files were found for conversion.")
else:
    print("All .glb files have been converted.")

# --- Update URDF file ---
print(f"\nUpdating URDF file '{urdf_file_path}'...")
try:
    with open(urdf_file_path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()

    soup = BeautifulSoup(urdf_content, 'xml')
    changes_made = False

    for mesh_tag in soup.find_all('mesh'):
        filename = mesh_tag.get('filename')
        if filename and filename.endswith('.glb'):
            new_filename = filename.replace('.glb', '.obj')
            mesh_tag['filename'] = new_filename
            changes_made = True
            print(f"  - Updated: '{filename}' -> '{new_filename}'")

    if changes_made:
        with open(urdf_file_path, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        print("URDF file updated successfully.")
    else:
        print("No .glb references were found in the URDF file.")

except FileNotFoundError:
    print(f"Error: URDF file '{urdf_file_path}' not found. Please check the path.")
except Exception as e:
    print(f"Error while updating the URDF file: {e}")

print("\n--- Script finished ---")
print("Please rerun the simulation environment and verify the visualization.")
