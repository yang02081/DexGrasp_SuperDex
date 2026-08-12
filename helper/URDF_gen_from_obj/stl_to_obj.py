import argparse
import os
import trimesh

def stl_to_obj_converter(source_folder):
    """
    Traverse the given folder and its subfolders, converting all `.stl` files to `.obj` files.
    """
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(".stl"):
                stl_path = os.path.join(root, file)
                obj_path = os.path.join(root, file.replace(".stl", ".obj"))
                
                print(f"Converting {stl_path} to {obj_path}...")
                try:
                    mesh = trimesh.load(stl_path)
                    mesh.export(obj_path)
                    print(f"Successfully converted {file}")
                except Exception as e:
                    print(f"Error converting {file}: {e}")

if __name__ == "__main__":
    default_source = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "rsc", "ycb_crossDex")
    )
    parser = argparse.ArgumentParser(description="Convert STL meshes under a directory to OBJ.")
    parser.add_argument("source_folder", nargs="?", default=default_source)
    args = parser.parse_args()
    stl_to_obj_converter(args.source_folder)

