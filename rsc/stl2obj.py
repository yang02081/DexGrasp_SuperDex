import os
import trimesh

def convert_stl_to_obj(input_dir, output_dir):
    """
    Converts all .STL files in a directory to .obj format.

    Args:
        input_dir (str): The directory containing .STL files.
        output_dir (str): The directory to save .obj files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".STL") or filename.endswith(".stl"):
            stl_path = os.path.join(input_dir, filename)
            
            # Create a new filename with .obj extension
            obj_filename = os.path.splitext(filename)[0] + ".obj"
            obj_path = os.path.join(output_dir, obj_filename)
            
            try:
                # Load the STL mesh
                mesh = trimesh.load_mesh(stl_path)
                
                # Export as OBJ
                mesh.export(obj_path)
                
                print(f"Converted {stl_path} to {obj_path}")
            except Exception as e:
                print(f"Failed to convert {stl_path}: {e}")

if __name__ == '__main__':
    # The URDF file seems to reference meshes from a 'revo2_description' package,
    # which is likely located relative to a known path.
    # Based on the project structure, the meshes are likely in:
    # rsc/brainco/meshes/revo2_right_hand/
    
    # Directory containing the .STL files
    input_mesh_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'brainco', 'meshes', 'revo2_right_hand'))
    
    # Directory to save the converted .obj files
    output_mesh_dir = input_mesh_dir
    
    print(f"Input directory: {input_mesh_dir}")
    print(f"Output directory: {output_mesh_dir}")
    
    convert_stl_to_obj(input_mesh_dir, output_mesh_dir)
