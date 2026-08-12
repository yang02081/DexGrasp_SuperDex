import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
import sys
import os

def replace_collision_bodies(envusing_urdf_path, dof6_urdf_path):
    """Replace collision bodies in an "envusing" URDF using a reference 6DoF URDF.

    For each <link>, this function replaces all <collision> elements in the envusing
    file with the corresponding <collision> elements from the 6DoF file.

    Args:
        envusing_urdf_path: Path to the URDF with the correct kinematic chain (envusing).
        dof6_urdf_path: Path to the URDF with the correct collision bodies (6DoF).

    Returns:
        The modified URDF content as a string.
    """

    # --- 1. Read and sanitize URDF contents ---
    
    try:
        with open(envusing_urdf_path, 'r', encoding='utf-8') as f:
            envusing_urdf_content = f.read()
        with open(dof6_urdf_path, 'r', encoding='utf-8') as f:
            dof6_urdf_content = f.read()
    except FileNotFoundError as e:
        return f"Error: file not found. Check the path: {e}"
    except Exception as e:
        return f"Error: failed to read file: {e}"

    def clean_urdf_content(content):
        # Remove non-standard XML headers/prefixes so the parser can find the <robot> root tag.
        content = re.sub(r'<\?xml version="1.0" .*?\?>', '', content, flags=re.DOTALL)
        content = re.sub(r'', '', content, flags=re.DOTALL)
        match = re.search(r'<robot.*</robot>', content, re.DOTALL)
        if match:
            content = match.group(0)
        return content

    envusing_cleaned = clean_urdf_content(envusing_urdf_content)
    dof6_cleaned = clean_urdf_content(dof6_urdf_content)

    # --- 2. Parse XML trees ---

    try:
        # Parse with ElementTree.
        envusing_root = ET.fromstring(envusing_cleaned)
        dof6_root = ET.fromstring(dof6_cleaned)
    except ET.ParseError as e:
        return f"Error: failed to parse URDF XML: {e}"

    # --- 3. Build link -> collision mapping from 6DoF URDF ---
    
    dof6_link_collisions = {}
    for dof6_link in dof6_root.findall('link'):
        link_name = dof6_link.get('name')
        if link_name:
            # Extract all <collision> elements.
            collisions = list(dof6_link.findall('collision'))
            dof6_link_collisions[link_name] = collisions

    # --- 4. Replace collisions in envusing URDF ---
    
    for envusing_link in envusing_root.findall('link'):
        link_name = envusing_link.get('name')
        
        if link_name and link_name in dof6_link_collisions:
            # Get collisions for the matching link name from the 6DoF URDF.
            new_collisions = dof6_link_collisions[link_name]
            
            # a) Remove all existing <collision> elements under this link.
            collisions_to_remove = envusing_link.findall('collision')
            for col in collisions_to_remove:
                envusing_link.remove(col)
            
            # b) Append cloned <collision> elements from the 6DoF URDF.
            # We must deep-clone because one Element can only have one parent.
            for col in new_collisions:
                # Deep copy
                new_col = ET.fromstring(ET.tostring(col, encoding='unicode'))
                envusing_link.append(new_col)
    
    # --- 5. Pretty-print XML ---
    
    # ElementTree output is not pretty; use minidom for indentation.
    rough_string = ET.tostring(envusing_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    
    pretty_xml_as_string = reparsed.toprettyxml(indent="  ")
    
    # Fix minidom formatting: remove extra XML declaration and blank lines.
    pretty_xml_as_string = re.sub(r'<\?xml version="1.0" .*\?>\n', '', pretty_xml_as_string, count=1)
    pretty_xml_as_string = '\n'.join([line for line in pretty_xml_as_string.splitlines() if line.strip()])
    
    # Add XML declaration manually.
    return '<?xml version="1.0" encoding="utf-8"?>\n' + pretty_xml_as_string

# --- CLI ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python replace_collision.py <envusing.urdf> <6dof.urdf> [output.urdf]"
        )
        sys.exit(2)

    envusing_path = sys.argv[1]
    dof6_path = sys.argv[2]
    output_filename = (
        sys.argv[3]
        if len(sys.argv) >= 4
        else os.path.splitext(envusing_path)[0] + "_modified_collision.urdf"
    )

    modified_urdf = replace_collision_bodies(envusing_path, dof6_path)

    if modified_urdf.startswith("Error"):
        print(modified_urdf)
        sys.exit(1)

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(modified_urdf)
        print(f"Done. Saved: {output_filename}")
    except Exception as e:
        print(f"Error: failed to write output file: {e}")
        sys.exit(1)