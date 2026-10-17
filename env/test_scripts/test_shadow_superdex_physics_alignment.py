#!/usr/bin/env python3
"""Static contract for the Raisim-aligned Shadow SuperDex resource."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'rsc/shadow/shadowhand_large.urdf'
ALIGNED = ROOT / 'rsc/shadow/shadowhand_raisim_geometry_superdex_valid_inertia.urdf'


def attrs(element):
    return {} if element is None else dict(element.attrib)


def main():
    source = ET.parse(SOURCE).getroot()
    aligned = ET.parse(ALIGNED).getroot()
    aligned_links = {x.get('name'): x for x in aligned.findall('link')}
    aligned_joints = {x.get('name'): x for x in aligned.findall('joint')}

    for link in source.findall('link'):
        other = aligned_links[link.get('name')]
        for path in ('inertial/origin', 'inertial/mass'):
            assert attrs(link.find(path)) == attrs(other.find(path)), (link.get('name'), path)

        inertia = other.find('inertial/inertia')
        mass_node = other.find('inertial/mass')
        mass = 0.0 if mass_node is None else float(mass_node.get('value', 0.0))
        if inertia is not None and mass > 0.0:
            tensor = np.array([
                [float(inertia.get('ixx')), float(inertia.get('ixy', 0.0)), float(inertia.get('ixz', 0.0))],
                [float(inertia.get('ixy', 0.0)), float(inertia.get('iyy')), float(inertia.get('iyz', 0.0))],
                [float(inertia.get('ixz', 0.0)), float(inertia.get('iyz', 0.0)), float(inertia.get('izz'))],
            ])
            assert np.isfinite(tensor).all(), link.get('name')
            assert np.linalg.eigvalsh(tensor).min() >= -1e-12, link.get('name')

        source_collisions = link.findall('collision')
        aligned_collisions = other.findall('collision')
        assert len(source_collisions) == len(aligned_collisions), link.get('name')
        for src_collision, dst_collision in zip(source_collisions, aligned_collisions):
            assert attrs(src_collision.find('origin')) == attrs(dst_collision.find('origin'))
            box = src_collision.find('geometry/box')
            mesh = dst_collision.find('geometry/mesh')
            if box is not None:
                assert mesh is not None, link.get('name')
                expected = np.fromstring(box.get('size'), sep=' ')
                loaded = trimesh.load_mesh(ALIGNED.parent / mesh.get('filename'), force='mesh')
                assert np.allclose(loaded.extents, expected, atol=1e-6), (
                    link.get('name'), loaded.extents, expected
                )

    for joint in source.findall('joint'):
        other = aligned_joints[joint.get('name')]
        for path in ('origin', 'axis', 'limit', 'dynamics'):
            assert attrs(joint.find(path)) == attrs(other.find(path)), (joint.get('name'), path)

    print('Shadow SuperDex physics resource alignment: PASS')


if __name__ == '__main__':
    main()
