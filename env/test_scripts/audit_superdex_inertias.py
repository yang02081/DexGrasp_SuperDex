#!/usr/bin/env python3
"""Audit URDF mass and inertia tensors for SuperDex physical validity."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET


def audit_urdf(path: Path, asset: str):
    rows=[]
    root=ET.parse(path).getroot()
    for link in root.findall('link'):
        inertial=link.find('inertial')
        if inertial is None: continue
        mass_el=inertial.find('mass'); inertia=inertial.find('inertia')
        if mass_el is None or inertia is None: continue
        mass=float(mass_el.get('value'))
        a=inertia.attrib
        matrix=np.array([[float(a['ixx']),float(a.get('ixy',0)),float(a.get('ixz',0))],
                         [float(a.get('ixy',0)),float(a['iyy']),float(a.get('iyz',0))],
                         [float(a.get('ixz',0)),float(a.get('iyz',0)),float(a['izz'])]])
        eig=np.linalg.eigvalsh(matrix)
        positive=bool(np.all(eig>=-1e-12))
        triangle=bool(eig[-1] <= eig[0]+eig[1]+1e-12)
        rows.append(dict(asset=asset,urdf=str(path),link=link.get('name'),mass=mass,
                         eig_min=float(eig[0]),eig_mid=float(eig[1]),eig_max=float(eig[2]),
                         positive_semidefinite=positive,triangle_inequality=triangle,
                         physically_valid=positive and triangle))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]; workspace=root.parent
    rows=audit_urdf(root/'rsc/shadow/shadowhand_large.urdf','shadow')
    ycb=root/'rsc/ycb_crossDex'
    for d in sorted(x for x in ycb.iterdir() if x.is_dir()):
        urdf=d/f'{d.name}.urdf'
        if urdf.is_file(): rows.extend(audit_urdf(urdf,d.name))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'inertia_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    bad=[r for r in rows if not r['physically_valid']]
    summary={'links_checked':len(rows),'invalid_count':len(bad),'invalid':bad}
    (a.output_dir/'inertia_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
