#!/usr/bin/env python3
"""Export Shadow collision-box world transforms at an official reset state."""
from __future__ import annotations
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parent
sys.path.insert(0,str(WORKSPACE))
sys.path.insert(0,str(ROOT/'env/test_scripts'))
from dexgrasp_zero.env.superdex.shadow_crosshand_superdex_env import ShadowCrossHandSuperDexEnv,_transforms,_rotation,_vec3
from eval_shadow_superdex import official_shadow_reset


def vec(text,default='0 0 0'): return np.fromstring(text or default,sep=' ')

def main():
 p=argparse.ArgumentParser();p.add_argument('--object-id',default='024_bowl');p.add_argument('--label-index',type=int,default=16);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 source=ET.parse(ROOT/'rsc/shadow/shadowhand_large.urdf').getroot()
 aligned=ET.parse(ROOT/'rsc/shadow/shadowhand_raisim_geometry_superdex_valid_inertia.urdf').getroot()
 aligned_links={x.get('name'):x for x in aligned.findall('link')}
 with ShadowCrossHandSuperDexEnv(object_id=a.object_id,object_root=ROOT/'rsc/ycb_crossDex') as env:
  official_shadow_reset(env,object_root=ROOT/'rsc/ycb_crossDex',object_id=a.object_id,update=a.label_index)
  tf=_transforms(env.backend.hand); idx={n:i for i,n in enumerate(env.backend.hand.get_articulated_shape_info().link_names)}
  rows=[]
  for link in source.findall('link'):
   name=link.get('name'); dst=aligned_links[name]
   if name not in idx: continue
   link_tf=tf[idx[name]]; link_pos=_vec3(link_tf.translation); link_rot=_rotation(link_tf)
   for ci,(sc,dc) in enumerate(zip(link.findall('collision'),dst.findall('collision'))):
    so=sc.find('origin'); do=dc.find('origin'); xyz=vec(None if so is None else so.get('xyz')); rpy=vec(None if so is None else so.get('rpy'))
    dxyz=vec(None if do is None else do.get('xyz')); drpy=vec(None if do is None else do.get('rpy'))
    box=sc.find('geometry/box'); mesh=dc.find('geometry/mesh')
    rows.append({'link':name,'collision_index':ci,'source_aligned_origin_match':bool(np.allclose(xyz,dxyz) and np.allclose(rpy,drpy)),'local_xyz':xyz.tolist(),'local_rpy':rpy.tolist(),'world_center':(link_pos+link_rot.apply(xyz)).tolist(),'world_quaternion_xyzw':(link_rot*Rotation.from_euler('XYZ',rpy)).as_quat().tolist(),'source_box_size':None if box is None else vec(box.get('size')).tolist(),'aligned_mesh':None if mesh is None else mesh.get('filename')})
 out={'object_id':a.object_id,'label_index':a.label_index,'collision_count':len(rows),'all_origins_match':all(x['source_aligned_origin_match'] for x in rows),'collisions':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('object_id','label_index','collision_count','all_origins_match')},indent=2))
if __name__=='__main__':main()
