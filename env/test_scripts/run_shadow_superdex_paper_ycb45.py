#!/usr/bin/env python3
"""Run the CrossDex 45-object, 25-trial Shadow paper protocol in SuperDex."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OBJECT_LIST = Path(__file__).with_name('crossdex_ycb_45.txt')
EVALUATOR = Path(__file__).with_name('eval_shadow_superdex.py')
DEFAULT_PYTHON = Path(sys.executable)


def valid_result(path: Path, episodes: int) -> bool:
    try:
        data=json.loads(path.read_text())
        protocol=data.get('evaluation_protocol', {})
        return (data.get('episodes') == episodes and
                protocol.get('paper_protocol') is True and
                protocol.get('lift_height_m') == 0.5 and
                protocol.get('hold_seconds') == 2.0)
    except Exception:
        return False


def run_one(python: Path, object_id: str, out_dir: Path, episodes: int):
    output=out_dir/f'{object_id}.json'
    log=out_dir/f'{object_id}.log'
    if valid_result(output, episodes):
        return object_id, 'cached'
    cmd=[str(python),str(EVALUATOR),'--object-id',object_id,
         '--episodes',str(episodes),'--label-index','0','--seed','1',
         '--paper-protocol','--physics-substeps','4','--output',str(output)]
    child_env=os.environ.copy()
    child_env.update({
        'OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1',
        'NUMEXPR_NUM_THREADS':'1',
    })
    with log.open('w') as stream:
        proc=subprocess.run(cmd,cwd=ROOT,env=child_env,
                            stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not valid_result(output, episodes):
        raise RuntimeError(f'{object_id} failed; inspect {log}')
    return object_id, 'completed'


def aggregate(out_dir: Path, objects: list[str], episodes_per_object: int):
    payloads=[json.loads((out_dir/f'{o}.json').read_text()) for o in objects]
    rows=[]
    per_object=[]
    for object_id,data in zip(objects,payloads):
        episodes=data['results']; rows.extend(episodes)
        per_object.append({
            'object_id':object_id,
            'episodes':len(episodes),
            'paper_success_rate':sum(bool(r['paper_success']) for r in episodes)/len(episodes),
            'mean_max_lift_height':sum(float(r['max_lift_height']) for r in episodes)/len(episodes),
            'mean_max_hold_seconds':sum(float(r['paper_max_hold_seconds']) for r in episodes)/len(episodes),
            'max_hold_seconds':max(float(r['paper_max_hold_seconds']) for r in episodes),
            'mean_peak_contact_force':sum(float(r['peak_contact_force']) for r in episodes)/len(episodes),
        })
    summary={
        'backend':'superdex','hand':'shadow','protocol':'crossdex_ycb45_paper',
        'objects':len(objects),'episodes_per_object':episodes_per_object,
        'episodes':len(rows),'lift_height_m':0.5,'hold_seconds':2.0,
        'control_dt':0.01,'physics_substeps':4,'strict_physics_alignment':True,
        'paper_success_rate':sum(bool(r['paper_success']) for r in rows)/len(rows),
        'successes':sum(bool(r['paper_success']) for r in rows),
        'mean_max_lift_height':sum(float(r['max_lift_height']) for r in rows)/len(rows),
        'mean_max_hold_seconds':sum(float(r['paper_max_hold_seconds']) for r in rows)/len(rows),
        'max_hold_seconds':max(float(r['paper_max_hold_seconds']) for r in rows),
        'mean_peak_contact_force':sum(float(r['peak_contact_force']) for r in rows)/len(rows),
        'max_peak_contact_force':max(float(r['peak_contact_force']) for r in rows),
        'object_ids':objects,
    }
    (out_dir/'aggregate_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out_dir/'per_object_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_object[0]))
        writer.writeheader(); writer.writerows(per_object)
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'runs/shadow_superdex/paper_ycb45_25')
    ap.add_argument('--episodes-per-object',type=int,default=25)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--python',type=Path,default=DEFAULT_PYTHON)
    ap.add_argument('--list-only',action='store_true')
    args=ap.parse_args()
    objects=[x.strip() for x in OBJECT_LIST.read_text().splitlines() if x.strip()]
    if len(objects)!=45 or len(set(objects))!=45:
        raise RuntimeError('CrossDex object list must contain exactly 45 unique objects')
    missing=[o for o in objects if not (ROOT/'rsc/ycb_crossDex'/o).is_dir()]
    if missing: raise FileNotFoundError(f'missing YCB resources: {missing}')
    if args.list_only:
        print('\n'.join(objects)); return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(run_one,args.python,o,args.output_dir,args.episodes_per_object):o for o in objects}
        done=0
        for future in as_completed(futures):
            object_id,status=future.result(); done+=1
            print(f'[{done:02d}/45] {object_id}: {status}',flush=True)
    aggregate(args.output_dir,objects,args.episodes_per_object)

if __name__=='__main__':
    main()
