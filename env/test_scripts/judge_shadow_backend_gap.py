#!/usr/bin/env python3
"""Paired outcome test for the same Shadow objects and pre-grasp labels."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def load_raisim(path: Path):
    data = json.loads(path.read_text())
    return {
        (str(r['object_id']), int(r['label_index'])): bool(r['success'])
        for r in data['results']
    }, data


def load_superdex(root: Path):
    outcomes = {}
    documents = []
    for path in sorted(root.glob('*.json')):
        if path.name == 'aggregate_summary.json':
            continue
        data = json.loads(path.read_text())
        if data.get('backend') != 'superdex':
            continue
        documents.append(data)
        object_id = path.stem
        for row in data['results']:
            outcomes[(object_id, int(row['label_index']))] = bool(row['paper_success'])
    return outcomes, documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raisim', type=Path, required=True)
    parser.add_argument('--superdex-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    raisim, raisim_doc = load_raisim(args.raisim)
    superdex, superdex_docs = load_superdex(args.superdex_dir)
    if not raisim or not superdex:
        raise RuntimeError('empty Raisim or SuperDex result set')
    if set(raisim) != set(superdex):
        only_r = sorted(set(raisim) - set(superdex))
        only_s = sorted(set(superdex) - set(raisim))
        raise RuntimeError(
            f'paired keys differ: Raisim-only={only_r[:10]}, SuperDex-only={only_s[:10]}'
        )

    keys = sorted(raisim)
    both_success = sum(raisim[k] and superdex[k] for k in keys)
    raisim_only = sum(raisim[k] and not superdex[k] for k in keys)
    superdex_only = sum(not raisim[k] and superdex[k] for k in keys)
    both_fail = sum(not raisim[k] and not superdex[k] for k in keys)
    n = len(keys)
    r_rate = float(np.mean([raisim[k] for k in keys]))
    s_rate = float(np.mean([superdex[k] for k in keys]))
    agreement = (both_success + both_fail) / n
    discordant = raisim_only + superdex_only
    p_value = float(binomtest(raisim_only, discordant, 0.5).pvalue) if discordant else 1.0

    per_object = {}
    grouped = defaultdict(list)
    for key in keys:
        grouped[key[0]].append(key)
    for obj, obj_keys in sorted(grouped.items()):
        per_object[obj] = {
            'episodes': len(obj_keys),
            'raisim_success_rate': float(np.mean([raisim[k] for k in obj_keys])),
            'superdex_success_rate': float(np.mean([superdex[k] for k in obj_keys])),
            'raisim_only_successes': sum(raisim[k] and not superdex[k] for k in obj_keys),
            'superdex_only_successes': sum(not raisim[k] and superdex[k] for k in obj_keys),
        }

    gap = r_rate - s_rate
    if r_rate >= 0.85 and gap >= 0.10 and raisim_only > superdex_only and p_value < 0.05:
        verdict = 'simulation_backend_gap_dominates'
        explanation = (
            'The unchanged policy succeeds significantly more often in Raisim on the same '
            'object-label pairs; the remaining gap is dominated by simulator dynamics.'
        )
    elif abs(gap) <= 0.05 and agreement >= 0.80:
        verdict = 'policy_or_subset_dominates'
        explanation = (
            'The backends have similar success and mostly agree on the same labels; failures '
            'are mainly attributable to the policy or this evaluation subset.'
        )
    else:
        verdict = 'mixed_or_inconclusive'
        explanation = (
            'Both simulator and policy/subset effects remain plausible; inspect per-object '
            'paired failures and step traces for the discordant labels.'
        )

    result = {
        'episodes': n,
        'raisim_success_rate': r_rate,
        'superdex_success_rate': s_rate,
        'paired_gap': gap,
        'both_success': both_success,
        'raisim_success_superdex_fail': raisim_only,
        'raisim_fail_superdex_success': superdex_only,
        'both_fail': both_fail,
        'paired_agreement_rate': agreement,
        'mcnemar_exact_p_value': p_value,
        'verdict': verdict,
        'explanation': explanation,
        'per_object': per_object,
        'provenance': {
            'raisim_checkpoint': raisim_doc.get('checkpoint'),
            'superdex_checkpoints': sorted({d.get('checkpoint') for d in superdex_docs}),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
