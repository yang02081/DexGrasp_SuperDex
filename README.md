# DexGrasp-Zero on SuperDex

This repository ports the **Shadow Hand evaluation path of DexGrasp-Zero** from Raisim to the SuperDex physics backend. The original MAGCN architecture, pretrained checkpoint, graph representation, 28-D action interface, labelled YCB pre-grasps, and evaluation data are retained; the simulation, controller, contact queries, and GUI run through SuperDex.

> Current scope: Shadow Hand evaluation and physics-alignment diagnostics. Allegro has been validated separately, but its cross-hand MAGCN backend is not yet included here. Other hand morphologies are future work.

## Implemented

- Shadow Hand loading with repaired inertial and Raisim-aligned collision resources;
- original 28-D residual policy action;
- 268-D MAGCN observation and 179-D global-state contracts;
- original `model/multi_hand/full_2700_r.pt` checkpoint loading;
- labelled YCB reset poses and YCB45 batch evaluation;
- 10 ms policy period with four SuperDex physics substeps;
- Pose Controller actuation, gravity, table support, and object release;
- hand-object contact and impulse features;
- root guidance, lift evaluation, GUI replay, and per-step traces;
- paired Raisim/SuperDex diagnostic entry point.

The port remains under calibration for backend-specific contact and lift-phase wrist dynamics. The options `--freeze-wrist-rotation` and `--zero-wrist-rotation-action` are diagnostic controls and must not be used when reporting final protocol results.

## Requirements

Use a Python environment containing:

- SuperDex;
- PyTorch compatible with the installed CUDA driver;
- NumPy, SciPy, trimesh, joblib, and PyYAML.

The repository already contains the original MAGCN model artifacts and YCB resources expected by the evaluator. Virtual environments and generated results are intentionally excluded from Git.

## GUI evaluation

Run from the repository root:

```bash
python env/test_scripts/eval_shadow_superdex.py \
  --object-id 058_golf_ball \
  --episodes 1 \
  --steps 200 \
  --seed 1 \
  --label-index 16 \
  --visualize \
  --realtime-factor 0.25 \
  --output runs/shadow_superdex/gui_label16.json
```

Useful diagnostic options:

```text
--trace-output PATH               Save every policy/control step
--disable-lift-guidance           Inspect interaction without the lift test
--lift-ramp-steps 50              Apply the lift target gradually
--freeze-wrist-rotation           Lock wrist rotation for diagnosis
--zero-wrist-rotation-action      Zero policy action[3:6] while retaining guidance
```

## YCB45 evaluation

```bash
python env/test_scripts/run_shadow_superdex_public_ycb45.py \
  --episodes-per-object 25 \
  --output-dir runs/shadow_superdex/ycb45
```

The object list is stored in `env/test_scripts/crossdex_ycb_45.txt`. Results should only be compared when the checkpoint, object list, labelled pre-grasps, episode length, lift command, and success rule are identical.

## Tests

```bash
python env/test_scripts/test_shadow_superdex_contract.py
python env/test_scripts/test_shadow_superdex_physics_alignment.py
```

## Code layout

- `env/superdex/shadow_superdex_env.py`: SuperDex scene, actors, controller, and contacts.
- `env/superdex/shadow_crosshand_superdex_env.py`: DexGrasp-Zero action, observation, reset, and root-guidance contract.
- `env/superdex/shadow_superdex_spec.py`: Shadow joint and action mapping.
- `env/test_scripts/eval_shadow_superdex.py`: single-object evaluation and GUI entry point.
- `env/test_scripts/run_shadow_superdex_*ycb45.py`: batch evaluation protocols.
- `env/test_scripts/compare_shadow_backends.py`: paired Raisim/SuperDex diagnosis; its Raisim side requires a valid Raisim license.
- `SUPERDEX.md`: additional implementation notes.

## Upstream project

This work is based on [DexGrasp-Zero](https://github.com/YliangWu/DexGrasp-Zero-code):

> DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping, RSS 2026.

- [Project page](https://yliangwu.github.io/DexGrasp-Zero/docs/)
- [Paper](https://www.roboticsproceedings.org/rss22/p122.pdf)

```bibtex
@misc{wu2026dexgraspzeromorphologyalignedpolicyzeroshot,
  title={DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping},
  author={Yuliang Wu and Yanhan Lin and WengKit Lao and Yuhao Lin and Yi-Lin Wei and Wei-Shi Zheng and Ancong Wu},
  year={2026},
  eprint={2603.16806},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2603.16806}
}
```

## License and assets

The upstream checkout does not currently include a license file. Review the DexGrasp-Zero, SuperDex, GraspXL, model, YCB, and robot-asset licenses before redistribution or public release.
