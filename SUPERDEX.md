# Shadow evaluation with the SuperDex backend

This port keeps the original DexGrasp-Zero MAGCN model, Shadow graph, 28-D action contract, labelled YCB pre-grasps, and checkpoint. It replaces the Raisim environment used by evaluation with a single-environment SuperDex implementation.

## Status

The Shadow path supports:

- the original 28-D residual policy action;
- the 268-D policy observation and 179-D global state contracts;
- labelled YCB reset poses;
- four physics substeps per 10 ms policy step;
- hand-object contact queries and table support;
- root guidance, lift evaluation, GUI replay, and per-step traces.

The port is still being calibrated for backend-specific contact and wrist dynamics. `--freeze-wrist-rotation` and `--zero-wrist-rotation-action` are diagnostic options and must not be used for final protocol numbers.

## Requirements

The checkout must contain the original project artifacts:

- `model/multi_hand/full_2700_r.pt`;
- `model/multi_hand/mean_r2700.csv` and `var_r2700.csv`;
- `rsc/ycb_crossDex` and its labelled pre-grasp files.

A Python environment with SuperDex, PyTorch, SciPy, NumPy, trimesh, and joblib is required. Activate that environment before running the commands below; virtual environments are not part of the repository.

## GUI smoke test

From the repository root:

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

Use `--trace-output PATH` to record every policy/control step. Use `--disable-lift-guidance` to inspect grasp interaction before the lift test.

## YCB45 evaluation

```bash
python env/test_scripts/run_shadow_superdex_public_ycb45.py \
  --episodes-per-object 25 \
  --output-dir runs/shadow_superdex/ycb45
```

See `env/test_scripts/crossdex_ycb_45.txt` for the object list. Compare results only when the checkpoint, object list, pre-grasp labels, episode length, lift command, and success rule are identical.

## Source layout

- `env/superdex/shadow_superdex_env.py`: low-level SuperDex scene and controller.
- `env/superdex/shadow_crosshand_superdex_env.py`: Raisim-compatible Shadow action, observation, reset, contact, and root-guidance contract.
- `env/superdex/shadow_superdex_spec.py`: Shadow joint/action mapping.
- `env/test_scripts/eval_shadow_superdex.py`: single-object evaluation and GUI entry point.
- `env/test_scripts/run_shadow_superdex_*ycb45.py`: batch evaluation protocols.
- `env/test_scripts/compare_shadow_backends.py`: paired Raisim/SuperDex comparison; Raisim requires a valid local license.

## Current wrist-rotation diagnosis

The root orientation target now follows the `SHADOW_TEST` implementation and uses the current-wrist-to-desired-wrist relative rotation. This removed continuous rotation before the lift step. Some lift-phase rotation remains under investigation; keep diagnostic and final evaluation results separate.
