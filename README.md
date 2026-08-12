
# DexGrasp-Zero

Official implementation of **DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping** (RSS 2026).

Project page: <https://yliangwu.github.io/DexGrasp-Zero/docs/>  
Paper: <https://arxiv.org/abs/2603.16806>

This project implements a reinforcement learning framework for cross-embodiment grasping generalization across multiple dexterous hands, with a unified semantic action space and graph-based state representation.

---

## Installation and build

For good practice for Python package management, it is recommended to use virtual environments (e.g., `virtualenv` or `conda`) to ensure packages from different projects do not interfere with each other.

1. **Download GraspXL and set up RaiSim**

   Download [GraspXL](https://github.com/zdchan/GraspXL) and set up RaiSim by following its README. A valid RaiSim license is required.

2. **Place the code**
   Clone or copy this repository as `dexgrasp_zero` under the GraspXL `raisimGymTorch` directory:

   ```
   GraspXL/raisimGymTorch/dexgrasp_zero/
   ```

3. **Build the RaiSim extension**

   The directory placement is required because the build resolves RaiSim, Eigen,
   and pybind11 from the surrounding GraspXL checkout. Put this repository at
   `GraspXL/raisimGymTorch/dexgrasp_zero/`, then install the dependencies
   and build the extension in editable mode:

   ```
   conda activate graspxl310
   cd GraspXL/raisimGymTorch/dexgrasp_zero
   # Install a PyTorch build matching your CUDA driver first.
   pip install -r requirements.txt
   python -m pip install -e . --no-build-isolation
   ```

   Re-run the editable-install command after changing a C++ environment header.
   A successful Python 3.10 build creates
   `env/bin/cross_hand.cpython-310-x86_64-linux-gnu.so`. The build locates RaiSim
   at `GraspXL/raisim/` and Eigen/pybind11 under `GraspXL/thirdParty/`.

## Experiments

The provided model artifacts are under `model/multi_hand`. Evaluation can use this directory directly. Training and evaluation default to the YCB dataset; the evaluator resolves `ycb` to `rsc/ycb_crossDex`.

Run the Python entry points from the GraspXL `raisimGymTorch` parent directory so that the `dexgrasp_zero` package is importable:

```
cd GraspXL/raisimGymTorch
```


Training example (you can choose one or more hand models for training):
```
python -m dexgrasp_zero.env.envs.cross_hand.runner \
    -hand  shadow \   
    -e 0105train \
    -ln train_for_shadow \
	-dataset ycb
```
Multi-hand training:

```
python -m dexgrasp_zero.env.envs.cross_hand.runner \
    -hand  allegro shadow ability schunk \   
    -e 0105train \
    -ln train_for_allegro+shadow+ability+schunk \
    -obj_repeate_scale 1 \
	-dataset ycb
```

Evaluation on YCB:

```
python -m dexgrasp_zero.env.test_scripts.eval \
    --hand leap \
    --weight_dir dexgrasp_zero/model/multi_hand \
    --updates 25 \
    --dataset ycb
```

- Automatically loads the latest checkpoint and its config under `weight_dir`;
- Runs 25 evaluation updates by default; `--updates 25` is shown explicitly for reproducibility;
- You can specify a particular checkpoint with `--weight_id`;
- Uses YCB (`rsc/ycb_crossDex`) by default, so `--dataset ycb` may be omitted;
- Supports zero-shot cross-hand testing (e.g., train on `allegro`, test on `shadow`).

## Visualization

DexGrasp-Zero uses the RaiSimUnity visualizer shipped with GraspXL. On Linux,
start RaiSimUnity in one terminal:

```
cd GraspXL
./raisimUnity/linux/raisimUnity.x86_64
```

Enable **Auto-connect** in the RaiSimUnity window. Then run a small visualized
evaluation in another terminal:

```
conda activate graspxl310
cd GraspXL/raisimGymTorch
python -m dexgrasp_zero.env.test_scripts.eval \
    --hand leap \
    --weight_dir dexgrasp_zero/model/multi_hand \
    --dataset ycb \
    --obj_num 1 \
    --obj_id 0 \
    --updates 1 \
    --visualize \
    --realtime
```

`--visualize` starts the RaiSim server on port `8081`, and only the first
environment is rendered. `--realtime` limits the simulation to the configured
control rate so that the motion is easier to inspect. Omit `--realtime` for
maximum evaluation speed. Only run one visualized evaluation at a time because
the visualization port is fixed.

## BibTeX

```
@misc{wu2026dexgraspzeromorphologyalignedpolicyzeroshot,
      title={DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping}, 
      author={Yuliang Wu and Yanhan Lin and WengKit Lao and Yuhao Lin and Yi-Lin Wei and Wei-Shi Zheng and Ancong Wu},
      year={2026},
      eprint={2603.16806},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.16806}, 
}
```

## License

The repository does not currently include a license file. Add the project license and review the RaiSim/GraspXL and asset licenses before public release.


# Acknowledgement 
This implementation builds on the following codebases:

raisim: https://raisim.com/

GraspXL: https://github.com/zdchan/GraspXL

RobustDexGrasp: https://zdchan.github.io/Robust_DexGrasp/

# Contact
If you have any questions, issues, or feedback, please contact Yuliang Wu (wuyliang7@mail2.sysu.edu.cn).