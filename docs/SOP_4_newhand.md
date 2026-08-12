# SOP: Integrating a New Dexterous Hand into GraspXL

This SOP explains how to integrate a dexterous hand, such as a LEAP or Faive Hand, into the GraspXL RaiSim environment. The integration must support contact detection, 6-DoF wrist control, observation construction, and reinforcement-learning training.

Core rules:

- Preserve the original URDF joint and link names.
- Make every change explicit, traceable, and documented.
- Confirm hand-specific settings manually from the URDF.
- Hard-code and validate observation, action, and contact dimensions.

---

## 1. Inspect the original URDF and meshes

Obtain the standard URDF, for example `leap_hand_right.urdf`, and verify that it contains:

- the complete joint hierarchy, including the palm or base link;
- visual and collision geometry;
- valid mesh paths, normally `meshes/xxx.obj` or `.STL` files.

Read the URDF and record the number of fingers, the DoF of each finger, the palm link name, every contactable finger link, and every tip link or tip frame used by observations.

> Deliverable: a table of contactable links and tip frames for every finger.

---

## 2. Create the environment URDF

Create a file such as `leap_hand_right_envusing.urdf` from the standard URDF.

### 2.0 Formatting requirements

- Use four-space indentation.
- Leave a blank line between each `<link>` and `<joint>` block.
- Make the file readable in common IDEs.
- Mark every edited block with `<!-- modified -->`.

### 2.0.1 Mesh compatibility

RaiSim does not reliably support `.glb` meshes. Convert all meshes referenced by URDF `visual` or `collision` elements to `.obj` before use. Blender can perform this conversion; enable **Triangulate Faces** during export.

### 2.1 Add the 6-DoF virtual base

Insert the following chain below the `<robot>` root and before the palm link:

```xml
<!-- modified: 6-DoF virtual base for wrist motion -->
<link name="world"/>
<joint name="world_to_base" type="fixed">
  <parent link="world"/>
  <child link="allegro_root0"/>
</joint>
<link name="allegro_root0"><inertial><mass value="0.0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>
<link name="allegro_root1"><inertial><mass value="0.0"/><inertia ... /></inertial></link>
<link name="allegro_root2"><inertial><mass value="0.0"/><inertia ... /></inertial></link>
<link name="allegro_root3"><inertial><mass value="0.0"/><inertia ... /></inertial></link>
<link name="allegro_root4"><inertial><mass value="0.0"/><inertia ... /></inertial></link>
<link name="allegro_root5"><inertial><mass value="0.0"/><inertia ... /></inertial></link>
<joint name="x_joint" type="prismatic"><axis xyz="1 0 0"/><limit lower="-0.5" upper="0.5"/><parent link="allegro_root0"/><child link="allegro_root1"/></joint>
<joint name="y_joint" type="prismatic"><axis xyz="0 1 0"/><limit lower="-0.5" upper="0.5"/><parent link="allegro_root1"/><child link="allegro_root2"/></joint>
<joint name="z_joint" type="prismatic"><axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5"/><parent link="allegro_root2"/><child link="allegro_root3"/></joint>
<joint name="x_rotation_joint" type="revolute"><axis xyz="1 0 0"/><limit lower="-3.14" upper="3.14"/><parent link="allegro_root3"/><child link="allegro_root4"/></joint>
<joint name="y_rotation_joint" type="revolute"><axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14"/><parent link="allegro_root4"/><child link="allegro_root5"/></joint>
<joint name="z_rotation_joint" type="revolute"><axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/><parent link="allegro_root5"/><child link="[PALM_LINK_NAME]"/></joint>
```

Replace `[PALM_LINK_NAME]` with the actual palm link, such as `palm` or `base_link`. Set every `allegro_root*` mass and inertia to zero. Remove any original fixed base joint and base link connecting the palm to `world`; the palm must have only one parent joint.

### 2.2 Configure joint dynamics

Add `<dynamics>` to every movable joint. Start with damping `1.0` and friction `0.0001`. Use damping `10.0` for wrist translation and rotation joints. Increase any joint velocity limit below `7` to `7`, unless the hand model requires a different value.

```xml
<!-- modified: joint dynamics -->
<joint name="your_joint_name" type="revolute">
  <!-- other joint properties -->
  <dynamics damping="1.0" friction="0.0001"/>
</joint>
```

### 2.3 Collision simplification (optional)

This step is currently optional. A fingertip collision may be simplified to a sphere, and proximal or middle links may use boxes to reduce mesh-mesh contact noise:

```xml
<link name="fftip">
  <visual>...</visual>
  <collision><geometry><sphere radius="0.015"/></geometry></collision>
</link>
```

Every contactable link must still have collision geometry, and every changed block must be marked `<!-- modified -->`.

---

## 3. Update `Environment.hpp`

### 3.1 Define `contact_bodies_r_`

List the palm and the proximal, middle, and distal links of every finger, excluding the tip. Names must exactly match URDF link names. For a multi-link thumb base, choose the single link that best represents the functional contact region and avoid redundant bodies.

```cpp
std::string contact_bodies_r_[?] = {
    "palm",
    "ffproximal", "ffmiddle", "ffdistal",
    "mfproximal", "mfmiddle", "mfdistal"
};
```

### 3.2 Define `num_contacts`

Define the array first and derive its length:

```cpp
num_contacts = sizeof(contact_bodies_r_) / sizeof(contact_bodies_r_[0]);
```

### 3.3 Define `body_parts_r_`

Use URDF joint names or frame names created with `addFrameToLink`, not link names. `getFramePosition` may fail or crash when passed a link name.

```cpp
std::string body_parts_r_[17] = {
    "z_rotation_joint",
    "1", "0", "2", "index_tip",
    "5", "4", "6", "middle_tip",
    "9", "8", "10", "ring_tip",
    "12", "13", "14", "thumb_tip"
};
```

### 3.4 Define `num_bodyparts`

```cpp
num_bodyparts = sizeof(body_parts_r_) / sizeof(body_parts_r_[0]);
```

Review every hard-coded `body_parts_r_[index]` access when this array changes. Visualization arrays such as `Cylinder`, `sphere`, and `joints_sphere` must have exactly `num_bodyparts` entries.

In `set_goals`, compute `hand_center_temp` by averaging the intermediate joints of the non-thumb fingers, then average that result with the thumb tip. Derive all indices and the final divisor from the actual array layout.

### 3.5 Compute `obDim_r_` and `gsDim_`

The raw observation is:

```cpp
obDouble_r_ <<
    target_center_dif, wrist_ori_error,
    gc_r_.tail(gcDim_ - 6), torque_error,
    wrist_vel, wrist_qvel,
    contacts_r_af, impulses_r_af,
    obj_vel_in_wrist, obj_qvel_in_wrist,
    contacts_r_non_af, impulses_r_non_af;
```

With `n = gcDim_` and `N = num_contacts`:

```text
obDim_r_ = 12 + 2*n + 4*N
obs_dim_r = obDim_r_ + 3*num_bodyparts + 3*num_bodyparts
```

The Python runner must use the final `obs_dim_r`.

For the regular environment:

```text
gsDim_ = 10 + 6*num_bodyparts
```

For the `_test` environment:

```text
gsDim_ = 3 + 3*num_bodyparts + 3*num_bodyparts + 3 + 3 + 1 + 3 + 4 + (gcDim_ - 6) + 8
       = 19 + 6*num_bodyparts + gcDim_
```

The corresponding state order is wrist/object pose, per-body-part axes and positions, object position, grasp-axis error, alignment scalar, wrist pose, non-base joint coordinates, and object state. Recompute all Python slices from this order.

### 3.6 Tune contact weights and impulse limits

Start `finger_weights_contact` with `setOnes(num_contacts)`. Tune fingertip and thumb weights as needed. Set `impulse_high/low` according to the hand's contact behavior, and ensure all vectors have length `num_contacts`.

### 3.7 Update the URDF loading path

```cpp
mano_r_ = world_->addArticulatedSystem(
    resourceDir + "/leap_hand/" + hand_model_r,
    "", {}, raisim::COLLISION(0),
    raisim::COLLISION(0) | raisim::COLLISION(2) | raisim::COLLISION(63));
```

### 3.8 C++ consistency checks

Verify that `num_contacts`, `num_bodyparts`, `gcDim_`, and `obDim_r_` match the URDF and the formulas above. RaiSim treats mimic joints as independent joints. Print these values while debugging.

---

## 4. Register the hand in Python

### 4.1 Update `RaisimGymVecEnvOther.py`

Add a loading branch for the new hand and set its YAML model:

```yaml
hand_model_r: "leap_hand_right_envusing.urdf"
```

The `dis_info` array is `[min_dis_af, min_dis_non_af]`; each half has `num_bodyparts` entries:

```python
affordance_reward_r = -np.sum(dis_info[:, :num_bodyparts] * finger_weights, axis=1)
non_affordance_reward_r = -np.sum(dis_info[:, num_bodyparts:] * finger_weights, axis=1)
```

An incorrect slice causes broadcasting errors. The action dimension is six virtual-base DoFs plus all independent finger DoFs. For Inspire, the example is `6 + 2 + 1 + 1 + 1 + 1 = 12`.

### 4.2 Configure the initial pose

Add the correct `rot_mat` in `helper/initial_pose_final.py`. It maps the canonical grasp direction to the hand-base coordinate system. Determine the hand's forward axis from the URDF and verify the result visually.

```python
elif hand_type == "leap":
    rot_mat = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ]).reshape(3, 3)
```

Pass the correct `hand_type` to every initial-pose function caller.

---

## 5. Update runner and evaluation scripts

1. Set `ob_dim_r = obDim_r_ + 6 * num_bodyparts`.
2. Obtain `act_dim` from `env.num_acts`.
3. Replace the hand name in the environment configuration.
4. Inspect reset values such as `qpos_reset_r[i, -4] = 1.7`; this index is hand-specific and must be mapped to the correct URDF joint.
5. Make `qpos_reset_r`, `qpos_reset_l`, `obj_pose_reset`, and reset velocities match the C++ dimensions.
6. Match `finger_weights` to the contact dimension.
7. Pass the new hand name to initial-pose and observation functions.
8. Recompute `_test` global-state slices for `lifted`, `current_direction_error`, and `current_angle_error`.
9. Verify every `dis_info` slice against `obDouble_r_`.

Use the matching `[hand]_[environment]` module and `NormalSampler`. Keep floating-specific logic, such as lowest-point processing, when synchronizing floating and test runners.

---

## 6. Validation checklist

- [ ] The URDF loads without errors and the hand is visible.
- [ ] `mano_r_->getBodyIdx(link_name)` returns a valid ID for every contact body.
- [ ] `mano_r_->getFramePosition(frame_name)` succeeds for every body part.
- [ ] `obDim_r_` equals the actual observation length.
- [ ] Fingertip visualization spheres are correctly positioned.
- [ ] Contact and impulse vectors have length `num_contacts`.
- [ ] The observation dimension calculation in `RaisimGymVecEnvOther.py` is correct.

The hand is ready for training only after every check passes.

The complete policy observation is:

```text
obDouble_r_ <<
  target_center_dif, wrist_ori_error,
  gc_r_.tail(gcDim_-6), torque_error,
  wrist_vel, wrist_qvel,
  contacts_r_af, impulses_r_af,
  obj_vel_in_wrist, obj_qvel_in_wrist,
  contacts_r_non_af, impulses_r_non_af,
  af_vec, non_aff_vec_obs;
```

---

## 7. Synchronize floating and test environments

Synchronize the runner, `Environment.hpp`, hand YAML configuration, Python hand registry, and observation slicing. Limit floating/test changes to hand integration and dimension updates; retain logic specific to those environments.
