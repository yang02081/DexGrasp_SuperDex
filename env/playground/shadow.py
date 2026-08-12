#!/usr/bin/python

from ruamel.yaml import YAML, dump, RoundTripDumper
from dexgrasp_zero.env.bin.cross_hand import Shadow_Floating as mano
from dexgrasp_zero.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from dexgrasp_zero.helper.utils import get_cross_hand_env_cfg
import sys
import termios
import tty
import os
import time
import numpy as np



def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

joint_info = [
    {"name": "x_joint", "axis": "1 0 0"},
    {"name": "y_joint", "axis": "0 1 0"},
    {"name": "z_joint", "axis": "0 0 1"},
    {"name": "x_rotation_joint", "axis": "1 0 0"},
    {"name": "y_rotation_joint", "axis": "0 1 0"},
    {"name": "z_rotation_joint", "axis": "0 0 1"},
    # index
    {"name": "FFJ4", "axis": "0 -1 0"},
    {"name": "FFJ3", "axis": "1 0 0"},
    {"name": "FFJ2", "axis": "1 0 0"},
    {"name": "FFJ1", "axis": "1 0 0"},
    # middle
    {"name": "MFJ4", "axis": "0 -1 0"},
    {"name": "MFJ3", "axis": "1 0 0"},
    {"name": "MFJ2", "axis": "1 0 0"},
    {"name": "MFJ1", "axis": "1 0 0"},
    # ring
    {"name": "RFJ4", "axis": "0 1 0"},
    {"name": "RFJ3", "axis": "1 0 0"},
    {"name": "RFJ2", "axis": "1 0 0"},
    {"name": "RFJ1", "axis": "1 0 0"},
    # little/pinky
    {"name": "LFJ5", "axis": "0.573576436 0 0.819152044"},
    {"name": "LFJ4", "axis": "0 1 0"},
    {"name": "LFJ3", "axis": "1 0 0"},
    {"name": "LFJ2", "axis": "1 0 0"},
    {"name": "LFJ1", "axis": "1 0 0"},

    #thumb
    {"name": "THJ5", "axis": "0 0 -1"},
    {"name": "THJ4", "axis": "1 0 0"},
    {"name": "THJ3", "axis": "1 0 0"},
    {"name": "THJ2", "axis": "0 -1 0"},
    {"name": "THJ1", "axis": "1 0 0"},
]

cat_name = 'mixed_train'
TASK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
home_path = os.path.abspath(os.path.join(TASK_PATH, "../.."))
obj_item=  "Knife_115"
obj_list = [obj_item]
obj_path_list=[os.path.join(f"{obj_item}/{obj_item}_fixed_base.urdf")]

# config (single base cfg; override hand_model_r in code)
cfg_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "envs", "cross_hand", "cfgs"))
base_cfg = os.path.join(cfg_dir, "cross_hand_base_cfg_reg_p2.yaml")
cfg, _ = get_cross_hand_env_cfg(
    hand="shadow",
    base_cfg_path=base_cfg,
    seed=1,
    visualize=True,
    cat_name=cat_name,
    len_obj=1,
    overrides={"environment": {"root_guided": False}},
)
# Environment definition
env = VecEnv(
    obj_list,
    mano(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
    home_path,
    cat_name=cat_name,
)


env.load_multi_articulated(obj_path_list)

while(1):
    qpos_reset_r = np.zeros((1, 28), dtype='float32')
    qpos_reset_r[0,-5]=1.7
    # qpos_reset_r[0,-4]=1.7
    # qpos_reset_r[0,-3]=1.7
    qpos_reset_r[0,:3]=[1., -0., 0.502]
    qpos_reset_l = np.zeros((1, 28), dtype='float32')
    obj_pose_reset = np.zeros((1, 8), dtype='float32')
    env.reset_state(qpos_reset_r,
                    qpos_reset_l,
                    np.zeros((1, 28), 'float32'),
                    np.zeros((1, 28), 'float32'),
                    obj_pose_reset,
                    )
    target_center = np.zeros((1, 3), dtype='float32')
    object_center = np.zeros((1, 3), dtype='float32')
    env.set_goals(target_center, object_center, *[np.zeros((1, 1), 'float32') for _ in range(8)])

    action_dim_now = 0
    print(
        f"Control info: dim={action_dim_now}, joint_name={joint_info[action_dim_now]['name']}, joint_axis={joint_info[action_dim_now]['axis']}"
    )

    while 1:
        ctrl_param = 0 # Initialize ctrl_param
        """
                Keys:
                    q: quit
                    d/a: next/previous DoF, then print current joint info
                    w/s: apply +1/-1 command on current DoF

                Logging: before each step, print joint name and joint axis.
        """
        order = getch()
        if order == "q":
            exit()
        elif order == "d":
            action_dim_now +=1 
            if action_dim_now==28:
                action_dim_now = 0
            print(
                f"Control info: dim={action_dim_now}, joint_name={joint_info[action_dim_now]['name']}, joint_axis={joint_info[action_dim_now]['axis']}"
            )
            continue
        elif order == "a":
            action_dim_now -=1
            if action_dim_now == -1:
                action_dim_now = 27
            print(
                f"Control info: dim={action_dim_now}, joint_name={joint_info[action_dim_now]['name']}, joint_axis={joint_info[action_dim_now]['axis']}"
            )
            continue
        elif order == "w":
            ctrl_param = +30
        elif order == "s":
            ctrl_param = -30
        else:
            print("Unknown key. Retry.")
            continue
        action_r = np.zeros([1,28])
        action_r[0,action_dim_now]=ctrl_param
        action_l = np.zeros_like(action_r)
        print(
            f"Control info: dim={action_dim_now}, joint_name={joint_info[action_dim_now]['name']}, joint_axis={joint_info[action_dim_now]['axis']}"
        )
        reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32'))
        time.sleep(0.01)
