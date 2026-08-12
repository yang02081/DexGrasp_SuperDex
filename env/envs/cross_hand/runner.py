"""Training entrypoint for the cross-hand (right-hand) Raisim environments.

This script builds one VecEnv per selected hand, shares a single base YAML config,
and overrides hand-specific fields (e.g., `environment.hand_model_r`) at runtime.
"""
from __future__ import annotations

import os
from typing import Callable
from ruamel.yaml import YAML, dump, RoundTripDumper
from dexgrasp_zero.env.bin.cross_hand import NormalSampler
from dexgrasp_zero.env.bin.cross_hand import Allegro_Floating, Leap_Floating, Shadow_Floating, Inspire_Floating, Ability_Floating, Schunk_Floating, Brainco_Floating
from dexgrasp_zero.env.RaisimGymVecEnvOther import RaisimGymVecEnvTest as VecEnv
from dexgrasp_zero.helper.raisim_gym_helper import ConfigurationSaver, load_param
import time
import numpy as np
import torch
import argparse
import  wandb
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import dexgrasp_zero.algo.ppo.module as ppo_module
import dexgrasp_zero.algo.ppo.ppo as PPO
from dexgrasp_zero.helper.urdf_parser import URDFGraphParser
from dexgrasp_zero.helper.graph_data_builder import GraphDataBuilder 
from dexgrasp_zero.helper.constants import (
    get_cross_hand_spec,
    get_cross_hand_urdf_path,
    get_data_dict,
    observe_kwargs_for_hand_type,
)
import dexgrasp_zero.algo.ppo.model_factory as model_factory

from dexgrasp_zero.helper.utils import (
    setup_seed,
    build_reset_params,
    get_cross_hand_env_cfg,
    reset_environment,
    process_obj_list,
    get_reward_info,
    find_weight_and_modelcfg,
)
TASK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


HAND_ENV_CTORS: dict[str, Callable] = {
    "allegro": Allegro_Floating,
    "leap": Leap_Floating,
    "shadow": Shadow_Floating,
    "inspire": Inspire_Floating,
    "ability": Ability_Floating,
    "schunk": Schunk_Floating,
    "brainco": Brainco_Floating,
}


def _build_data_dict_from_hands(hands: list[str]) -> dict:
    flags = dict(
        TRAIN_LEAP=("leap" in hands),
        TRAIN_ALLEGRO=("allegro" in hands),
        TRAIN_SHADOW=("shadow" in hands),
        TRAIN_SCHUNK=("schunk" in hands),
        TRAIN_INSPIRE=("inspire" in hands),
        TRAIN_ABILITY=("ability" in hands),
        TRAIN_BRAINCO=("brainco" in hands),
        TRAIN_BARRETT=("barrett" in hands),
    )
    return get_data_dict(**flags)


def main(parser):
    assert torch.cuda.is_available(), "Only support cuda device" 
    device = torch.device('cuda')   
    args = parser.parse_args()

    selected_hands = list(args.hand)
    for h in selected_hands:
        if h not in HAND_ENV_CTORS:
            raise ValueError(f"Unknown/unsupported train hand: {h}. Valid: {sorted(HAND_ENV_CTORS.keys())}")
        get_cross_hand_spec(h)

    use_wandb = args.use_wandb
    task_name = args.exp_name
    home_path = args.home_path
    visualize = args.visualize
    dataset_name = args.dataset
    if dataset_name == "ycb":
        dataset_name = "ycb_crossDex"


    setup_seed(args.seed)
    
    print(f"train hands: {selected_hands}")
    assert len(selected_hands) > 0, "At least one hand should be trained"
    assert dataset_name in ['mixed_train',"ycb_crossDex"] , "dataset_name error"
    print(f"Experiment name: \"{task_name}\"")


    obj_list, obj_path_list,obj_full_path_list = process_obj_list(home_path, dataset_name,"floating",args.obj_num,obj_repeate_scale=args.obj_repeate_scale)
    data_dict = _build_data_dict_from_hands(selected_hands)
    for hand_type in data_dict.keys():
        data_dict[hand_type]["num_envs_per_hand"] = len(obj_list)
    max_obs_dim = max( data_dict[hand_name]["obs_dim"] for hand_name in data_dict.keys() ) 
    max_act_dim = max(data_dict[hand_name]["act_dim"] for hand_name in data_dict.keys())
        
    # weight_dir = args.weight_dir
    # weight_file, model_config,model_cfg_path = find_weight_and_modelcfg(weight_dir, args.weight_id)


    model_cfg_path = args.config
    model_cfg_path = model_cfg_path.replace("v0",args.version)
    model_cfg_path = model_cfg_path.replace("L4",args.layer)
    print("using config: ",model_cfg_path)
    with open(model_cfg_path, 'r') as f:
        import yaml
        model_config = yaml.safe_load(f)

    if use_wandb:
        wandb.init(project=task_name, config=None, name=args.log_name)

    
    exist_cfg=None
    cfgs={}
    cfg_names=[]
    envs={}
    hand_order = []
    for hand in selected_hands:
        cfg, used_cfg_path = get_cross_hand_env_cfg(
            hand=hand,
            args=args,
            cat_name=dataset_name,
            len_obj=len(obj_list),
            visualize=visualize,
        )
        env_ctor = HAND_ENV_CTORS[hand]
        env = VecEnv(
            obj_list,
            env_ctor(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)),
            home_path,
            cat_name=dataset_name,
        )
        env.seed(cfg.get('seed', args.seed))
        env.load_multi_articulated(obj_path_list)

        ht = get_cross_hand_spec(hand).hand_type
        cfgs[ht] = cfg
        cfg_names.append(used_cfg_path)
        envs[ht] = env
        if exist_cfg is None:
            exist_cfg = cfg # if their are multiple hands, we will use the first hand's env cfg for training config like total steps, eval_every_n etc. since they should be the same across different hands in our setting
        hand_order.append(ht)

    num_envs = len(obj_list)*len(envs)
    use_urdf = bool(model_config.get('use_urdf'))
    print(f"use urdf: {use_urdf}")


    unique_cfg_names = list(dict.fromkeys(cfg_names))
    saver = ConfigurationSaver(log_dir=args.home_path + "/raisimGymTorch/" + args.storedir + "/" + args.exp_name,
                            dir_name=args.log_name,
                             test_dir=False,
                             save_items=[ 
                                         *[cfg_path for cfg_path in unique_cfg_names]],
                             config_path=model_cfg_path)
    
    # Training
    reward_clip = -2.0
    n_steps_r = 130       
    total_steps_r = n_steps_r * len(obj_list) * len(hand_order)  # total steps per update (for fps/reporting)
    
    #--------------- load urdf feature and graph data tool  -------------------#
    if use_urdf:
        URDFParser = URDFGraphParser()
        for hand in selected_hands:
            ht = get_cross_hand_spec(hand).hand_type
            URDFParser.parse_urdf(get_cross_hand_urdf_path(home_path, hand), ht)
    else:
        URDFParser = None
    version=model_config.get("version","v4")
    if version in ["v2", "v3"]:
        print(f"[runner] Deprecated model version '{version}', auto-upgrade to 'v4'.")
        version = "v4"
        model_config["version"] = "v4"
    print(f"model version: {version}")
    GraphDataBuilder_ = GraphDataBuilder(URDFParser, data_dict, max_act_dim, version)
    
    #--------------- end urdf feature and graph data tool  -------------------#

    #--------------start RL network -------------------#

    actor_backbone = model_factory.build_model_from_config(model_config, GraphDataBuilder_, model_role='actor',data_dict=data_dict)
    sampler = NormalSampler(max_act_dim)
    sampler.seed(args.seed)
    distribution = ppo_module.MultivariateGaussianDiagonalCovariance(max_act_dim, num_envs, 1.0, sampler)
    actor_r = ppo_module.Actor(actor_backbone, distribution, str(device))

    critic_backbone = model_factory.build_model_from_config(model_config, GraphDataBuilder_, model_role='critic',  data_dict=data_dict)
    critic_r = ppo_module.Critic(critic_backbone, str(device))

    ppo_r = PPO.PPO(actor=actor_r,
                    critic=critic_r,
                    num_envs=num_envs,
                    num_transitions_per_env=n_steps_r,
                    num_learning_epochs=4,  
                    gamma=0.996,    
                    lam=0.95,   
                    num_mini_batches=4, 
                    device=str(device),
                    log_dir=saver.data_dir,
                    shuffle_batch= False , 
                    max_action_shape = max_act_dim,
                    max_obs_shape= max_obs_dim,
                    aux_invalid_axis_l2_coef=0.0,
                    )
    

    # load_param(actor_r,critic_r,ppo_r.optimizer, envs, weight_file,saver.data_dir)
        
        
    #--------------end RL network -------------------#
    
    
    #------------- init params ----------------#
    finger_weights_={}
    for hand_type, value in data_dict.items():
        finger_weights = np.ones((len(obj_list), value["keypoints_num"])).astype('float32') 
        tip_ids = value.get("fingertips_id")
        for idx in tip_ids:
            finger_weights[:, idx] *= 4.0
        finger_weights /= finger_weights.sum(axis=1).reshape(-1, 1)
        finger_weights *= value["finger_weights_scale"]
        finger_weights_[hand_type] = finger_weights
    
    params = {}
    for hand_type in hand_order:
        env = envs[hand_type]
        param = build_reset_params(env, hand_type, act_dim=data_dict[hand_type]["act_dim"], env_mode="floating", obj_full_path_list=obj_full_path_list)
        params[hand_type] = param

    #-------------- end init params ----------------#
    update_start = 0
    if args.re_iter:
        update_start = args.re_iter
        print(f"restart from iter: {update_start}")

    for update in range(update_start,args.num_iterations):
        start = time.time()
        ### Evaluate trained model visually (note always the first environment gets visualized)
        if update % exist_cfg['environment']['eval_every_n'] == 0 and update != 0:
            print("Visualizing and evaluating the current policy")
            torch.save({
                'actor_architecture_state_dict': actor_r.architecture.state_dict(),
                'actor_distribution_state_dict': actor_r.distribution.state_dict(),
                'critic_architecture_state_dict': critic_r.architecture.state_dict(),
                'optimizer_state_dict': ppo_r.optimizer.state_dict(),
            }, saver.data_dir + "/full_" + str(update) + '_r.pt')
            for _,env in envs.items():
                env.save_scaling(saver.data_dir, str(update))

        obs_r_all = []
        contain_non_aff_all = []
        rewards_r_sum_all = [] #list:dict
        handtypes_all = []
        obj_pose_reset_all = []
        for hand_type in hand_order:
            env = envs[hand_type]
            contain_non_aff, obs_new_r, rewards_r_sum, handtypes = reset_environment(env, hand_type,params[hand_type],data_dict[hand_type]["act_dim"])            
            obj_pose_reset_all.append(params[hand_type][5])
            obs_new_r = np.pad(obs_new_r, pad_width=((0,0),(0,max_obs_dim -obs_new_r.shape[1])),mode="constant",constant_values=0)
            obs_r_all.append(obs_new_r) 
            contain_non_aff_all.append(contain_non_aff) # [num_envs,1]
            rewards_r_sum_all.append(rewards_r_sum) # list:dict
            handtypes_all.append(handtypes)    #[numenv,1]
        handtypes_all = np.concatenate(handtypes_all,axis=0) # [numenv,1] 
        obs_new_r = np.concatenate(obs_r_all, axis=0)#[numenv,max_obs_dim]

        for step in range(n_steps_r):
            obs_r = obs_new_r  
            obs_r = obs_r[:].astype('float32')
            actions_all = ppo_r.act(obs_r, handtypes_all)  #->[numenv,max_act_dim]   #add_transition

            reward_all = []  #[num_envs,1]
            dones_all = []  #[num_envs,1]
            obs_r_all = []
            # for each hand
            for hand_id, hand_type in enumerate(hand_order):
                env = envs[hand_type]
                start_idx = sum(envs[ht].num_envs for ht in hand_order[:hand_id])
                end_idx = start_idx + env.num_envs
                action_dim = data_dict[hand_type]["act_dim"]
                action_r = actions_all[start_idx:end_idx, :action_dim]      
                
                action_l = np.zeros_like(action_r)
                reward_r, _, dones = env.step(action_r.astype('float32'), action_l.astype('float32')) 
                dones_all.append(dones)
                obs_new_r, dis_info = env.observe(
                    contain_non_aff_all[hand_id],
                    **observe_kwargs_for_hand_type(hand_type)
                    )

                obs_r_all.append(obs_new_r)
                
                #get reward for each env
                reward_r, rewards_r = get_reward_info(
                    reward_r=reward_r,
                    env=env,
                    hand_type=hand_type,  
                    dis_info=dis_info,
                    obs_new_r=obs_new_r,
                    cfg=cfgs[hand_type],  
                    finger_weights=finger_weights_[hand_type],  
                    device=device,
                    env_mode="floating"
                )
                reward_all.append(reward_r)
                
                # record reward for each each hand,each env
                rewards_r_sum = rewards_r_sum_all[hand_id]
                # for each env
                for i in range(len(rewards_r_sum)):
                    for k in rewards_r_sum[i].keys():
                        rewards_r_sum[i][k] = rewards_r_sum[i][k] + rewards_r[i][k]
                rewards_r_sum_all[hand_id] = rewards_r_sum 


            reward_all = np.concatenate(reward_all,axis=0)
            reward_all.clip(min=reward_clip)
            dones_all = np.concatenate(dones_all,axis=0)
            ppo_r.step(value_obs=obs_r, rews=reward_all, dones=dones_all, handtype=handtypes_all)
            obs_r_all_padded = [np.pad(o,((0,0),(0,max_obs_dim - o.shape[1]))) for o in obs_r_all]
            obs_new_r = np.concatenate(obs_r_all_padded,axis=0)

        #get all obs final
        obs_r_all_list = []
        handtypes_all = []
        for hand_id, hand_type in enumerate(hand_order):
            env = envs[hand_type]
            obs_r, _ = env.observe(contain_non_aff_all[hand_id], **observe_kwargs_for_hand_type(hand_type))
            obs_r_padded = np.pad(obs_r, ((0,0),(0,max_obs_dim - obs_r.shape[1])))
            obs_r_all_list.append(obs_r_padded)    
            handtypes = np.ones((env.num_envs,1),dtype='int32') * hand_type
            handtypes_all.append(handtypes)
        obs_r_all = np.concatenate(obs_r_all_list,axis=0)
        obs_r_all = obs_r_all[:, :].astype('float32')
        handtypes_all = np.concatenate(handtypes_all,axis=0)        
        
        ppo_r.update(value_obs=obs_r_all,handtypes = handtypes_all, log_this_iteration=update % 10 == 0, update=update)

        
        ####  ———————————————————————————————————— log ————————————————————————————————————
        actor_r.distribution.enforce_minimum_std((torch.ones(max_act_dim) * 0.2).to(device))
        end = time.time()
        
        all_keys = set()
        for rewards in rewards_r_sum_all:
            if rewards:  
                all_keys.update(rewards[0].keys())
        ave_reward = {k:0.0 for k in all_keys}
        
        for k in ave_reward.keys():
            for i in range(len(rewards_r_sum_all[0])):
                ave_reward[k] = ave_reward[k] + sum(rewards_r_sum_all[j][i][k] for j in range(len(rewards_r_sum_all)))
            ave_reward[k] = ave_reward[k] / (len(rewards_r_sum_all[0]) * n_steps_r * len(envs))
        if use_wandb:
            wandb.log(ave_reward,step=update)

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average reward: ", '{:0.10f}'.format(ave_reward['reward_sum'])))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_r / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps_r / (end - start)
                                                                        * exist_cfg['environment']['control_dt'])))
        print('{:<40}'.format("reward detail: "))
        for k in ave_reward.keys():
            if k != 'reward_sum':
                print('{:<40} {:>6}'.format(k, '{:0.10f}'.format(ave_reward[k])))

        aux_mean = getattr(ppo_r, "last_mean_aux_invalid_axis_l2", None)
        aux_coef = float(getattr(ppo_r, "aux_invalid_axis_l2_coef", 0.0))
        if aux_mean is None:
            print('{:<40} {:>6}'.format("aux_invalid_axis_l2 (mean): ", "N/A"))
        else:
            aux_mean_f = float(aux_mean)
            print('{:<40} {:>6}'.format("aux_invalid_axis_l2 (mean): ", '{:0.10f}'.format(aux_mean_f)))
            print('{:<40} {:>6}'.format("aux_invalid_axis_l2 * coef: ", '{:0.10f}'.format(aux_mean_f * aux_coef)))
        print('std: ')
        print(np.exp(actor_r.distribution.std.cpu().detach().numpy()))

        print('----------------------------------------------------\n')
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default=f"{TASK_PATH}/env/envs/cross_hand/model_yamls/model_gnn_cfg_L6_v4.yaml", help='Path to model config yaml')
    parser.add_argument('--layer', type=str, default='L6', help='layer token used in config filename (e.g. L4)')
    parser.add_argument('--version', type=str, default='v4', choices=['v4'], help='model version token used in config filename')
    parser.add_argument("-v","--visualize",type=bool,default=True)
    parser.add_argument("--use_wandb",type=bool,default=True)
    parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="debug" )
    parser.add_argument(
        "-hand",
        '--hand',
        nargs='+',
        required=True,
        help=f"hand type (train supported): ability, allegro, brainco, inspire, leap, schunk, shadow.",
    )
    
    parser.add_argument('-re_iter', type=int , default=None, help='restart training from this iter')
    parser.add_argument('--home_path',type=str,default=os.path.abspath(os.path.join(TASK_PATH, "../..")))
    parser.add_argument('-sd', '--storedir', type=str, default='exps')
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-itr', '--num_iterations', type=int, default=15000)
    parser.add_argument('-ln', '--log_name', type=str, default='single_obj')
    parser.add_argument("--eval_itr",type=int, default=100)
    # parser.add_argument('--weight_dir','-weight_dir', type=str, required=False, help='Directory containing model weights and config (for resume or evaluation)')
    parser.add_argument('--weight_id', type=str, default=None, help='Checkpoint id (e.g. 3000), if None use latest')
    parser.add_argument("--obj_num","-obj_num",type=int, default=-1)
    parser.add_argument("-dataset","--dataset",type=str,default="ycb",choices=["mixed_train","ycb"] ) 
    parser.add_argument("-obj_repeate_scale","--obj_repeate_scale",type=int,default=3 )

    main(parser)
