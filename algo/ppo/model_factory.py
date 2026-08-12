import dexgrasp_zero.algo.ppo.module as ppo_module

def build_model_from_config(config, GraphDataBuilder, model_role='actor',  data_dict=None):
    model_type = config['model_type'].lower()
    print(f"*****************model_type: {model_type}*****************")
    if model_type == 'gnn':
        if model_role == 'actor':
            return ppo_module.GNN.from_config(config, GraphDataBuilder,  data_dict)
        else:
            return ppo_module.GNNCritic.from_config(config, GraphDataBuilder,  data_dict)
    raise ValueError(f"Unknown model_type: {model_type}")
