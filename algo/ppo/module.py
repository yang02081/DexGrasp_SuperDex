import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
import torch_geometric.nn as tgnn
import torch
import torch.nn as nn
import numpy as np

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs,handtype):
        """
        Args:
            obs: Observations for all hands, shape [batchsize, max_dim].
            handtype: Shape [batchsize, 1].

        Logic:
            Split the batch by hand type, then feed each group into the network.

        Output:
            action: [batchsize, max_dim]; this must be a unified size.
        """
        obss = []
        handtypes = []
        handtype_num = len(np.unique(handtype))
        obs_size = obs.shape[0] // handtype_num  # number of envs per hand, matching len(obj_list) in main
        for i in range(handtype_num):  # split data, ensure obs and handtype are aligned one-to-one
            obss.append(obs[obs_size * i:obs_size * (i + 1)])
            handtypes.append(handtype[obs_size*i]) 
        actions = []
        for i in range(handtype_num):
            action_mean = self.architecture(obss[i],handtypes[i][0]).cpu().numpy()  # action: [batchsize, max_size]
            actions.append(action_mean)
        self.action_mean = np.concatenate(actions, axis=0) 
        actions, log_prob = self.distribution.sample(self.action_mean)  
        return actions, log_prob

    
    def evaluate(self,obs,actions,handtypes):
        """
        args:
            obs: [batch_size,max_obs_dim], device:cuda0
            handtypes: [batch_size,1], device:cpu
            actions: [batch_size,max_action_dim] ,device:cuda0
        """
        handtypes = handtypes.flatten().to(self.device)  
        
        unique_ht, inverse_indices, counts = torch.unique(
            handtypes,
            return_inverse=True,
            return_counts=True,
            sorted=False
        )
        
        sorted_indices = torch.argsort(inverse_indices)
        sorted_obs = obs[sorted_indices] 

        split_points = torch.cat([torch.tensor([0],device=self.device), counts.cumsum(0)])
        action_means = []
        aux_losses = []
        aux_weights = []
        for i in range(len(unique_ht)):
            batch_obs = sorted_obs[split_points[i]:split_points[i+1]]
            action_means.append(self.architecture(batch_obs, int(unique_ht[i].item())))

            # Optional auxiliary loss from the architecture (e.g., GNN v4 invalid-axis suppression).
            aux = getattr(self.architecture, "aux_invalid_axis_l2", None)
            if isinstance(aux, torch.Tensor):
                aux_losses.append(aux)
                aux_weights.append(int(batch_obs.shape[0]))
        
        self.action_mean = torch.cat(action_means)[torch.argsort(sorted_indices)]

        # Weighted mean across hand-type groups. Keep it as a Tensor so PPO can backprop.
        if len(aux_losses) > 0:
            total_w = sum(aux_weights)
            if total_w > 0:
                self.aux_invalid_axis_l2 = sum(a * w for a, w in zip(aux_losses, aux_weights)) / float(total_w)
            else:
                self.aux_invalid_axis_l2 = None
        else:
            self.aux_invalid_axis_l2 = None
        
        return self.distribution.evaluate(self.action_mean, actions)
        
        

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    def update(self):
        self.distribution.update()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
        self.device = device

    def predict(self, obs, handtype):
        """
            obs_shape: [batchsize, max_dim]
            handtype: [batchsize, 1]
        or
            obs_shape: [batchsize, num_env, max_dim]
            handtype: [batchsize, num_env, 1]
        """
        if len(obs.shape) == 2:
            obss = []
            handtypes = []
            handtype_num = len(np.unique(handtype))
            obs_size = obs.shape[0] // handtype_num
            for i in range(handtype_num):
                obss.append(obs[obs_size * i:obs_size * (i + 1)])
                handtypes.append(handtype[obs_size*i])
            actions = []
            for i in range(handtype_num):
                action_mean = self.architecture(obss[i],handtypes[i][0])  # action: [batchsize, max_size]
                actions.append(action_mean)
            action_mean = torch.cat(actions, dim=0)
        elif len(obs.shape) == 3:
            obss = []
            handtypes = []
            handtype_num = len(np.unique(handtype))
            obs_size = obs.shape[1] // handtype_num
            for i in range(handtype_num):
                obss.append(obs[:,obs_size * i:obs_size * (i + 1)])
                handtypes.append(handtype[0,obs_size*i])
            actions = []
            for i in range(handtype_num):
                action_mean = self.architecture(obss[i],handtypes[i][0])  # action: [batchsize, env, max_size]
                actions.append(action_mean)
            action_mean = torch.cat(actions, dim=1)
        return action_mean.detach()


    def evaluate(self,obs,handtypes):
        """
        args:
            obs: [batch_size,max_obs_dim]   cuda0
            handtypes: [batch_size,1]   cpu
            actions: [batch_size,max_action_dim]
        """
        handtypes = handtypes.flatten().to(self.device)
        unique_ht, inverse_indices, counts = torch.unique(
            handtypes,
            return_inverse=True,
            return_counts=True,
            sorted=False
        )
        
        sorted_indices = torch.argsort(inverse_indices)
        sorted_obs = obs[sorted_indices] 

        split_points = torch.cat([torch.tensor([0],device=self.device), counts.cumsum(0)])
        res = []
        for i in range(len(unique_ht)):
            batch_obs = sorted_obs[split_points[i]:split_points[i+1]]
            res.append(self.architecture(batch_obs, unique_ht[i].item()))
        
        res = torch.cat(res)[torch.argsort(sorted_indices)]
        
        return res


    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


# Sampler used to draw actions from a Gaussian distribution.
class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        """Update the cached standard deviation."""
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        """logits: mean of the distribution
           outputs: actions taken

           function: calculate the log probability of the actions taken
           returns: actions_log_prob (log probability of the actions), entropy
        """
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1) 
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    # Prevent numerical instability
    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach() 
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std  # Keep the standard deviation away from zero.



class GNN(nn.Module):
    def __init__(self, graph_data_builder_instance, node_dim=24, edge_dim=2, global_dim=26, gcn_layers=None, global_mlp=None, rt_head=None, joint_head=None,  data_dict=None,version="v0", infoB_config=None, infoC_config=None):
        super().__init__()
        self.graph_data_builder = graph_data_builder_instance
        self.version = version
        self.data_dict = data_dict if data_dict is not None else {}
        self.gcn_layers = nn.ModuleList()

        infoB_config = infoB_config or {}
        infoC_config = infoC_config or {}

        # Resolve InfoB input dim from the graph builder when available.
        resolved_infoB_input_dim = getattr(self.graph_data_builder, "infoB_input_dim", None)
        if resolved_infoB_input_dim is None:
            resolved_infoB_input_dim = infoB_config.get('input_dim')
        if resolved_infoB_input_dim is None:
            raise ValueError("infoB_config.input_dim is required when GraphDataBuilder does not expose infoB_input_dim")
        if infoB_config.get('input_dim') is not None and int(infoB_config.get('input_dim')) != int(resolved_infoB_input_dim):
            print(f"[GNN] Override infoB_config.input_dim {infoB_config.get('input_dim')} -> {resolved_infoB_input_dim}")
        infoB_config = dict(infoB_config)
        infoB_config['input_dim'] = int(resolved_infoB_input_dim)

        # InfoB Fusion MLP (for v3 + v4)
        self.use_infoB_fusion_mlp = infoB_config.get('use_fusion_mlp') 
        if self.version in ["v3", "v4"] and self.use_infoB_fusion_mlp:
            infoB_input_dim = infoB_config.get('input_dim') 
            infoB_fusion_mlp_cfg = infoB_config.get('fusion_mlp') 
            self.infoB_embedded_dim = infoB_fusion_mlp_cfg.get('embedded_dim') 
            self.infoB_fusion_mlp = build_mlp(
                infoB_input_dim,
                infoB_fusion_mlp_cfg.get('hidden_dims'),
                self.infoB_embedded_dim,
                activation=infoB_fusion_mlp_cfg.get('activation', 'relu')
            )
        
        current_gcn_input_dim = node_dim 

        for i, layer_cfg in enumerate(gcn_layers):
            actual_in_dim = current_gcn_input_dim
            if self.version in ["v3", "v4"] and i > 0:
                if  self.use_infoB_fusion_mlp:
                    actual_in_dim = current_gcn_input_dim + self.infoB_embedded_dim
                else:
                    actual_in_dim = current_gcn_input_dim + infoB_config.get('input_dim')
            self.gcn_layers.append(
                GCNLayer(
                    actual_in_dim,
                    layer_cfg['out_dim'], 
                    edge_dim,
                    activation=layer_cfg.get('activation', 'relu'), 
                    norm=layer_cfg.get('norm', 'layer') 
                )
            )
            current_gcn_input_dim = layer_cfg['out_dim'] 
        last_dim = current_gcn_input_dim

        
        self.global_mlp = build_mlp(global_dim, global_mlp['hidden_dims'], 256, global_mlp['activation']) 

        self.rt_head = build_mlp(256 + last_dim, rt_head['hidden_dims'], rt_head['out_dim'], rt_head['activation']) 

        infoC_input_dim = infoC_config.get('input_dim', 3) 
        joint_head_input_dim = last_dim
        if self.version in ["v2", "v3", "v4"]:
            joint_head_input_dim += infoC_input_dim # axis activate info from config
        
        self.joint_head = build_mlp(joint_head_input_dim, joint_head['hidden_dims'], joint_head['out_dim'], joint_head.get('activation', 'leaky_relu'))
        

        self.architecture = nn.Sequential(
            *self.gcn_layers,
            *self.global_mlp,
            *self.rt_head,
            *self.joint_head
        )


        self.init_weights()

    @classmethod
    def from_config(cls, config, graph_data_builder_instance,  data_dict=None):
        gnn_params = config.get('gnn_params', {})
        infoB_config = config.get('infoB_config', {})
        infoC_config = config.get('infoC_config', {})
        return cls(
            graph_data_builder_instance,
            node_dim=gnn_params.get('node_dim', 24),
            edge_dim=gnn_params.get('edge_dim', 2),
            global_dim=gnn_params.get('global_dim', 26),
            gcn_layers=gnn_params.get('gcn_layers'),
            global_mlp=gnn_params.get('global_mlp'),
            rt_head=gnn_params.get('rt_head'),
            joint_head=gnn_params.get('joint_head'),
            data_dict=data_dict,
            version=config.get('version',"v0"),
            infoB_config=infoB_config, # Pass InfoB config
            infoC_config=infoC_config  # Pass InfoC config
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, obs,handtype):
        """
        args: 
            Single-hand observations.
            obs: [batch, max_dim] or [num_env, batch_size, max_dim].
            handtype: HandType.ALLEGRO.value / HandType.SHADOW.value, etc.
        return:
            action: [batch, max_size] (padded to the maximum action size)
        """
        handtype_val = handtype.item() if isinstance(handtype, torch.Tensor) else handtype
        
        # Split the raw observation according to data_dict
        original_obs_dim = self.data_dict[handtype_val]["obs_dim"]
        raw_obs = obs[:, :original_obs_dim]  # raw observation part
        

        # Build the base graph data using raw_obs
        graph_data = self.graph_data_builder.vec_to_Graph(raw_obs, handtype) 

        x, adj, global_feat= graph_data['nodes'], graph_data['edges'], graph_data['global']


    
        if self.version in ["v3", "v4"] :
            if  self.use_infoB_fusion_mlp:
                infoB_nodes = graph_data['infoB_nodes']
                fused_infoB = self.infoB_fusion_mlp(infoB_nodes)
            else:
                fused_infoB = graph_data['infoB_nodes']
        # Apply GCN layers
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            if self.version in ["v3", "v4"] and i < len(self.gcn_layers) - 1 : 
                x = torch.cat([x, fused_infoB], dim=-1)  # Concatenate for the next GCN layer input

        
        global_emb = self.global_mlp(global_feat)
        root_feat = x[:, 0, :] 
        rt_input = torch.cat([root_feat, global_emb], dim=1)
        rt = self.rt_head(rt_input)
        joint_head_input = x


        infoC_nodes = None
        if self.version == "v4":
            infoC_nodes = graph_data.get('infoC_nodes')
            if infoC_nodes is not None:
                joint_head_input = torch.cat([joint_head_input, infoC_nodes], dim=-1)
            


        joints = self.joint_head(joint_head_input) 

        # Optional auxiliary regularization term:
        # penalize outputs on invalid semantic axes using the activation mask (infoC_nodes).
        # Stored on the module so PPO can optionally consume it without changing action APIs.
        self.aux_invalid_axis_l2 = None
        if self.version == "v4" and infoC_nodes is not None:
            # infoC_nodes: [B, N, 3] with 0/1 for (FLEX, ABD, ROT) validity
            invalid = 1.0 - infoC_nodes
            invalid_sum = invalid.sum()
            # Normalized mean-squared magnitude on invalid channels
            l2 = (joints * invalid).pow(2).sum() / (invalid_sum + 1e-6)
            self.aux_invalid_axis_l2 = torch.where(invalid_sum > 0, l2, joints.new_zeros(()))
        
        action = self.graph_data_builder.G_to_action(rt, joints, handtype)
        return action


class GNNCritic(nn.Module):
    def __init__(self, GraphDataBuilder, node_dim=24, edge_dim=2, global_dim=26, gcn_layers=None, global_mlp=None, value_head=None,  data_dict=None):
        super().__init__()
        self.graph_data_builder = GraphDataBuilder
        self.data_dict = data_dict if data_dict is not None else {}

        self.gcn_layers = nn.ModuleList()
        last_dim = node_dim
        if gcn_layers is None:
            gcn_layers = [
                {'out_dim': 128, 'activation': 'relu', 'norm': 'layer'},
                {'out_dim': 256, 'activation': 'relu', 'norm': 'layer'}
            ]
        for layer_cfg in gcn_layers:
            self.gcn_layers.append(
                GCNLayer(
                    last_dim,
                    layer_cfg.get('out_dim', 128),
                    edge_dim,
                    activation=layer_cfg.get('activation', 'relu'),
                    norm=layer_cfg.get('norm', 'layer')
                )
            )
            last_dim = layer_cfg.get('out_dim', 128)
        if global_mlp is None:
            global_mlp = {'hidden_dims': [256], 'activation': 'relu'}
        self.global_mlp = build_mlp(global_dim, global_mlp['hidden_dims'], 256, global_mlp.get('activation', 'relu'))
        if value_head is None:
            value_head = {'hidden_dims': [128], 'activation': 'relu', 'out_dim': 1}
        self.value_head = build_mlp(256 + last_dim, value_head['hidden_dims'], value_head['out_dim'], value_head.get('activation', 'relu'))
        self.init_weights()

    @classmethod
    def from_config(cls, config, GraphDataBuilder,  data_dict=None):
        critic_gnn_params = config.get('critic_gnn_params', {})
        gnn_params = config.get('gnn_params', {}) # Fallback to gnn_params if not in critic_gnn_params
        
        return cls(
            GraphDataBuilder,
            node_dim=critic_gnn_params.get('node_dim', gnn_params.get('node_dim', 24)),
            edge_dim=critic_gnn_params.get('edge_dim', gnn_params.get('edge_dim', 2)),
            global_dim=critic_gnn_params.get('global_dim', gnn_params.get('global_dim', 26)),
            gcn_layers=critic_gnn_params.get('gcn_layers', gnn_params.get('gcn_layers')),
            global_mlp=critic_gnn_params.get('global_mlp', gnn_params.get('global_mlp')),
            value_head=critic_gnn_params.get('value_head'),
            data_dict=data_dict,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs, handtypes):
        """
        args:
            obs: [batch, max_dim] or [num_env, batch_size, max_dim]
            handtypes: HandType.ALLEGRO.value / HandType.SHADOW.value, etc.
        """
        # Graph construction (same as Actor)
        batch_size = obs.shape[0]
        handtype_val = handtypes.item() if isinstance(handtypes, torch.Tensor) else handtypes

        original_obs_dim = self.data_dict[handtype_val]["obs_dim"]
        if len(obs.shape)==3:
            raw_obs = obs[:,:,:original_obs_dim]
        else:
            raw_obs = obs[:, :original_obs_dim]

        graph_data = self.graph_data_builder.vec_to_Graph(raw_obs,handtypes)
        x, adj, global_feat = graph_data['nodes'], graph_data['edges'], graph_data['global']
        

        # If 4D, flatten it
        len_shape = len(x.shape)
        original_shape = x.shape

        # Apply the first GCN layer
        if len_shape == 4:
            x = x.view(-1, *original_shape[-2:])
            global_feat = global_feat.view(-1, global_feat.shape[-1])
            
        if len(adj.shape) == 4:
            adj = adj.view(-1, *adj.shape[-2:])  
        elif len(adj.shape) ==5:
            adj = adj.view(-1, *adj.shape[-3:])
        x = self.gcn_layers[0](x, adj) 
        # Apply the remaining GCN layers
        for gcn in list(self.gcn_layers)[1:]:
            x = gcn(x, adj) 


        global_emb = self.global_mlp(global_feat)
        root_feat = x[:, 0, :]  # Root node feature is always the first node.
        value_input = torch.cat([root_feat, global_emb], dim=1)
        value = self.value_head(value_input)
        if len_shape == 4:
            value = value.view(*original_shape[:-2], -1)
        return value

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, activation='relu', norm='layer'):
        super().__init__()
        self.edge_dim = edge_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if norm == 'layer' else nn.Identity()
        self.activation = get_activation(activation)
    def forward(self, x, adj):
        message = torch.bmm(adj, x)
        out = self.linear(message)
        out = self.norm(out)
        return self.activation(out)

def build_mlp(input_dim, hidden_dims, output_dim, activation='relu'):
    layers = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(get_activation(activation))
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        return nn.ReLU()
