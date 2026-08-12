import imp
import torch.nn as nn
import torch
import collections # Added this import
from .constants import (
    FingerIdType,
    HandType,
    NodeTypeSemantic,
    build_joint_semantic_map,
    build_node_to_joint_sequence,
)


FLEX = 0   # Flexion/Extension (bending)
ABD  = 1   # Abduction/Adduction (spreading)
ROT  = 2   # Axial Rotation (twisting)

class GraphDataBuilder:
    """
    A tool class to manage graph-structured data for different hand models.

    This class is responsible for:
    - Parsing URDF files to get joint and link information.
    - Pre-computing topological information for each hand type, such as parent-child
      relationships, joint depths, and sequential joint order for auto-regressive models.
    - Providing an interface to construct graph data structures (like adjacency matrices)
      and action tensors based on the pre-computed topologies.
    """
    def __init__(self, urdf_parser, data_dict, max_action_dim, version):
        self.version = version
        """
        Initializes the GraphDataBuilder.
        Args:
            urdf_parser: An object capable of parsing URDF files.
            data_dict (dict): A dictionary containing metadata for each hand type,
                              such as connection pairs and special node IDs.
            max_action_dim (int): The maximum dimension of the action space.
        """
        self.urdf_parser = urdf_parser
        self.data_dict = data_dict
        self.max_action_dim = max_action_dim
        self.use_urdf = urdf_parser is not None

        # Static feature dimensions (kept as attributes so models can adapt automatically).
        self.finger_id_dim = len(FingerIdType)
        self.node_semantic_dim = len(NodeTypeSemantic)
        self.urdf_node_dim = 0
        if self.use_urdf:
            # Determine URDF node feature dim from the first available hand.
            for ht in self.data_dict.keys():
                feat = self.urdf_parser.get_feature(ht)
                if feat is not None and "node_feature" in feat:
                    self.urdf_node_dim = int(feat["node_feature"].shape[-1])
                    break

        self.infoB_input_dim = self.finger_id_dim + self.node_semantic_dim
        if self.version == "v4" and self.use_urdf:
            self.infoB_input_dim += self.urdf_node_dim
        
        # Use self.data_dict consistently, no need for self.params = data_dict
        for handtype,data in self.data_dict.items():
            data["adj_matrix"] =  nn.Parameter(self._build_topology(handtype), requires_grad=True) 
        
        self.root_id = 0  # wrist node

        self.joint_semantic_map = build_joint_semantic_map()
        self.node_to_joint_sequence = build_node_to_joint_sequence()

        # Precompute static node features for all hand types
        self._precompute_static_node_features()
    
    def _get_handtypes(self):
        return self.data_dict.keys()
    


    def _build_topology(self,handtype):
        adj = torch.eye(self.data_dict[handtype]["keypoints_num"])
        connections = self.data_dict[handtype]["connection"]
        for i,j in connections:
            adj[i,j] = 1
            adj[i,i] = 1 # self-loop
            adj[j,i] = 1
        return adj

    def _precompute_static_node_features(self):
        for handtype in self._get_handtypes():
            num_nodes = self.data_dict[handtype]["keypoints_num"]
            connections = self.data_dict[handtype]["connection"]
            fingertips = set(self.data_dict[handtype]["fingertips_id"])
            wrist_id = self.root_id # Usually 0

            # Initialize
            node_semantic = torch.zeros(num_nodes, self.node_semantic_dim, dtype=torch.float)
            finger_id_one_hot = torch.zeros(num_nodes, self.finger_id_dim, dtype=torch.float) # 0 for wrist, 1-5 for fingers
            flex_valid = torch.zeros(num_nodes, dtype=torch.bool)
            abd_valid = torch.zeros(num_nodes, dtype=torch.bool)
            rot_valid = torch.zeros(num_nodes, dtype=torch.bool)

            # Build adjacency list for graph traversal (undirected for BFS)
            adj_list = collections.defaultdict(list)
            for u, v in connections:
                adj_list[u].append(v)
                adj_list[v].append(u)

            # Use pre-defined mappings for node_semantic and finger_id_one_hot
            finger_id_mapping = self.data_dict[handtype]["finger_id_mapping"]
            node_semantic_mapping = self.data_dict[handtype]["node_semantic_mapping"]

            for node_id in range(num_nodes):
                finger_id_one_hot[node_id, finger_id_mapping[node_id].value] = 1.0
                node_semantic[node_id, node_semantic_mapping[node_id].value] = 1.0


            # --- Calculate semantic axis validity ---
            node_joint_seq = self.node_to_joint_sequence[handtype]
            joint_semantic_map = self.joint_semantic_map[handtype]

            for node_id in range(num_nodes):
                # Wrist (node_id 0) might not directly control semantic axes in the same way fingers do
                # For now, we assume only finger joints contribute to these validities.
                # If wrist joints also have FLEX/ABD/ROT semantics, this logic needs extension.
                if node_id == wrist_id: 
                    continue 

                joint_names = node_joint_seq[node_id]
                for jn in joint_names:
                    if jn in joint_semantic_map and joint_semantic_map[jn] is not None:
                        axis_type, _ = joint_semantic_map[jn]
                        if axis_type == FLEX:
                            flex_valid[node_id] = True
                        elif axis_type == ABD:
                            abd_valid[node_id] = True
                        elif axis_type == ROT:
                            rot_valid[node_id] = True
            
            self.data_dict[handtype]["node_semantic"] = node_semantic
            self.data_dict[handtype]["finger_id_one_hot"] = finger_id_one_hot
            self.data_dict[handtype]["flex_valid"] = flex_valid
            self.data_dict[handtype]["abd_valid"] = abd_valid
            self.data_dict[handtype]["rot_valid"] = rot_valid


    def G_to_action(self, RT, joints, hand_type):
        """
        Convert graph node outputs (semantic FLEX/ABD/ROT) to physical action vector.
        
        Args:
            RT: [batch, 6] (wrist dof)
            joints: [batch, num_nodes, 3] where 3 = [FLEX, ABD, ROT]
            hand_type: str
        
        Returns:
            action: [batch, action_dim] (padded to max_action_dim)
        """
        action = RT  # Start with wrist dof [batch, 6]
        
        # Get hand-specific mappings
        joint_semantic = self.joint_semantic_map[hand_type]
        node_joint_seq = self.node_to_joint_sequence[hand_type]
        
        num_nodes = self.data_dict[hand_type]["keypoints_num"]
        
        for node_id in range(num_nodes):
            if node_id==0:
                continue
            # Get joint names controlled by this node
            joint_names = node_joint_seq[node_id]
            for jn in joint_names:
                if jn not in joint_semantic:
                    raise KeyError(f"Joint '{jn}' not found in joint_semantic_map for {hand_type}")
                axis_type, sign = joint_semantic[jn]
                # Pick the corresponding dimension from the node's 3D semantic output
                semantic_val = joints[:, node_id, axis_type:axis_type+1]  # [B,1]
                # Apply sign: +1 keep, -1 flip
                physical_action = sign * semantic_val
                action = torch.cat([action, physical_action], dim=1)
        
        # Pad to max dimension
        if action.size(1) < self.max_action_dim:
            pad = torch.zeros(
                action.size(0), 
                self.max_action_dim - action.size(1),
                device=action.device, 
                dtype=action.dtype
            )
            action = torch.cat([action, pad], dim=1)
        
        return action

    def vec_to_Graph(self, obs , handtype):
        """
        Args:
            obs: Observation tensor of shape [batch_size, obs_dim] or [num_env, batch_size, obs_dim].
            handtype: Enum value, see constants.HandType.

        Returns:
            Graph-structured dict:
                'nodes':  [num_env, batch_size, keypoint_num, d]
                'edges':  [num_env, batch_size, kp_num, kp_num]
                'global': [num_env, batch_size, d]
        """
        # dim check
        original_dim = obs.dim()
        if original_dim == 2:
            obs = obs.unsqueeze(0)  # [1, batch_size, d]
            
        # Slice observation fields
        obs_dim = self.data_dict[handtype]["obs_dim"]
        obs = obs[:,:,:obs_dim]  # keep the first obs_dim dimensions
        
        num_envs = obs.size(0) 
        batch_size = obs.size(1)
        device = obs.device
        global_feats, nodes, edges, infoB_nodes, infoC_nodes = self._extract_node_features(obs, handtype, num_envs, batch_size)

       
        result = {
            'nodes': nodes,  # [num_envs, batch_size, num_nodes, dim]
            'edges': edges,  # [num_envs, batch_size, num_nodes, num_nodes,dim]
            'global': global_feats
        }
        if self.version == "v2":
            result['infoC_nodes'] = infoC_nodes
        elif self.version in ["v3", "v4"]:
            result['infoB_nodes'] = infoB_nodes
            result['infoC_nodes'] = infoC_nodes
        
         ## Add URDF features (optional)
        if self.use_urdf:
            urdf_features = self.urdf_parser.get_feature(handtype)
            urdf_nodes = urdf_features['node_feature']
            urdf_nodes_expanded = urdf_nodes.unsqueeze(0).unsqueeze(0).expand(num_envs, batch_size, -1, -1).to(device)   # [num_nodes,22] -> [num_envs,batch_size,num_nodes,22]
            if self.version =="v4":
                result['infoB_nodes'] = torch.cat([infoB_nodes, urdf_nodes_expanded], dim=-1)
            else:
                raise ValueError(f"urdf version {self.version} unset")
            

        # Restore original shape
        if original_dim == 2:
            # Squeeze num_envs dimension
            result = {k: v.squeeze(0) for k,v in result.items()}
            
        return result

    def _extract_node_features(self, obs, hand_type, num_envs, batch_size):
        """
        Unified node feature extractor for all hand types.
        """
        device = obs.device
        d = self.data_dict[hand_type]
        
        # Get precomputed static node features
        node_semantic_raw = d["node_semantic"] # [num_nodes]
        finger_id_one_hot_raw = d["finger_id_one_hot"] # [num_nodes, 6]
        flex_valid_raw = d["flex_valid"] # [num_nodes]
        abd_valid_raw = d["abd_valid"] # [num_nodes]
        rot_valid_raw = d["rot_valid"] # [num_nodes]

        # Expand static features to match batch dimensions
        # [num_nodes,node_semantic_dim] -> [num_envs, batch_size, num_nodes, node_semantic_dim]
        node_semantic_expanded = node_semantic_raw.unsqueeze(0).unsqueeze(0).expand(num_envs, batch_size, -1, -1).to(device=device, dtype=obs.dtype)
        
        # [num_nodes, 6] -> [num_envs, batch_size, num_nodes, 6]
        finger_id_one_hot_expanded = finger_id_one_hot_raw.unsqueeze(0).unsqueeze(0).expand(num_envs, batch_size, -1, -1).to(device=device, dtype=obs.dtype)

        # [num_nodes] -> [num_envs, batch_size, num_nodes, 1]
        flex_valid_expanded = flex_valid_raw.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(num_envs, batch_size, -1, -1).to(device=device, dtype=obs.dtype)
        abd_valid_expanded = abd_valid_raw.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(num_envs, batch_size, -1, -1).to(device=device, dtype=obs.dtype)
        rot_valid_expanded = rot_valid_raw.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(num_envs, batch_size, -1, -1).to(device=device, dtype=obs.dtype)
        
        # === 1. Compute derived constants ===
        gc_dim = d["act_dim"]           # total joint DOF (including wrist)
        num_nodes = d["keypoints_num"]
        fingertips = set(d["fingertips_id"])
        wrist_id = 0
        
        # Number of contact points (non-fingertip, non-wrist nodes)
        num_contacts = num_nodes - len(fingertips) 

        # === 2. Pre-slice obs fields ===
        offset = 0
        target_center_dif = obs[:, :, offset:offset+3]; offset += 3
        wrist_ori_error = obs[:, :, offset:offset+3]; offset += 3
        
        # Finger joint angles: excludes wrist (first 6 of gc_dim)
        finger_joint_angles = obs[:, :, offset:offset + (gc_dim - 6)]; offset += (gc_dim - 6)
        
        # All joint torques (including wrist)
        torque_error = obs[:, :, offset:offset + gc_dim]; offset += gc_dim
        
        wrist_vel = obs[:, :, offset:offset+3]; offset += 3
        wrist_qvel = obs[:, :, offset:offset+3]; offset += 3
        
        contacts_r_af = obs[:, :, offset:offset+num_contacts] ; offset += num_contacts
        impulses_r_af = obs[:, :, offset:offset+num_contacts] ; offset += num_contacts
        
        obj_vel_in_wrist = obs[:, :, offset:offset+3]; offset += 3
        obj_qvel_in_wrist = obs[:, :, offset:offset+3]; offset += 3
        
        contacts_r_non_af = obs[:, :, offset:offset+num_contacts] ; offset += num_contacts
        impulses_r_non_af = obs[:, :, offset:offset+num_contacts] ; offset += num_contacts
        
        aff_dist = obs[:, :, offset:offset + 3*num_nodes]; offset += 3*num_nodes
        nonaff_dist = obs[:, :, offset:offset + 3*num_nodes]; offset += 3*num_nodes
        assert offset == obs.shape[-1], f"culculate error: offset {offset} != obs.shape[-1]:{obs.shape[-1]}"
        # === 3. Global features ===

        global_feats = torch.cat([
            target_center_dif,
            wrist_ori_error,
            wrist_vel,
            wrist_qvel,
            obj_vel_in_wrist,
            obj_qvel_in_wrist,
            torque_error[:, :, :6]  # wrist torques
        ], dim=-1)

        # === 4. Build node features ===
        node_data = []
        finger_joint_ptr = 0  # pointer into finger_joint_angles / torque_error[6:]
        contact_ptr = 0       # pointer into contact/impulse arrays

        # Get mappings
        node_joint_seq = self.node_to_joint_sequence[hand_type]
        joint_semantic = self.joint_semantic_map[hand_type]

        for node_id in range(num_nodes):
            # 4.1 Distance features
            dist_start = 3 * node_id
            node_aff = aff_dist[:, :, dist_start:dist_start+3]
            node_nonaff = nonaff_dist[:, :, dist_start:dist_start+3]

            if node_id == wrist_id:
                # Wrist node: no joint_angle/torque
                joint_angle = torch.zeros_like(node_aff)
                joint_torque = torch.zeros_like(node_aff)
                
                # Contact for wrist (always index 0 in contact arrays, if exists)
                c_af = contacts_r_af[:, :, 0:1]
                c_nonaf = contacts_r_non_af[:, :, 0:1]
                i_af = impulses_r_af[:, :, 0:1]
                i_nonaf = impulses_r_non_af[:, :, 0:1]
                contact_ptr = 1
                    
            elif node_id in fingertips:
                # Fingertip: no joint, no contact
                joint_angle = torch.zeros_like(node_aff)
                joint_torque = torch.zeros_like(node_aff)
                c_af = torch.zeros(num_envs, batch_size, 1, device=device)
                c_nonaf = torch.zeros(num_envs, batch_size, 1, device=device)
                i_af = torch.zeros(num_envs, batch_size, 1, device=device)
                i_nonaf = torch.zeros(num_envs, batch_size, 1, device=device)
                
            else:
                # Regular joint node
                joint_angle = torch.zeros(num_envs, batch_size, 3, device=device)  # [FLEX, ABD, ROT]
                joint_torque = torch.zeros(num_envs, batch_size, 3, device=device)
                joint_names = node_joint_seq[node_id]
                for jn in joint_names:
                    if jn not in joint_semantic:
                        raise ValueError(f"joint name: {jn} not in joint_semantic of hand {hand_type}")
                    
                    axis_type, sign = joint_semantic[jn]
                    # Get raw joint angle/torque (skip wrist 6dof)
                    q = finger_joint_angles[:, :, finger_joint_ptr: finger_joint_ptr+1]
                    tau = torque_error[:, :, 6 + finger_joint_ptr: 6 + finger_joint_ptr+1]
                    
                    # Convert to semantic space: semantic = sign * raw
                    joint_angle[:, :, axis_type:axis_type+1] = sign * q
                    joint_torque[:, :, axis_type:axis_type+1] = sign * tau
                    
                    finger_joint_ptr += 1

                # Contact features
                if contact_ptr < num_contacts:
                    c_af = contacts_r_af[:, :, contact_ptr:contact_ptr+1] if contacts_r_af is not None else torch.zeros(num_envs, batch_size, 1, device=device)
                    c_nonaf = contacts_r_non_af[:, :, contact_ptr:contact_ptr+1] if contacts_r_non_af is not None else torch.zeros(num_envs, batch_size, 1, device=device)
                    i_af = impulses_r_af[:, :, contact_ptr:contact_ptr+1] if impulses_r_af is not None else torch.zeros(num_envs, batch_size, 1, device=device)
                    i_nonaf = impulses_r_non_af[:, :, contact_ptr:contact_ptr+1] if impulses_r_non_af is not None else torch.zeros(num_envs, batch_size, 1, device=device)
                    contact_ptr += 1
                else:
                    c_af = torch.zeros(num_envs, batch_size, 1, device=device)
                    c_nonaf = torch.zeros(num_envs, batch_size, 1, device=device)
                    i_af = torch.zeros(num_envs, batch_size, 1, device=device)
                    i_nonaf = torch.zeros(num_envs, batch_size, 1, device=device)

            # 4.2 Concatenate node feature based on version
            current_node_features_list = [
                node_aff,
                node_nonaff,
                joint_angle,
                joint_torque,
                c_af,
                c_nonaf,
                i_af,
                i_nonaf
            ]

            # Prepare Info B features for current node
            # [num_envs, batch_size, 6], [num_envs, batch_size, 1]
            current_node_finger_id_one_hot = finger_id_one_hot_expanded[:, :, node_id, :]
            current_node_semantic = node_semantic_expanded[:, :, node_id, :]
            
            infoB_current_node_features_list = [
                current_node_finger_id_one_hot,
                current_node_semantic
            ]


            if self.version == "v0":
                features = torch.cat(current_node_features_list, dim=-1)
            elif self.version == "v2": # infoA + infoB in encoder, infoC separate
                features = torch.cat(current_node_features_list + infoB_current_node_features_list, dim=-1)
            elif self.version in ["v3", "v4"]: # infoA in encoder, infoB separate, infoC separate
                features = torch.cat(current_node_features_list, dim=-1)
            else:
                raise ValueError(f"Unknown version: {self.version}")
            
            node_data.append(features)

        nodes = torch.stack(node_data, dim=-2)  # [num_envs, batch_size, num_nodes, D_nodes]
        edges = self.data_dict[hand_type]["adj_matrix"].repeat(num_envs, batch_size, 1, 1).to(device=device, dtype=obs.dtype)
        
        # Consolidate infoB and infoC features for return
        # These will be [num_envs, batch_size, num_nodes, D_infoB] and [num_envs, batch_size, num_nodes, D_infoC]
        infoB_nodes_list = []
        infoC_nodes_list = []
        for n_id in range(num_nodes):
            infoB_node_concat = torch.cat([
                finger_id_one_hot_expanded[:,:,n_id,:], 
                node_semantic_expanded[:,:,n_id,:]
            ], dim=-1)
            infoB_nodes_list.append(infoB_node_concat)

            infoC_node_concat = torch.cat([
                flex_valid_expanded[:,:,n_id,:], 
                abd_valid_expanded[:,:,n_id,:], 
                rot_valid_expanded[:,:,n_id,:]
            ], dim=-1)
            infoC_nodes_list.append(infoC_node_concat)
        
        infoB_nodes = torch.stack(infoB_nodes_list, dim=-2) # [num_envs, batch_size, num_nodes, 7]
        infoC_nodes = torch.stack(infoC_nodes_list, dim=-2) # [num_envs, batch_size, num_nodes, 3]

        return global_feats, nodes, edges, infoB_nodes, infoC_nodes