import gymnasium as gym
import torch 
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from link_generation.models.curiousity_models import GNN
from link_generation.predicting_signature.utils import get_node_features
from torch_geometric.data import Data, Batch
import numpy as np


class BraidFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int):
        '''
        A feature extractor for braids before inputing the features into an RL algorithm

        observation_space: the observation space of the environment
        features_dim: the output dimension of the feature extractor
        '''
        super().__init__(observation_space, features_dim)
        
        hidden_channels = 32
        num_heads = 8
        num_layers = 4
        self.braid_gnn = GNN(hidden_channels=hidden_channels, num_heads=num_heads, 
                             num_layers=num_layers, dropout=0,
                             classification=False, both=True, ohe_inverses=True, 
                             double_features=True)
        
        # Should I also experiment with added the difference between the signatures 
        # and a bool that says if the exact signature has been achieved?
        self.signature_net = nn.Sequential(
            nn.Linear(2, 2),  # 2 is for achieved and desired goals
            nn.ReLU(),
            nn.Linear(2, 2)
        )
        
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_channels*(2**(num_layers-1))*num_heads + 2, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        batch = Batch.from_data_list([self._create_braid_graph(braid_word) for braid_word in observations['observation']])
        braid_features = self.braid_gnn(batch)
        goal_features = self.goal_net(torch.cat([observations['achieved_goal'], 
                                                 observations['desired_goal']], dim=1))
        combined = torch.cat([braid_features, goal_features], dim=1)
        return self.combine_net(combined)
    
    def _create_braid_graph(self, braid_word) :
        # remove all adjacent inverse cancellations
        if len(braid_word) > 2 : 
            stack = []
            for generator in braid_word:
                if stack and stack[-1] + generator == 0:
                    stack.pop()
                else:
                    stack.append(generator)
            braid_word = stack

        # calculate edges for braid graph
        edges = []
        abs_braid_word = np.abs(braid_word)
        for i in range(len(braid_word)) :
            gen = abs_braid_word[i]
            left_out_edge_found = False
            right_out_edge_found = False
            for j in range(i+1,len(braid_word)) :
                if abs_braid_word[j] == gen :
                    edges.append([i, j]) 
                    edges.append([j, i])
                    left_out_edge_found = True
                    right_out_edge_found = True
                    break
                elif (abs_braid_word[j] == gen-1) and (not left_out_edge_found) :
                    edges.append([i, j])
                    edges.append([j, i])
                    left_out_edge_found = True
                elif (abs_braid_word[j] == gen+1) and (not right_out_edge_found) :
                    edges.append([i, j])
                    edges.append([j, i])
                    right_out_edge_found = True
                if left_out_edge_found and right_out_edge_found :
                    break

        node_features = get_node_features(braid_word, both=True, pos_neg=False, ohe_inverses=True)

        return Data(x=node_features, edge_index=torch.LongTensor(edges).t())