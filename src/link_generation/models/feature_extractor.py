import gymnasium as gym
import torch 
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from link_generation.models.curiousity_models import GNN
from link_generation.predicting_signature.utils import get_node_features, braid_word_to_knot_geom_data
from torch_geometric.data import Data, Batch
import numpy as np


class GoalBraidFeaturesExtractor(BaseFeaturesExtractor):
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
                             double_features=True, return_features=True)
        
        # Should I also experiment with added the difference between the signatures 
        # and a bool that says if the exact signature has been achieved?
        # self.signature_net = nn.Sequential(
        #     nn.Linear(2, 2),  # 2 is for achieved and desired goals
        #     nn.ReLU(),
        #     nn.Linear(2, 2)
        # )
        
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_channels*(2**(num_layers-1))*num_heads + 1, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        batch = Batch.from_data_list([self._create_braid_graph(state) for state in observations['observation']]).to('cuda')
        braid_features = self.braid_gnn(batch)
        # goal_features = self.goal_net(torch.cat([observations['achieved_goal'], 
        #                                          observations['desired_goal']], dim=1))
        # print('braid features:', braid_features.shape)
        # print('target_signatures:', observations['desired_goal'].shape)
        try :
            combined = torch.cat([braid_features, observations['desired_goal']], dim=1)
        except RuntimeError :
            for state in observations['observation'] :
                print(state)
        return self.combine_net(combined)
    
    def _create_braid_graph(self, state) :
        braid_word = state[state != 0]

        # remove all adjacent inverse cancellations
        if len(braid_word) > 2 : 
            stack = []
            for generator in braid_word:
                if stack and stack[-1] + generator == 0:
                    stack.pop()
                else:
                    stack.append(generator)
            if len(stack) > 0 :
                braid_word = torch.tensor(stack)

        # calculate edges for braid graph
        edges = []
        abs_braid_word = torch.abs(braid_word)
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

        # bad news if all braid graphs in batch don't have any edges, so if there's no edges we give
        # each crossing a self edge
        if edges == [] :
            edges = [[i,i] for i in range(len(braid_word))]

        node_features = get_node_features(braid_word, both=True, pos_neg=False, ohe_inverses=True, laplacian=False, edges=edges)

        return Data(x=node_features, edge_index=torch.LongTensor(edges).t())
    

class ObsBraidFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, num_heads: int, 
                 num_layers: int, hidden_channels: int, braid_or_knot_graph: str, braid_index: int):
        '''
        A feature extractor for braids before inputing the features into an RL algorithm

        observation_space: the observation space of the environment
        features_dim: the output dimension of the feature extractor
        braid_or_knot_graph: whether to create the braid graph or knot graph
        '''
        features_dim = (2**(num_layers-1))*hidden_channels*num_heads
        super().__init__(observation_space, features_dim)
        
        self.braid_or_knot_graph = braid_or_knot_graph
        self.braid_index = braid_index
        self.gnn = GNN(hidden_channels=hidden_channels, num_heads=num_heads, 
                       num_layers=num_layers, dropout=0,
                       classification=False, both=False, ohe_inverses=True, 
                       double_features=True, laplacian=False, k=1, 
                       return_features=True, braid_index=braid_index)

    def forward(self, observations) -> torch.Tensor:
        if self.braid_or_knot_graph == 'braid' :
            batch = Batch.from_data_list([self._create_braid_graph(state) for state in observations]).to('cuda')
        elif self.braid_or_knot_graph == 'knot' :
            batch = Batch.from_data_list([self._create_knot_graph(state) for state in observations]).to('cuda')
        features = self.gnn(batch)
        return features
    
    def _create_braid_graph(self, state) :
        braid_word = self.process_state_to_braid_word(state)

        # calculate edges for braid graph
        edges = []
        abs_braid_word = torch.abs(braid_word)
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

        # bad news if all braid graphs in batch don't have any edges, so if there's no edges we give
        # each crossing a self edge
        if edges == [] :
            edges = [[i,i] for i in range(len(braid_word))]

        node_features = get_node_features(braid_word, both=False, pos_neg=False, ohe_inverses=True, 
                                          laplacian=False, k=1, edges=edges, braid_index=self.braid_index)

        return Data(x=node_features, edge_index=torch.LongTensor(edges).t())
    
    def _create_knot_graph(self, state) : 
        braid_word = self.process_state_to_braid_word(state)
        return braid_word_to_knot_geom_data(braid_word, y=0, both=False, pos_neg=False, ohe_inverses=True, 
                                            undirected=True, laplacian=False, k=1, braid_index=self.braid_index)
    
    def process_state_to_braid_word(self, state) : 
        braid_word = state[state != 0]

        # remove all adjacent inverse cancellations
        if len(braid_word) > 2 : 
            stack = []
            for generator in braid_word:
                if stack and stack[-1] + generator == 0:
                    stack.pop()
                else:
                    stack.append(generator)
            if len(stack) > 0 :
                braid_word = torch.tensor(stack)
        return braid_word