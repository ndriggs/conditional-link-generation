from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
import torch
import numpy as np
from typing import Union
import json

def load_braid_words(train_test_or_val: str)
    with open(f'braid_{train_test_or_val}.txt', 'r') as f :
        braid_words = json.load(f)
    return braid_words

def remove_cancelations(train_test_or_val: str) :
    '''Removes all adjacent inverses from all braid words in the list
    including ones at the front and back
    e.g. [1,-3,3,2,1] becomes [1,2,1] and [2,1,-3,-2] becomes [1,-3]
    
    returns: list of lists (list of braid words)'''
    braid_words = load_braid_words(train_test_or_val)
    for i, braid_word in enumerate(braid_words) :
        stack = []
        # remove all adjacent inverses
        for generator in braid_word:
            if stack and stack[-1] + generator == 0:
                stack.pop()
            else:
                stack.append(generator)
        # remove all inverses at the front and back
        while stack[0] + stack[-1] == 0 :
            stack = stack[1:-1]

        braid_words[i] = stack
    
    return braid_words


def pad_braid_words(braid_words, pad_value=0):

    # get lengths of braid words
    lengths = [len(braid_word) for braid_word in braid_words]

    # convert braid words to torch tensors
    tensor_seqs = [torch.LongTensor(braid_word) for braid_word in braid_words]
    
    # Pad braid words to be the same length
    padded_seqs = pad_sequence(tensor_seqs, batch_first=True, padding_value=pad_value)
    
    # Create a tensor of sequence lengths
    lengths_tensor = torch.LongTensor(lengths)
    
    return padded_seqs, lengths_tensor

def braid_word_to_geom_data(braid_word, ohe_inverses: bool) :
    '''Converts a braid word (as a list of integers) to a torch_geometric.data.Data object
    containing node features and adjacency list information
    
    ohe_inverses: if True then one-hot-encodes all generators for node features. If False 
    then only one-hot-encodes the positive sigmas and uses -1 in the appropriate spot for 
    their inverses. e.g. if the braid index was 4 and ohe_inverses = True, then sigma_1 = 
    [1,0,0,0,0,0] and sigma_{-1} = [0,0,0,1,0,0]. If ohe_inverses = False then sigma_1 = 
    [1,0,0] and sigma_{-1} = [-1,0,0]'''

    # initialize edges with the edge between the first and last generator in the braid word
    edges = [[0, len(braid_word)-1], [len(braid_word)-1, 0]]
    # add an edge between all adjacent spots in the braid word
    for i in range(len(braid_word)-1) :
        edges.append([i, i+1])
        edges.append([i+1, i])

    # construct the node features, with each generator in the braid word as a node
    braid_word = np.array(braid_word)
    braid_index = 7
    if ohe_inverses :
        # move the negative sigmas to the spots after the positive ones
        # convert -1 to 7, -2 to 8, -3 to 9, etc. 
        braid_word = np.abs(braid_word) + (1-np.sign(braid_word))*((braid_index-1)/2)
        node_features = np.zeros((len(braid_word),(braid_index-1)*2))
        node_features[np.arange(len(braid_word)),braid_word-1] = 1
    else : 
        node_features = np.zeros((len(braid_word),braid_index-1))
        node_features[np.arange(len(braid_word)),np.abs(braid_word)-1] = np.sign(braid_word)




class BraidDataset(Dataset):
    def __init__(self, data: Union[np.ndarray, torch.Tensor], targets: np.ndarray, 
                 classification:bool, cnn:bool=False, 
                 seq_lengths=None, lk_matrix_size=21):

        if isinstance(data, torch.Tensor) : # braid word sequences
            self.data = data 
        else : # LK representation data 
            if cnn : # reshape to have dim (batch, channel, height, width)
                self.data = torch.from_numpy(data.reshape(-1,lk_matrix_size,lk_matrix_size)).float().unsqueeze(1)
            else :
                self.data = torch.from_numpy(data).float()
                
        if classification :
            self.targets = torch.from_numpy(targets).long()
        else : 
            self.targets = torch.from_numpy(targets).float()
        self.seq_lengths = seq_lengths # only use for transformer_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.seq_lengths is not None :
            return (self.data[idx], self.seq_lengths[idx]), self.targets[idx]
        else :
            return self.data[idx], self.targets[idx]
