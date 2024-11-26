from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from typing import Union
import json
import os

BRAID_INDEX = 7

def load_braid_words(train_test_or_val: str):
    if os.getcwd() == '/home/nattd/repos/link-generation/src/link_generation/predicting_signature' :
        with open(f'{train_test_or_val}_braids.txt', 'r') as f :
            braid_words = json.load(f)
    else :
        with open(f'src/link_generation/predicting_signature/{train_test_or_val}_braids.txt', 'r') as f :
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
        if len(stack) > 1 :
            while stack[0] + stack[-1] == 0 :
                stack = stack[1:-1]
        if len(stack) > 1 :
            braid_words[i] = stack
    
    return braid_words


def pad_braid_words(braid_words, pad_value=0):

    # get lengths of braid words
    lengths = [len(braid_word) for braid_word in braid_words]

    # convert the generators to tokens
    # sigma_{1} = 1, sigma_{6} = 6, simga_{-1} = 7, sigma_{-2} = 8, etc.
    braid_index = 7
    for i, braid_word in enumerate(braid_words) : 
        braid_word = np.array(braid_word)
        braid_word = np.abs(braid_word) + (1-np.sign(braid_word))*((braid_index-1)/2)
        braid_words[i] = braid_word

    # convert braid words to torch tensors
    tensor_seqs = [torch.LongTensor(braid_word) for braid_word in braid_words]
    
    # Pad braid words to be the same length
    padded_seqs = pad_sequence(tensor_seqs, batch_first=True, padding_value=pad_value)
    
    # Create a tensor of sequence lengths
    lengths_tensor = torch.LongTensor(lengths)
    
    return padded_seqs, lengths_tensor

def braid_word_to_circular_geom_data(braid_word, y, both: bool, ohe_inverses: bool) :
    '''Converts a braid word (as a list of integers) to a torch_geometric.data.Data object
    containing node features and adjacency list information. Puts braid word generators 
    in a circle each connected to its adjacent neighbors. 

    parameters: 
    braid_word: braid word as a list 
    y: the target invariant value(s)
    both: whether to include both pos_neg and ohe_inverses/not ohe_inverses all together 
    ohe_inverses: if True then one-hot-encodes all generators for node features. If False 
    then only one-hot-encodes the positive sigmas and uses -1 in the appropriate spot for 
    their inverses. e.g. if the braid index was 4 and ohe_inverses = True, then sigma_1 = 
    [1,0,0,0,0,0] and sigma_{-1} = [0,0,0,1,0,0]. If ohe_inverses = False then sigma_1 = 
    [1,0,0] and sigma_{-1} = [-1,0,0]'''

    # initialize edges with the edge between the first and last generator in the braid word
    edges = [[0, len(braid_word)-1], [len(braid_word)-1, 0]]
    # add an edge between all adjacent spots in the braid word
    if len(braid_word) > 2 :
        for i in range(len(braid_word)-1) :
            edges.append([i, i+1])
            edges.append([i+1, i])

    # construct the node features, with each generator in the braid word as a node
    node_features = get_node_features(braid_word, both=both, pos_neg=False, ohe_inverses=ohe_inverses)

    # now return the graph Data object
    return Data(x=node_features, edge_index=torch.LongTensor(edges).t(), y=torch.tensor(y))

def braid_word_to_knot_geom_data(braid_word, y, both: bool, pos_neg: bool, ohe_inverses: bool, undirected: bool,
                                 laplacian: bool, k: int, braid_index: int = 7) :
    '''
    Converts a braid word (as a list of integers) to a torch_geometric.data.Data object
    containing node features and adjacency list information. Each crossing has a directed edge to 
    the two* crossings that its strands go to (looking at the braid from top to bottom) and has two*
    incoming edges from the crossings its strands came from. When a generator is repeated or followed 
    by its inverse, there is only one edge since torch_geometric.nn.conv.TransformerConv doesn't 
    support edge_weights. Edges can circle from the bottom back to the top if that is where the strands
    lead in the braid closure. 

    parameters: 
    braid_word: braid word as a list of integers
    y: the target invariant value(s)
    ohe_inverses: if True then one-hot-encodes all generators for node features. If False 
    then only one-hot-encodes the positive sigmas and uses -1 in the appropriate spot for 
    their inverses. e.g. if the braid index was 4 and ohe_inverses = True, then sigma_1 = 
    [1,0,0,0,0,0] and sigma_1^{-1} = [0,0,0,1,0,0]. If ohe_inverses = False then sigma_1 = 
    [1,0,0] and sigma_1^{-1} = [-1,0,0]
    undirected: whether or not the graph should be directed or undirected
    '''
    edges = []
    abs_braid_word = np.abs(braid_word)
    for i in range(len(braid_word)) :
        gen = abs_braid_word[i]
        left_out_edge_found = False
        right_out_edge_found = False
        for j in range(1,len(braid_word)) :
            if abs_braid_word[(i+j)%len(braid_word)] == gen :
                edges.append([i, (i+j)%len(braid_word)]) 
                if undirected and ([(i+j)%len(braid_word), i] not in edges) :
                    edges.append([(i+j)%len(braid_word), i])
                left_out_edge_found = True
                right_out_edge_found = True
                break
            elif (abs_braid_word[(i+j)%len(braid_word)] == gen-1) and (not left_out_edge_found) :
                edges.append([i, (i+j)%len(braid_word)])
                if undirected and ([(i+j)%len(braid_word), i] not in edges) :
                    edges.append([(i+j)%len(braid_word), i])
                left_out_edge_found = True
            elif (abs_braid_word[(i+j)%len(braid_word)] == gen+1) and (not right_out_edge_found) :
                edges.append([i, (i+j)%len(braid_word)])
                if undirected and ([(i+j)%len(braid_word), i] not in edges) :
                    edges.append([(i+j)%len(braid_word), i])
                right_out_edge_found = True
            if left_out_edge_found and right_out_edge_found :
                break
    if not left_out_edge_found :
        edges.append([i, i])
    if (not right_out_edge_found) and ([i,i] not in edges) :
        edges.append([i, i])

    node_features = get_node_features(braid_word, both=both, pos_neg=pos_neg, ohe_inverses=ohe_inverses,
                                      laplacian=laplacian, k=k, edges=edges, braid_index=braid_index)

    return Data(x=node_features, edge_index=torch.LongTensor(edges).t(), y=torch.tensor(y))



def get_node_features(braid_word, both: bool, pos_neg: bool, ohe_inverses: bool, laplacian: bool,
                      k:int=0, edges=None, braid_index=7) : 
    '''
    braid_word: braid word as a list of integers
    both: whether or not to include both positive/negative crossing info and generator ohe-ing
    pos_neg: if true returns 2 node features. The positivity and negativity of the crossing
    ohe_inverses: if true returns each generator one hot encoded separately, else returns the 
    inverses as -1 in the spot of the corresponding positive generator
    laplacian: if true, uses only laplacian positional encodings as node features, otherwise adds them 
               to the other features if k is greater than 0
    k: includes the k first eigenvectors of the graph laplacian as node features, adds none if set to 0
    edges: edges of the knot graph
    '''
    if not isinstance(braid_word, torch.Tensor) :
        braid_word = torch.tensor(braid_word)
    if laplacian :
        node_features = get_laplacian_positional_ecoding(edges, k, len(braid_word))
    elif both : 
        if ohe_inverses :
            node_features = torch.zeros((len(braid_word),(braid_index-1)*2 + 2 + k))
            node_features[:,:k] = get_laplacian_positional_ecoding(edges, k, len(braid_word))
            node_features[:,k:-2] = get_ohe_inverses_node_features(braid_word, braid_index=braid_index)
            node_features[:,-2:] = get_pos_neg_node_features(braid_word)
        else : 
            node_features = torch.zeros((len(braid_word),braid_index+1+k))
            node_features[:,:k] = get_laplacian_positional_ecoding(edges, k, len(braid_word))
            node_features[:,k:-2] = get_not_ohe_inverses_node_features(braid_word, braid_index=braid_index)
            node_features[:,-2:] = get_pos_neg_node_features(braid_word)
    elif pos_neg : # only encode if the crossing is a positive or negative crossing
        node_features = torch.zeros((len(braid_word), 2+k))
        node_features[:,:k] = get_laplacian_positional_ecoding(edges, k, len(braid_word))
        node_features[:,k:] = get_pos_neg_node_features(braid_word)
    elif ohe_inverses :
        node_features = torch.zeros((len(braid_word),(braid_index-1)*2 + k))
        node_features[:,:k] = get_laplacian_positional_ecoding(edges, k, len(braid_word))
        node_features[:,k:] = get_ohe_inverses_node_features(braid_word, braid_index=braid_index)
    else : 
        node_features = torch.zeros((len(braid_word),braid_index-1+k))
        node_features[:,:k] = get_laplacian_positional_ecoding(edges, k, len(braid_word))
        node_features[:,k:] = get_not_ohe_inverses_node_features(braid_word, braid_index=braid_index)
    return node_features

def get_laplacian_positional_ecoding(edges, k, num_nodes) :
    ''''
    Gets the laplcian positional encoding of the graph, i.e. returns 
    the first k eigenvectors of the graph laplacian. 
    '''
    if (k == 0) :
        return torch.zeros((num_nodes, 0))
    A = np.zeros((num_nodes,num_nodes))
    for edge in edges :
        A[edge[1], edge[0]] = 1
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eig(L)
    if num_nodes < k :
        return torch.cat([torch.tensor(eigvecs[:,eigvals.argsort()[::-1][:k]]), torch.zeros((num_nodes, k-num_nodes))],dim=1)
    return torch.tensor(eigvecs[:,eigvals.argsort()[::-1][:k]])


def get_pos_neg_node_features(braid_word) :
    '''One hot encodes whether each "node" (braid letter) is a postive crossing or 
    negative crossing'''
    node_features = torch.zeros((len(braid_word), 2))
    node_features[:,0] = (torch.sign(braid_word) > 0).to(torch.float32)
    node_features[:,1] = (torch.sign(braid_word) < 0).to(torch.float32)
    return node_features

def get_ohe_inverses_node_features(braid_word, braid_index=7) :
    '''One hot encodes all sigmas and their inverses as distinct generators'''
    # move the negative sigmas to the spots after the positive ones
    # convert -1 to 7, -2 to 8, -3 to 9, etc. 
    braid_word = torch.abs(braid_word) + (1-torch.sign(braid_word))*((braid_index-1)/2)
    node_features = torch.zeros((len(braid_word),(braid_index-1)*2))
    node_features[torch.arange(len(braid_word)),braid_word.to(torch.int64)-1] = 1.0
    return node_features

def get_not_ohe_inverses_node_features(braid_word, braid_index=7) :
    '''Encodes each inverse as negative the one hot encoding of its corresponding positive generator'''
    node_features = torch.zeros((len(braid_word),braid_index-1))
    node_features[torch.arange(len(braid_word)),torch.abs(braid_word)-1] = torch.sign(braid_word).to(torch.float32)
    return node_features 

def get_circular_graph_dataloader(braid_words, targets, both:bool, ohe_inverses:bool, batch_size:int, shuffle:bool) :
    data_list = [braid_word_to_circular_geom_data(braid_word, y, both, ohe_inverses) for braid_word, y in zip(braid_words, targets)]
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

def get_knot_graph_dataloader(braid_words, targets, both:bool, pos_neg:bool, ohe_inverses:bool, undirected:bool, 
                              laplacian:bool, k:int, batch_size:int, shuffle:bool) :
    data_list = [
        braid_word_to_knot_geom_data(braid_word, y, both, pos_neg, ohe_inverses, undirected, laplacian, k) \
            for braid_word, y in zip(braid_words, targets)
    ]
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle, num_workers=79)


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


def get_experiment_name(args) :
    '''Returns a unique experiment name string based on hyperparameters'''
    model = args.model[:3]
    
    task = 'cls' if args.classification else 'reg'
    
    preproc = args.preprocessing 
    if preproc == 'remove_cancellations' :
        preproc = 'rm'
    elif preproc == 'do_nothing' :
        preproc = ''
    
    if args.model == 'mlp' : 
        return f'{model}_{preproc}_{task}_h{args.hidden_size}_d{args.dropout}'

    elif args.model == 'cnn' :
        ln = 'ln' if args.layer_norm else ''
        return f'{model}_{preproc}_{task}_k{args.kernel_size}_{ln}'

    elif args.model in ['transformer_encoder', 'reformer'] :
        return f'{model}_{preproc}_{task}_d{args.d_model}_h{args.nheads}_l{args.num_layers}'

    elif args.model == 'gnn' :
        ohe = 'ohe' if args.ohe_inverses else 'neg'
        return f'{model}_{preproc}_{task}_d{args.dropout}_l{args.num_layers}_{ohe}'
    
    elif args.model  == 'knot_gnn' :
        # ohe = 'pos_neg' if args.pos_neg else 'ohe'
        # undir = 'undir' if args.undirected else 'directed'
        # both = 'both' if args.both else 'single'
        return f'{model}_nheads{args.nheads}_nlayers{args.num_layers}'
