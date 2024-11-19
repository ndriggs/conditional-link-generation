import torch
import numpy as np

def state_to_potholder_pytorch(s) :
    # converts from a flattened state of length n^2 - 2 to an n by n
    # matrix with the Reidemeister 1 moves added in
    n = np.sqrt(s.shape[1]+2)
    if n % 2 != 1 :
        raise ValueError('State must be of size n^2 - 2 where n is odd')
    n = int(n)
    P_flat = torch.zeros((s.shape[0], n**2), dtype=torch.float32)
    P_flat = P_flat.to('cuda')
    indexes = [i for i in range(n**2) if i not in [n-1, n**2-n]]
    P_flat[:,indexes] = s
    P = P_flat.view(s.shape[0],n,n)

    # P_flat_clone = P_flat.clone()
    # P_flat_clone[indexes] = s
    # P = P_flat_clone.reshape((n,n))
    return P

def state_to_potholder_numpy(s) :
    # converts from a flattened state of length n^2 - 2 to an n by n
    # matrix with the Reidemeister 1 moves added in
    n = np.sqrt(len(s)+2)
    if n % 2 != 1 :
        raise ValueError('State must be of size n^2 - 2 where n is odd')
    n = int(n)
    P_flat = np.zeros(n**2)
    indexes = [i for i in range(n**2) if i not in [n-1, n**2-n]]
    P_flat[indexes] = s
    P = P_flat.reshape((n,n))
    return P

def create_checkerboard(k):
    # creates checkerboard of the number of the shaded regions
    # of the potholder diagram
    if k % 2 == 1:
        raise ValueError("Input must be an even integer")

    board = np.zeros((k, k), dtype=int)
    counter = 1
    for i in range(k):
        for j in range(k):
            if (i + j) % 2 == 0:
                board[i, j] = counter
                counter += 1
    return board

def find_value_indices(checkerboard, value):
    # Find the position of the value
    positions = np.where(checkerboard == value)
    if positions[0].size == 0:
        print("Value not found in the checkerboard.")
        return None
    else:
        # Assuming the value is unique, return the first match's indices
        return positions[0][0], positions[1][0]
    

def get_neighbors(box_num, checkerboard) :
    # takes in box_num which is the number in the checkerboard
    # returns neighbors which is the numbers of the diagonally adjacent
    # regions minus 1 (their row/column index in the Goeritz matrix)
    k = checkerboard.shape[0]
    i, j = find_value_indices(checkerboard, box_num)
    neighbors = []
    for left_right in [-1,1] :
        for up_down in [-1,1] :
            if (i + up_down in np.arange(k)) and (j + left_right in np.arange(k)) :
                neighbors.append(checkerboard[i+up_down, j+left_right])
            elif 0 not in neighbors :
                neighbors.append(0)
    return neighbors

def get_zero_boardering_potholder_corners(region_index_i, region_index_j, k) :
    # accepts the checkerboard indices of a region that boarders the edge
    # k is checkerboard.shape[0] (or equivalently checkerboard.shape[1])
    # returns the indices for the potholder matrix where the region touches region 0
    if (region_index_i == 0) and (region_index_j == 0) :
        return [[0,0],[0,1],[1,0]]
    elif (region_index_i == k-1) and (region_index_j == k-1) :
        return [[k,k],[k-1,k],[k,k-1]]
    elif region_index_i == 0 :
        return [[0, region_index_j],[0, region_index_j+1]]
    elif region_index_j == 0 :
        return [[region_index_i, 0], [region_index_i+1, 0]]
    elif region_index_i == k - 1 :
        return [[k, region_index_j],[k, region_index_j+1]]
    elif region_index_j == k - 1 :
        return [[region_index_i, k], [region_index_i+1, k]]
    

def potholder_to_goeritz_pytorch(P) :
    batch_size = P.shape[0]
    # P = self.state_to_potholder(s)
    n = P.shape[1] # P is n x n, n is odd
    checkerboard = create_checkerboard(n-1) # checkerboard is k x k, k is even
    m = int(((n-1)**2)/2 + 1)
    G_tilda = torch.zeros((batch_size, m, m), dtype=torch.float32) # pre-Goeritz matrix is m x m, m is odd
    G_tilda = G_tilda.to('cuda')
    for i in range(1,m) : # for each region i
        i_neighbors = get_neighbors(i, checkerboard)
        i_checkerboard_index = find_value_indices(checkerboard, i)
        for j in i_neighbors : # for each adjacent region j
            if j == 0 :
                crossings = get_zero_boardering_potholder_corners(i_checkerboard_index[0], i_checkerboard_index[1], n-1)
                for cross in crossings :
                    potholder_i, potholder_j = cross
                    if (potholder_i + potholder_j) % 2 == 0 :
                        G_tilda[:,i,j] += -1 + 2*P[:,potholder_i,potholder_j]
                    else :
                        G_tilda[:,i,j] += 1 - 2*P[:,potholder_i,potholder_j]
                G_tilda[:,j,i] = G_tilda[:,i,j]
        else :
            j_checkerboard_index = find_value_indices(checkerboard, j)
            potholder_i = np.max([i_checkerboard_index[0], j_checkerboard_index[0]])
            potholder_j = np.max([i_checkerboard_index[1], j_checkerboard_index[1]])
            if (potholder_i + potholder_j) % 2 == 0 :
                G_tilda[:,i,j] = -1 + 2*P[:,potholder_i,potholder_j]
            else :
                G_tilda[:,i,j] = 1 - 2*P[:,potholder_i,potholder_j]
    G_tilda[:,range(G_tilda.shape[1]), range(G_tilda.shape[2])] = -torch.sum(G_tilda, axis=1)
    G = G_tilda[:,1:,1:]
    return G

def state_to_goeritz_numpy(s) :
    P = state_to_potholder_numpy(s)

    n = P.shape[0] # P is n x n, n is odd
    checkerboard = create_checkerboard(n-1) # checkerboard is k x k, k is even
    m = int(((n-1)**2)/2 + 1)
    G_tilda = np.zeros((m, m)) # pre-Goeritz matrix is m x m, m is odd

    for i in range(1,m) : # for each region i
        i_neighbors = get_neighbors(i, checkerboard)
        i_checkerboard_index = find_value_indices(checkerboard, i)
        for j in i_neighbors : # for each adjacent region j
            if j == 0 :
                crossings = get_zero_boardering_potholder_corners(i_checkerboard_index[0], i_checkerboard_index[1], n-1)
                for cross in crossings :
                    potholder_i, potholder_j = cross
                    if (potholder_i + potholder_j) % 2 == 0 :
                        G_tilda[i,j] += 1 if P[potholder_i,potholder_j] == 1 else -1
                    else :
                        G_tilda[i,j] += -1 if P[potholder_i,potholder_j] == 1 else 1
                G_tilda[j,i] = G_tilda[i,j]
            else :
                j_checkerboard_index = find_value_indices(checkerboard, j)
                potholder_i = np.max([i_checkerboard_index[0], j_checkerboard_index[0]])
                potholder_j = np.max([i_checkerboard_index[1], j_checkerboard_index[1]])
                if (potholder_i + potholder_j) % 2 == 0 :
                    G_tilda[i,j] = 1 if P[potholder_i,potholder_j] == 1 else -1
                else :
                    G_tilda[i,j] = -1 if P[potholder_i,potholder_j] == 1 else 1
    np.fill_diagonal(G_tilda, -np.sum(G_tilda, axis=1))
    G = G_tilda[1:,1:]
    return G

def state_to_signature(s) :
  G = state_to_goeritz_numpy(s)
  sign_G = np.sum(np.sign(np.linalg.eigvals(G)))
  mu = 0 # there are no type II crossings so the correction term is zero
  return sign_G - mu

def goeritz_to_signature(G) :
    return np.sum(np.sign(np.linalg.eigvals(G)))

def goeritz_to_det(G) :
    return round(np.abs(np.linalg.det(G)))

def get_ij_map(n) :
    # creates a dictionary mapping each non-Redimeister I move (i,j) 
    # location in the potholder matrix to its index in the flattened vector
    # n should be odd, n is the side length of the potholder matrix
    ij_map = {}
    for i in range(n) :
        for j in range(n) :
            idx = n*i + j
            if (idx != n-1) and (idx != n**2-n) :
                if idx > n**2-n :
                    idx -= 1
                if idx > n-1 :
                    idx -= 1
                ij_map[(i,j)] = idx

def get_potholder_graph_edges(n) :
    # n: the side length of the potholder matrix (should be odd)
    # returns the undirected graph edges for an n x n potholder
    # in the format necessary for a torch_geometric.data.Data object

    ij_map = get_ij_map(n)

    directed_edges = []

    for i in range(n) :
        for j in range(n) :

            # skip the top right and bottom left corners
            if (i == 0) and (j == n-1) :
                continue
            if (i == n-1) and (j == 0) :
                continue

            # easiest case - interior
            if (i>0) and (j>0) and (i<n-1) and (j<n-1) :
                if i % 2 == 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j-1)]])
                else :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j+1)]])
                if j % 2 == 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i-1,j)]])
                else :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i+1,j)]])

            # left side
            if j == 0 :
                if i % 2 == 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i+1,j)]])
                else :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j+1)]])
                if i > 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i-1,j)]])

            # top side
            if i == 0 :
                if j % 2 == 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j+1)]])
                else :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i+1,j)]])
                if j > 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j-1)]])

            # bottom side
            if i == n-1 :
                if j % 2 == 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i-1,j)]])
                else :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j+1)]])
                if (j > 1) and (j < n-1) :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j-1)]])
                elif j == 1 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i-1,j-1)]])

            # right side
            if j == n-1 :
                if i % 2 == 0 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i,j-1)]])
                else :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i+1,j)]])
                if (i > 1) and (i < n-1) :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i-1,j)]])
                elif i == 1 :
                    directed_edges.append([ij_map[(i,j)],ij_map[(i-1,j-1)]])

    undirected_edges = directed_edges
    for edge in directed_edges :
        if [edge[1], edge[0]] not in undirected_edges :
            undirected_edges.append([edge[1], edge[0]])

    return torch.LongTensor(undirected_edges).t()


# here's a simple example
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

n = 9
random_potholder = np.random.randint(2, size=n**2 - 2) # 9 x 9 potholder (minus the 2 Riedimiester I moves)
sig = state_to_signature(random_potholder)
edges = get_potholder_graph_edges(n)

# create the node features 
potholder = state_to_potholder_numpy(random_potholder)
node_features = []
for i in range(n) :
    for j in range(n) :
        # skip the top right and bottom left corners
        if (i == 0) and (j == n-1) :
            continue
        if (i == n-1) and (j == 0) :
            continue
        
        # one-hot encode positive/negative crossing information
        if (i+j) % 2 == 0 :
            node_features.append([potholder[i,j], 1-potholder[i,j]])
        else :
            node_features.append([1-potholder[i,j], potholder[i,j]])
node_features = torch.tensor(node_features)

# create pytorch_geometric.data.Data object
data = Data(x=node_features, edge_index=edges, y=torch.tensor(sig))

# create a pytorch_geometric.loader.DataLoader object
data_list = [data]
dataloader = DataLoader(data_list, batch_size=32, shuffle=True)
