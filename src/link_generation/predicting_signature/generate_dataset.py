from sage.all import BraidGroup, Link, Integer
import numpy as np
import json

def lk_rep(n,k):
    M=np.zeros((n*(n-1)//2,n*(n-1)//2), dtype=np.float64)
    q=np.sqrt(2)
    t=np.pi
    for i in range(1,n):
        for j in range(i+1,n+1):
            if (k<i-1)or(j<k):
                M[index(n,i,j),index(n,i,j)]=1
            elif k==i-1:
                M[index(n,i-1,j),index(n,i,j)]= 1
                M[index(n,i,j),index(n,i,j)] = 1-q
            elif (k==i) and (k<j-1):
                M[index(n,i,i+1),index(n,i,j)] = t*q*(q - 1)
                M[index(n,i+1,j),index(n,i,j)] = q
            elif (k==i) and (k ==j-1):
                M[index(n,i,j),index(n,i,j)] = t*q*q
            elif (i<k) and (k<j - 1):
                M[index(n,i,j),index(n,i,j)] = 1
                M[index(n,k,k+1),index(n,i,j)] = t*q**(k - i)*(q - 1)**2
            elif (k==j-1):
                M[index(n,i,j-1),index(n,i,j)] = 1
                M[index(n,j-1,j),index(n,i,j)] = t*q**(j-i)*(q - 1)
            elif (k==j):
                M[index(n,i,j),index(n,i,j)]=1-q
                M[index(n,i,j+1),index(n,i,j)]=q
    return M

# used in the lk_rep function
def index(n,i,j):
    return int((i-1)*(n-i/2)+j-i-1)

generator_lk_matrices = {}
braid_index = 7
for sigma_i in range(-braid_index+1,braid_index):
    if np.sign(sigma_i) == -1:
        generator_lk_matrices[sigma_i]=np.linalg.inv(lk_rep(braid_index,np.abs(sigma_i)))
    elif np.sign(sigma_i) == 1:
        generator_lk_matrices[sigma_i]=lk_rep(braid_index,np.abs(sigma_i))


def braid_word_to_lk_rep(braid_word) :
    lk_rep = generator_lk_matrices[braid_word[0]]
    for gen in braid_word[1:] :
        lk_rep = lk_rep @ generator_lk_matrices[gen]
    return lk_rep

generators = [i for i in np.arange(-braid_index+1, braid_index) if i != 0]

B = BraidGroup(braid_index)

lk_reps = []
# for _ in range(2250) :
#     braid_word = np.random.choice(generators, size=45, replace=True)
#     lk_rep = generator_lk_matrices[braid_word[0]]
#     for i, gen in enumerate(braid_word[1:]) :
#         lk_rep = lk_rep @ generator_lk_matrices[gen]
#         sig = Link(B([Integer(sigma) for sigma in braid_word[:i+2]])).signature()
#         lk_reps.append(list(lk_rep.flatten()) + [sig])
braid_words = []
for braid_word_length in range(2, 46) :
    for _ in range(braid_word_length*110) :
        braid_word = np.random.choice(generators, size=braid_word_length, replace=True)
        lk_rep = braid_word_to_lk_rep(braid_word)
        sig = Link(B([Integer(sigma) for sigma in braid_word])).signature()
        lk_reps.append(list(lk_rep.flatten()) + [sig])
        braid_words.append(list(braid_word))
    for _ in range(braid_word_length*70) :
        sigma = np.random.choice(generators)
        braid_word = [sigma]
        insert_opp_prob = np.random.uniform(low=0,high=0.25)
        while len(braid_word) < braid_word_length :
            probabilities = np.zeros(len(generators))
            probabilities[(1+np.sign(sigma))*3:(1+np.sign(sigma))*3 + 6] = 0.1
            probabilities[generators.index(sigma)] = 0.5
            sigma = np.random.choice(generators, p=probabilities)
            braid_word.append(sigma)
            if (np.random.uniform() <= insert_opp_prob) and (len(braid_word) < braid_word_length) :
                braid_word.append(-1*np.sign(sigma)*np.random.randint(1,braid_index))
        lk_rep = braid_word_to_lk_rep(braid_word)
        sig = Link(B([Integer(sigma) for sigma in braid_word])).signature()
        lk_reps.append(list(lk_rep.flatten()) + [sig])
        braid_words.append(braid_word)

lk_reps = np.array(lk_reps)
np.save('lk_and_sig.npy', lk_reps)

# work around, courtesy of 
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NpEncoder, self).default(obj)

with open("braid_words.txt", 'w') as f:
    json.dump(braid_words, f, cls=NpEncoder)