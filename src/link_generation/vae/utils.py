from sage.all import BraidGroup, Link, Integer
from link_generation.predicting_signature.utils import load_braid_words
import numpy as np

def get_knot_braids_sig_and_det(train_val_or_test: str, ipynb: bool = False) :
    # load the invariants
    if ipynb : 
        sig = np.load(f'../predicting_signature/y_{train_val_or_test}.npy')
        det = np.load(f'../predicting_signature/det_{train_val_or_test}.npy')
    else : 
        sig = np.load(f'src/link_generation/predicting_signature/y_{train_val_or_test}.npy')
        det = np.load(f'src/link_generation/predicting_signature/det_{train_val_or_test}.npy')

    # load the braid words
    braids = load_braid_words(train_val_or_test)

    # initiate the braid group, specific to our data from predicting signature
    B = BraidGroup(7)

    # filter out the knots from the links
    knot_idxs = []
    knot_braids = []
    for i, braid in enumerate(braids) :
        if Link(B([Integer(sig) for sig in braid])).is_knot() :
            knot_idxs.append(i)
            knot_braids.append(braid)

    # filter the signature and determinant of the knots
    return knot_braids, sig[knot_idxs], np.log(det[knot_idxs])
