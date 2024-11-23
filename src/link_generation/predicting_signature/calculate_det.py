from sage.all import BraidGroup, Link, Integer
from link_generation.predicting_signature.utils import load_braid_words
import numpy as np

train_braids = load_braid_words('train')
val_braids = load_braid_words('val')
test_braids = load_braid_words('test')

B = BraidGroup(7)
for train_val_test, braid_words in zip(['train','val','test'],[train_braids, val_braids, test_braids]) :
    det_list = []
    for braid_word in braid_words :
        det_list.append(Link(B(braid_word)).determinant())
    np.save(f'det_{train_val_test}.npy', np.array(det_list))

