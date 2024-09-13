import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import snappy 
from sage.all import BraidGroup, Link


B = BraidGroup(4)
L = Link(B([1,-2,-2,3,2,1]))
L.plot()