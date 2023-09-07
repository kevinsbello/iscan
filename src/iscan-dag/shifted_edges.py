import os
# change to your r-base path
os.environ['R_HOME'] = "C:/Program Files/R/R-4.3.0/"

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import numpy as np
import inspect
from shifted_nodes import iSCAN

def py_find_par(X, shift_index, topo_order):
    rpy2.robjects.numpy2ri.activate()
    nr, nc = X.shape
    X = ro.r.matrix(X, nr, nc)

    # since R index begin at 1
    shift_index = [x + 1 for x in shift_index]
    shift_index = ro.r.matrix(shift_index, 1, len(shift_index))

    # topo_order need to plus 1 for R
    topo_order = [x + 1 for x in topo_order]
    topo_order = ro.r.matrix(topo_order, 1, len(topo_order))

    path = os.path.dirname(os.path.abspath(inspect.getfile(iSCAN)))
    r = ro.r
    r.source(os.path.join(path,"my_foci.R"))
    p = r.find_par(X, shift_index, topo_order)
    return p

def iSCAN_foci(X,Y,detect_shift_node,topo_order):
    x_adj=py_find_par(X.numpy(),detect_shift_node,topo_order)
    y_adj=py_find_par(Y.numpy(),detect_shift_node,topo_order)
    ddag = np.abs(x_adj-y_adj)
    return ddag