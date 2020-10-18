# There should be no main() in this file!!! 
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1, 
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,) 
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

import numpy as np
import scipy.stats as scp

NORM = 1
#NORM=2**332
#NORM = 1e100
#EPS=1 / (NORM * NORM)
EPS = 1e-20

def pa(params, model):
    l = params['amin']
    r = params['amax']
    n = r - l + 1
    prob = np.array([1/n] * n)
    val = np.array(list(range(l, r + 1)))
    return prob, val
    
def pb(params, model):
    l = params['bmin']
    r = params['bmax']
    n = r - l + 1
    prob = np.array([1/n] * n)
    val = np.array(list(range(l, r + 1)))
    return prob, val
    
def pc_ab(a, b, params, model):
    la = params['amin']
    ra = params['amax']
    lb = params['bmin']
    rb = params['bmax']
    p1 = params['p1']
    p2 = params['p2']
    a = np.array(list(a))
    b = np.array(list(b))
    c = np.arange(0, ra + rb + 1, dtype=int)
    if model == 3:
        a2 = np.arange(0, ra + 1, dtype=int)
        prob = (
            (scp.binom.pmf(
                    a2.reshape(1, 1, 1, -1), 
                    a.reshape(1, -1, 1, 1), 
                    p1
                ) * NORM) * 
                (scp.binom.pmf(
                    (np.maximum(c.reshape(-1, 1, 1, 1) - a2.reshape(1, 1, 1, -1), 0)),
                    b.reshape(1, 1, -1, 1), 
                    p2
                ) * NORM)
        ).sum(axis=-1) / (NORM * NORM)
    else:
        prob = scp.poisson.pmf(c.reshape(-1, 1, 1), (a.reshape(1, -1, 1) * p1 + b.reshape(1, 1, -1) * p2))
    val = np.array([[[c for _ in b] for _ in a] for c in range(0, ra + rb + 1)])
    return prob, val

def pc(params, model):
    la = params['amin']
    ra = params['amax']
    lb = params['bmin']
    rb = params['bmax']
    prob, _ = pc_ab(range(la, ra + 1), range(lb, rb + 1), params, model)
    prob = np.sum(prob * NORM, axis=(1, 2)) / ((ra - la + 1) * (rb - lb + 1)) / NORM
    val = np.array(list(range(0, ra + rb + 1)))
    return prob, val
    
def pd_c(c, params, model):
    p3 = params['p3']
    ra = params['amax']
    rb = params['bmax']
    c = np.array(list(c))
    d = np.arange(0, 2 * (ra + rb) + 1)
    prob = scp.binom.pmf(d.reshape(-1, 1) - c.reshape(1, -1), c.reshape(1, -1), p3)
    val = np.array([[d for c1 in c] for d in range(0, 2 * (ra + rb) + 1)])
    return prob, val

def pd(params, model):
    p3 = params['p3']
    ra = params['amax']
    rb = params['bmax']
    prob_c, _ = pc(params, model)
    prob_c = prob_c.reshape(1, -1) * NORM
    prob, _ = pd_c(range(0, ra + rb + 1), params, model)
    prob = prob * NORM
    prob = np.sum(prob * prob_c, axis=-1) / (NORM * NORM)
    val = np.array(list(range(0, 2 * (ra + rb) + 1)))
    return prob, val
    
def pc_d(d, params, model):
    ra = params['amax']
    rb = params['bmax']
    c = np.arange(0, (ra + rb) + 1)
    d = np.array(list(d))
    prob_c, _ = pc(params, model)
    prob_d, _ = pd(params, model)
    prob_d_c, _ = pd_c(c, params, model)
    prob_c *= NORM
    prob_d_c = prob_d_c[d, :] * NORM
    prob_d = prob_d[d] * NORM
    prob = (prob_d_c.transpose(1, 0) * prob_c.reshape(-1, 1)) / (prob_d.reshape(1, -1) + EPS) / NORM
    val = np.array([[c1 for _ in d] for c1 in range(0, (ra + rb) + 1)])
    return prob, val
    
def generate(N, a, b, params, model):
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']
    if model == 3:
        c = scp.binom.rvs(a, p1, size=N) + scp.binom.rvs(b, p2, size=N)
    else:
        c = scp.poisson.rvs(a * p1 + b * p2, size=N)
    d = c + scp.binom.rvs(c, p3, size=N)
    return d
    
def pd_b(b, params, model):
    la = params['amin']
    ra = params['amax']
    lb = params['bmin']
    rb = params['bmax']
    
    a = np.arange(la, ra + 1)
    b = np.array(list(b))
    c = np.arange(0, (ra + rb) + 1)
    d = np.arange(0, 2 * (ra + rb) + 1)
    prob_d_c, _ = pd_c(c, params, model)
    prob_d_c = prob_d_c
    prob_d_c = prob_d_c.reshape(prob_d_c.shape[0], 1, prob_d_c.shape[1])
    
    prob_c_ab, _ = pc_ab(a, b, params, model)
    prob_c_b = prob_c_ab.transpose(2, 0, 1).sum(axis=-1) / (ra - la + 1)
    prob_c_b = prob_c_b.reshape(1, prob_c_b.shape[0], prob_c_b.shape[1])
    
    prob = (prob_c_b * prob_d_c).sum(axis=-1)
    val = np.array([[d1 for _ in b] for d1 in d])
    
    return prob, val
    
def pb_d(d, params, model):
    lb = params['bmin']
    rb = params['bmax']
    b = np.arange(lb, rb + 1, dtype=int)
    d = np.array(list(d))
    prob_d, _ = pd(params, model)
    prob_d = prob_d[d]
    
    prob_b = 1 / (rb - lb + 1)
    
    prob_d_b, _ = pd_b(b, params, model)
    prob_d_b = prob_d_b.transpose(1, 0)[:, d]
    
    prob = (prob_d_b * NORM) * (prob_b * NORM) / ((prob_d * NORM).reshape(1, -1) + EPS) / NORM
    val = np.array([[b1 for _ in d] for b1 in b])
    
    return prob, val

def pd_a(a, params, model):
    la = params['amin']
    ra = params['amax']
    lb = params['bmin']
    rb = params['bmax']
    
    a = np.array(list(a))
    b = np.arange(lb, rb + 1)
    c = np.arange(0, (ra + rb) + 1)
    d = np.arange(0, 2 * (ra + rb) + 1)
    
    prob_d_c, _ = pd_c(c, params, model)
    prob_d_c = prob_d_c.reshape(prob_d_c.shape[0], 1, prob_d_c.shape[1])
    
    prob_c_ab, _ = pc_ab(a, b, params, model)
    prob_c_a = prob_c_ab.transpose(1, 0, 2).sum(axis=-1) / (rb - lb + 1)
    prob_c_a = prob_c_a.reshape(1, *prob_c_a.shape)
    
    prob = (prob_d_c * prob_c_a).sum(axis=-1)
    val = np.array([[d1 for _ in a] for d1 in d])
    
    return prob, val
    
def pd_ab(a, b, params, model):
    la = params['amin']
    ra = params['amax']
    lb = params['bmin']
    rb = params['bmax']
    
    a = np.array(list(a))
    b = np.array(list(b))
    c = np.arange(0, (ra + rb) + 1)
    d = np.arange(0, 2 * (ra + rb) + 1)
    
    prob_d_c, _ = pd_c(c, params, model)
    prob_d_c = prob_d_c.reshape(prob_d_c.shape[0], 1, 1, prob_d_c.shape[1])
    
    prob_c_ab, _ = pc_ab(a, b, params, model)
    prob_c_ab = prob_c_ab.transpose(1, 2, 0)
    prob_c_ab = prob_c_ab.reshape(1, *prob_c_ab.shape)
    
    prob = (prob_d_c * prob_c_ab).sum(axis=-1)
    val = np.array([[[d1 for _ in b] for _ in a] for d1 in d])
    
    return prob, val
    
def pb_ad(a, d, params, model):
    lb = params['bmin']
    rb = params['bmax']
    a = np.array(list(a))
    b = np.arange(lb, rb + 1, dtype=int)
    d = np.array(list(d))
    prob_d_a, _ = pd_a(a, params, model)
    prob_d_a = prob_d_a[d, :].transpose(1, 0)
    prob_d_a = prob_d_a.reshape(1, *prob_d_a.shape)
    
    prob_b = 1 / (rb - lb + 1)
    
    prob_d_ab, _ = pd_ab(a, b, params, model)
    prob_d_ab = prob_d_ab.transpose(2, 1, 0)[:, :, d]
    
    prob = (prob_d_ab * NORM) * (prob_b * NORM) / ((prob_d_a * NORM) + EPS) / NORM
    val = np.array([[[b1 for _ in d] for _ in a] for b1 in b])
    
    return prob, val

