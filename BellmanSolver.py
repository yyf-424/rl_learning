import numpy as np

def StateValSolver(gamma, P:np.ndarray, r:np.ndarray, v:np.ndarray, allo_error = 1e-6):
    if abs(gamma) >= 1 :
        gamma = 0.9

    if v.ndim == 1:
        v = v.reshape(-1, 1)
    else :
        return
    
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    else :
        return

    while True:
        v_ = r+ gamma*P@v
        if np.linalg.norm(v_-v) < allo_error:
            break
        v = v_

    return v_