import numpy as np

def normalize_vol(x, norm=0.1):
    return x / (x.std()*np.sqrt(261)) * norm