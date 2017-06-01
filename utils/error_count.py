"""
    G. Crowther - OptaSense Ltd. 24/05/2017
    
    Requirements:
        numpy
        
    Usage:
    
"""
import numpy as np

decimation = 4
threshold = 2 * np.pi
conversion = np.pi / (2 ** 15)

def error_job(y):
    tseries = conversion * np.diff(y, n = decimation)
    failures = tseries[tseries > threshold].shape[0]
    return failures / total_samples
    