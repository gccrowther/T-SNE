import numpy as np
from time import time
import matplotlib.pylab as plt
import utils

import os

from sklearn.manifold import t_sne

def norm(X):
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    return X

def save_tsv(data, fn):
    np.savetxt(fn, data, fmt='%.5f', delimiter = '\t')

def tsne(samples, data_root, prefix, initial_dims=30, perplexity=30, plot = True):
    if not os.path.exists('tsne'):
        os.mkdir('tsne')
    if not os.path.exists('plot'):
        os.mkdir('plot')
    
    figsize = (16, 16)
    pointsize = 16
    
    samples -= np.mean(samples, axis = 0)
    cov_x = np.dot(np.transpose(samples), samples)
    [eig_val, eig_vec] = np.linalg.eig(cov_x)
    
    eig_vec = eig_vec[:, eig_val.argsort()[::-1]]
    
    if initial_dims > len(eig_vec):
        intial_dims = len(eig_vec)
        
    eig_vec = eig_vec[:, :initial_dims]
    samples = np.dot(samples, eig_vec)
    
    sample_dim = len(samples[0])
    sample_count = len(samples)
    
    m_2d = t_sne.TSNE(n_components=2, perplexity=perplexity)
    X_2d = m_2d.fit_transform(samples)
    X_2d = norm(X_2d)
    save_tsv(X_2d, os.path.join(data_root, 'tsne/{}.{}.{}.2d.tsv'.format(prefix, initial_dims, perplexity)))
    
    m_3d = t_sne.TSNE(n_components=3, perplexity=perplexity)
    X_3d = m_3d.fit_transform(samples)
    X_3d = norm(X_3d)
    save_tsv(X_3d, os.path.join(data_root, 'tsne/{}.{}.{}.3d.tsv'.format(prefix, initial_dims, perplexity)))
    
    if plot:
        plt.figure(figsize=figsize)
        plt.scatter(X_2d[:,0], X_2d[:,1], edgecolor='', s=pointsize)
        plt.tight_layout()
        plt.savefig(os.path.join(data_root, 'plot/{}.{}.{}.png'.format(prefix, initial_dims, perplexity)))
        plt.close()

        plt.figure(figsize=figsize)
        plt.scatter(X_2d[:,0], X_2d[:,1], edgecolor='', s=pointsize, c=X_3d)
        plt.tight_layout()
        plt.savefig(os.path.join(data_root, 'plot/{}.{}.{}.png'.format(prefix, initial_dims, perplexity)))
        plt.close()
    
def job(params):
    start = time()
    tsne(params[0], params[1], params[2], initial_dims = params[3], perplexity = params[4])
    print('inital_dims = {0}, perplexity = {1}, {2} seconds'.format(params[3], params[4], time() - start))