import os

import numpy as np
import dask.array as da

from bokeh.io import curdoc
from bokeh.layouts import row, column
import bokeh.plotting as bk
from bokeh.models.widgets import PreText, Select
from bokeh.colors import RGB
from bokeh.models import HoverTool, ResetTool, BoxSelectTool, BoxZoomTool, SaveTool, WheelZoomTool, PanTool, TapTool, LassoSelectTool

def list_tsne(dir_name):
    files = os.listdir(dir_name)
    
    tsne_files = {}
    for file in files:
        key = 'id {0} - px {1}'.format(file.split('.')[1], file.split('.')[2])
        if key in tsne_files.keys():
            tsne_files[key].append(os.path.join(dir_name, file))
        else:
            tsne_files[key] = [os.path.join(dir_name, file)]
            
    return tsne_files

def load_tsne(tsne_list):
    
    channels = np.fromfile('channels.npy', dtype = np.int32)[20:]
    
    X2d = np.fromfile(tsne_list[0], dtype = np.float32, sep = '\t')
    X2d = X2d.reshape(X2d.shape[0] // 2, -1)
    X3d = np.fromfile(tsne_list[1], dtype = np.float32, sep = '\t')
    X3d = X3d.reshape(X3d.shape[0] // 3, -1)
    
    colors = [RGB(X3d[n, 0] * 255, X3d[n, 1] * 255, X3d[n, 2] * 255) for n in range(X3d.shape[0])]
    
    return dict(ix = np.arange(len(channels)),
                x = X2d[:, 0], 
                y = X2d[:, 1], 
                z = X3d, 
                c = colors,
                l = channels)
                
def load_truth(tsne_list, channel_list):
    channels = np.fromfile('channels.npy', dtype = np.int32)[20:]
    ix = np.arange(len(channels))
    
    loc = [i for i, j in zip(ix, channels) if j in channel_list]
    
    X2d = np.fromfile(tsne_list[0], dtype = np.float32, sep = '\t')
    X2d = X2d.reshape(X2d.shape[0] // 2, -1)
    X3d = np.fromfile(tsne_list[1], dtype = np.float32, sep = '\t')
    X3d = X3d.reshape(X3d.shape[0] // 3, -1)
    
    colors = [RGB(X3d[n, 0] * 255, X3d[n, 1] * 255, X3d[n, 2] * 255) for n in range(X3d.shape[0])]
    
    l_X2d = []
    l_X3d = []
    l_colors = []
    
    for i,l in enumerate(loc):
        l_X2d.append(X2d[l, :])
        l_X3d.append(X3d[l, :])
        l_colors.append(colors[l])
        
    l_X2dn = np.stack(l_X2d)
    l_X3dn = np.stack(l_X3d)
        
    return dict(ix = np.arange(len(channels)),
            x = l_X2dn[:, 0], 
            y = l_X2dn[:, 1], 
            z = l_X3dn, 
            c = l_colors,
            l = channel_list)

def update_fingerprint(index_list):
    
    xs = [np.arange(fingerprints.shape[1]) for index in index_list]
    mean_x = np.arange(fingerprints.shape[1])
    ys = []
    
    for i, index in enumerate(index_list):
        ys.append(fingerprints[index, :])
        
    mean = np.stack(ys).mean(axis = 0)
    return dict(xs = xs, ys = ys), dict(mean_x = mean_x, mean = mean)
    
def update_result(index_list):
    
    xs = [np.arange(result.shape[1]) for index in index_list]
    mean_x = np.arange(result.shape[1])
    ys = []
    
    for i, index in enumerate(index_list):
        ys.append(result[index, :])
        
    mean = np.stack(ys).mean(axis = 0)
    return dict(xs = xs, ys = ys), dict(mean_x = mean_x, mean = mean)
    
    
result = np.load('samples.npy')
result = result.reshape(len(result), -1)

fingerprints = np.load('fingerprints.npy')
fingerprints = fingerprints.reshape(len(fingerprints), -1)  
        
SOURCE_DIR = 'tsne'

tsne = list_tsne(SOURCE_DIR)

ticker = Select(value = None, options = [key for key in tsne.keys()])

source = bk.ColumnDataSource(data=dict(ix = [], x=[], y=[], z=[], c=[], l=[]))

leak_source = bk.ColumnDataSource(data=dict(ix = [], x=[], y=[], z=[], c=[], l=[]))

f_source = bk.ColumnDataSource(data = dict(xs = [], ys = []))
fmean_source = bk.ColumnDataSource(data = dict(mean_x = [], mean = []))
r_source = bk.ColumnDataSource(data = dict(xs = [], ys = []))
rmean_source = bk.ColumnDataSource(data = dict(mean_x = [], mean = []))

TOOLS = [HoverTool(), ResetTool(), BoxZoomTool(), TapTool(), BoxSelectTool(), LassoSelectTool()]

p = bk.figure(plot_width = 800, plot_height = 800, tools = TOOLS, background_fill_color = 'black')
p.grid.grid_line_color = 'black'
p.scatter('x', 'y', fill_color = 'c', source = source, line_alpha = 0)
p.scatter('x', 'y', fill_color = 'c', source = leak_source, line_color = 'red', size = 10)


q = bk.figure(plot_width = 800, plot_height = 400)
q.multi_line(xs = 'xs', ys = 'ys', source = f_source, color = 'blue', alpha = 0.05)
q.line(x = 'mean_x', y = 'mean', source = fmean_source, color = 'black')

r = bk.figure(plot_width = 800, plot_height = 400)
r.multi_line(xs = 'xs', ys = 'ys', source = r_source, color = 'blue', alpha = 0.05)
r.line(x = 'mean_x', y = 'mean', source = rmean_source, color = 'black')


def ticker_change(attrname, old, new):
    source.data = load_tsne(tsne[ticker.value])
    leak_source.data = load_truth(tsne[ticker.value], np.arange(278, 289))
    
def select_change(attrname, old, new):
    ids = source.selected['1d']['indices']
    if len(ids) > 0:
        f_source.data, fmean_source.data = update_fingerprint(ids)
        r_source.data, rmean_source.data = update_result(ids)
    else:
        pass
        
ticker.on_change('value', ticker_change)
source.on_change('selected', select_change)

layout = row(column(p, ticker), column(q, r))

curdoc().add_root(layout)
curdoc().title = 't-sne test'