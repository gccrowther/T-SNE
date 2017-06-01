# T-SNE for dimensionality reduction on Distributed Acoustic Sensing timeseries

This directory contains the output of a T-SNE processing chain and a bokeh server applet for visualising the results.

## Requirements

- Python 3.x
- >= Bokeh 0.12.3
- Numpy
- Dask

## Usage

Clone the directory to a local path, then from the command line run:
    'bokeh serve --show tsne_app.py'