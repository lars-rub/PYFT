# PYnamicFieldTheory

Welcome to the GPU accelerated CEDAR prototype PYnamicFieldTheory (PYFT)!

## How to run this

1. Insert your root path (folder that contains run_config.json) at the top of util.py, or leave it empty if you plan to run the main script from the root path
2. Run main.py

## Beware

This is a prototype! So beware of the following (and more):
- Some parameter values are hardcoded, e.g., the amplitude of gauss kernels.
- Most parameters and steps that exist in CEDAR are not supported/implemented.
- The GaussInput is currently very restricted, i.e., only square 2D matrices with the peak being located at the center of the matrix are supported.
- The code is not documented, commented, typed, etc.
- I am aware that the current file/folder structure is horrible :D

## TODO

- Extend this readme.
- More TODOs can be found at the bottom of main.py