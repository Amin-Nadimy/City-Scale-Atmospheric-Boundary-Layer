# City-Scale Atmospheric Boundary Layer Modelling using NN4PDEs

This repository presents a novel approach to the discretisation and solution of the Navier-Stokes Equations using finite difference (FD), finite volume (FV), and finite element methods (FE). This method reformulates the discretised system of equations as a discrete convolution, analogous to a convolutional layer in a neural network, and solves it using a V-cycle geometric multigrid.

## Key Features:
- **Platform-Agnostic Code**: Runs seamlessly on CPUs, GPUs, and AI-optimised processors.
- **Neural Network Integration**: Easily integrates with trained neural networks for sub-grid-scale models, surrogate models, or physics-informed approaches.
- **Accelerated Development**: Leverages machine-learning libraries to speed up model development.

## Applications:
- The incompressible Navier-Stokes equations are modelled using NN4PDEs to simulate airflow over an urban area.
- It is a scalable method that runs in serial on CPU and GPU.
- It has been run in parallel on 2,4, and 8 GPUs on a local machine and a GPU cluster.

### Domain of the problem
- The domain is the South Kensington area in London, covering 5 km by 4 km.
- The resolution in this case is 1 m in x, y and z directions.

![Boundary Layer](https://github.com/Amin-Nadimy/City-Scale-Atmospheric-Boundary-Layer/blob/main/Documents/South_Kensington.jpg)


## Result
![Demo](https://github.com/Amin-Nadimy/City-Scale-Atmospheric-Boundary-Layer/blob/main/Documents/South_Kensington_demo.gif)

## Installation Guide

1. Download the package from the IRP repository on GitHub
2. Navigate into the directory which contains `environment.yml` and create the environment. Run the following command in the terminal.
```bash
conda env create -f environment.yml
```
3. Activate the environment created. 

4. You can install it as an independent `python` package by running the command
```bash
pip install .
```

## Contact and references
For more information please get in touch with me via:
- Email: amin.nadimy19@imperial.ac.uk
