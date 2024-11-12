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

## Installation

### Prerequisites

Before proceeding, ensure that you have the following:

- **Python 3.10**: It is recommended to use Python 3.10 for compatibility with the required packages and libraries.

- **Conda (Preferred)**: Although not essential, it is recommended to use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for managing your Python environment and dependencies. Conda simplifies package management and helps avoid conflicts.

- **GPU with CUDA support**: A GPU that supports CUDA and has at least 20GB of VRAM. Ensure that your CUDA drivers are correctly installed and configured.

### Environment Setup

To set up the environment, run:

```bash
conda env create -f environment.yml
```

Alternatively, you can install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Contact and references
For more information please get in touch with me via:
- Email: amin.nadimy19@imperial.ac.uk
