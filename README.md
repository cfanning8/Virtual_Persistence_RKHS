# Virtual Persistence RKHS Library

A minimal Python library for computing reproducing kernel Hilbert space (RKHS) kernels and loss functions for virtual persistence diagrams (VPDs). Provides translation-invariant kernels via heat spectral multipliers on discrete abelian groups.

## Mathematical Framework

Virtual persistence diagrams are elements of the Grothendieck group $K(X,A) \cong \mathbb{Z}^{|X\setminus A|}$, represented as integer-valued vectors encoding multiplicities on a grid. The library implements:

- **Heat Kernel**: Translation-invariant kernel $k_t(\alpha, \beta)$ on $K(X,A)$ constructed via heat spectral multipliers on the Pontryagin dual torus
- **Random Fourier Features (RFF)**: Efficient approximation $\Phi_{t,R}(\alpha)$ providing unbiased kernel estimates
- **Topological Loss**: Squared RKHS distance $\|\Phi_{t,R}(\gamma)\|^2$ for VPD differences $\gamma$

The heat kernel uses Dirichlet symbols on the dual torus with temperature parameter $t$, and RFF samples frequencies from the normalized heat measure.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src import TopologicalRKHSLoss, gudhi_persistence_to_vpd_vector

grid_size = 50
loss_fn = TopologicalRKHSLoss(grid_size=grid_size, random_state=14)

# Convert persistence diagram to VPD vector
vpd = gudhi_persistence_to_vpd_vector(persistence_pairs, grid_size=grid_size, dimension=0)

# Compute loss between VPDs
vpd_diff = vpd_pred - vpd_gt
loss = loss_fn(vpd_diff)
```

See `notebooks/persistent_homology/example.ipynb` for a complete working example.

## Library Structure

- `src/kernels.py`: Heat kernel and RFF implementation
- `src/loss.py`: Topological loss function for VPD vectors
- `src/vpd.py`: Utilities for converting persistence diagrams to VPD vectors

The library does not compute persistence diagrams. Use tools like [gudhi](https://gudhi.inria.fr/) for persistence computation.

## Key Parameters

- `grid_size`: Grid dimension for VPD representation (default: 50)
- `n_components`: Number of random features $R$ (default: 256)
- `temperature`: Heat kernel temperature parameter $t$ (default: 0.2)
- `lambda_weights`: Optional Laplacian edge weights (default: uniform for $d=2$)
