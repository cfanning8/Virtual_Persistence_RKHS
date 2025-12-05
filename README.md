# Virtual Persistence RKHS Library

A minimal library for computing RKHS kernels and loss functions for virtual persistence diagrams, based on the mathematical framework described in the paper.

## Mathematical Framework

### Virtual Persistence Diagrams

For a finite metric pair $(X,d,A)$, the Grothendieck group $K(X,A)$ of persistence diagrams forms a discrete locally compact abelian (LCA) group. We identify $K(X,A) \cong \mathbb{Z}^{|X\setminus A|}$, where each element represents a virtual persistence diagram (VPD) as an integer-valued vector encoding multiplicities in a grid.

### Heat Kernel on Virtual Diagrams

The heat kernel is constructed via heat spectral multipliers on the Pontryagin dual torus $\mathbb{T}^{|X\setminus A|}$:

1. **Laplacian Symbol**: For angles $\theta \in \mathbb{T}^{|X\setminus A|}$, the Dirichlet symbol is
   $$
   \lambda(\theta) = \sum_{\{u,v\} \in E} w_{uv} \left(1 - \cos(\operatorname{dist}(\phi_\theta(u), \phi_\theta(v)))\right)
   $$
   where $\phi_\theta$ is the phase function on the quotient space $X/A$ and $w_{uv}$ are edge weights.

2. **Heat Measure**: The heat measure on the dual torus is defined as
   $$
   d\nu_t(\theta) = e^{-t\lambda(\theta)} d\mu(\theta)
   $$
   where $\mu$ is normalized Haar measure and $t > 0$ is the temperature parameter.

3. **Heat Kernel**: The translation-invariant kernel on $K(X,A)$ is
   $$
   k_t(\alpha, \beta) = \int_{\mathbb{T}^{|X\setminus A|}} \chi_\theta(\alpha - \beta) e^{-t\lambda(\theta)} d\mu(\theta)
   $$
   where $\chi_\theta(\alpha) = e^{i\langle\alpha, \theta\rangle}$ are characters on the virtual diagram group.

### Random Fourier Features

The random Fourier feature (RFF) approximation samples frequencies $\theta^{(1)}, \ldots, \theta^{(R)}$ from the normalized heat measure and defines the feature map:

$$
\Phi_{t,R}(\alpha) = \sqrt{\frac{\nu_t(\mathbb{T}^{|X\setminus A|})}{R}} \left(\cos\langle\alpha, \theta^{(r)}\rangle, \sin\langle\alpha, \theta^{(r)}\rangle\right)_{r=1}^R \in \mathbb{R}^{2R}
$$

This provides an unbiased approximation: $\mathbb{E}[\langle\Phi_{t,R}(\alpha), \Phi_{t,R}(\beta)\rangle] = k_t(\alpha, \beta)$.

### Topological Loss Function

For a prediction $\hat{y}$ and ground truth $y$, let $D_{\hat{y}}, D_y$ be their persistence diagrams, and set $\gamma = D_{\hat{y}} - D_y \in K(X,A)$. The topological loss is:

$$
\mathcal{L}_{\text{topo}}(\gamma) \approx \|\Phi_{t,R}(\gamma)\|^2
$$

This approximates the squared RKHS distance $\|k_t(\gamma, \cdot) - k_t(0, \cdot)\|_{\mathcal{H}_t}^2$.

## Library Structure

The library is **minimal** and focused solely on the RKHS kernel contribution:

- `src/kernels.py`: RKHS kernel implementation with `HeatRandomFeatures` class
- `src/loss.py`: Topological loss function `TopologicalRKHSLoss` that works with VPD vectors
- `src/vpd.py`: Utilities for converting persistence diagrams to VPD vectors
- `notebooks/persistent_homology/`: Examples showing how to use the library with gudhi

**Note**: The library does **not** compute persistence diagrams itself. Persistence diagram computation should be done using tools like gudhi (see notebooks for examples).

## Key Parameters

- `grid_size`: Grid dimension for VPD representation (default: 50, giving $N = 2500$ dimensional space)
- `n_components`: Number of random features $R$ (default: 256)
- `temperature`: Heat kernel temperature parameter $t$ (default: 0.2)
- `lambda_weights`: Optional Laplacian edge weights (default: uniform for $d=2$, required for $d > 2$)

## Usage

The library works with VPD vectors directly. First, compute persistence diagrams using gudhi or other tools, then convert them to VPD vectors:

```python
import numpy as np
import gudhi
from src import (
    HeatRandomFeatures,
    gudhi_persistence_to_vpd_vector,
    TopologicalRKHSLoss
)

# 1. Compute persistence diagrams using gudhi (see notebooks for examples)
persistence_pairs = ...  # From gudhi

# 2. Convert to VPD vectors
grid_size = 50
vpd_h0 = gudhi_persistence_to_vpd_vector(persistence_pairs, grid_size=grid_size, dimension=0)
vpd_h1 = gudhi_persistence_to_vpd_vector(persistence_pairs, grid_size=grid_size, dimension=1)
vpd = vpd_h0 + vpd_h1

# 3. Use RKHS kernel
rff = HeatRandomFeatures(
    input_dim=grid_size * grid_size,
    n_components=256,
    temperature=0.2,
    random_state=14
)
features = rff.transform(vpd.reshape(1, -1))

# 4. Compute topological loss from VPD difference
vpd_diff = vpd_pred - vpd_gt
loss_fn = TopologicalRKHSLoss(grid_size=grid_size, n_components=256, temperature=0.2)
loss = loss_fn(vpd_diff)
```

See `notebooks/persistent_homology/example_basic.ipynb` for a complete working example.

## References

See `main.tex` for the complete mathematical formulation and proofs.
