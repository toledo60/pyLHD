# pyLHD: Latin Hypercube Designs for Python

pyLHD is a python implementation for something in between the R packages [LHD](https://cran.r-project.org/web/packages/LHD/index.html)
and [DiceDesign](https://cran.r-project.org/web/packages/DiceDesign/index.html).

This package is primarily designed for educational purposes and may not be highly optimized. For more efficient and thoroughly tested functions, I recommend utilizing the `scipy.qmc` module 


## Installation

Currently `pyLHD` can be installed from Github

```
pip install git+https://github.com/toledo60/pyLHD.git
```

The main dependency for `pyLHD` is [NumPy](https://numpy.org/) and currently tested on Python 3.9+

## Overview

With `pyLHD` you can generate the following designs:

- `random_lhd`: Generate a random Latin hypercube design
- `GLPdesign`: Generate a good lattice point design
- `OLHD_Butler01`: Orthogonal Latin hypercube design. Based on the construction method of Butler (2001)
- `OLHD_Cioppa07`: Orthogonal Latin hypercube design. Based on the construction method of Cioppa and Lucas (2007)
- `OLHD_Lin09`: Orthogonal Latin hypercube design. Based on the construction method of Lin et al. (2009)
- `OLHD_Sun10`: Orthogonal Latin hypercube design. Based on the construction method of Sun et al. (2010)
- `OLHD_Ye98`: Orthogonal Latin hypercube design. Based on the construction method of Ye (1998)

Calculate design properties such as:

- `AvgAbsCor`: Calculate the average absolute correlation
- `inter_site`: Calculate the Inter-site Distance (rectangular/Euclidean) between the *ith* and *jth* row
- `coverage`: Calculate the coverage measure
- `discrepancy`: Calculate the discrepancy of a given sample
- `MaxAbsCor`: Calculate the maximum absolute correlation
- `MaxProCriterion`: Calculate the maximum projection criterion
- `mesh_ratio`: Calculate the meshratio criterion
- `minimax`: Calculate the minimax criterion
- `phi_p`: Calculate the phi_p criterion

Optimization algorithms to improve LHD's based on desired criteria:

- `LA_LHD`: Lioness Algorithm for Latin hypercube design
- `SA_LHD`: Simulated Annealing for Latin hypercube design

Other helper functions include:

- `distance_matrix`: Calculates the distance matrix  (row pairwise) from a specified distance measure
- `swap_elements`: Swap two random elements from a specified column or row in a matrix
- `OA2LHD`: Convert an orthogonal array (OA) into a Latin hypercube design (LHD)
- `scale`: Scales a sample in the hypercube to be within `[lower_bounds, upper_bounds]`
- `williams_transform`: Apply Williams transformation to a specified design


For further details on any of the above function(s) check the corresponding reference(s) under [REFERENCES.md](https://github.com/toledo60/pyLHD/blob/main/REFERENCES.md). 


