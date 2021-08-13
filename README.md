# pyLHD: Latin Hypercube Designs for Python

pyLHD is a python implementation of the R package [LHD](https://cran.r-project.org/web/packages/LHD/index.html) by Hongzhi Wang, Qian Xiao, Abhyuday Mandal.

Check out the streamlit app for pyLHD, a point-click interface to generate Latin hypercube designs

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/toledo60/pylhd-streamlit/main/app.py)

## Overview

With `pyLHD` you can generate the following designs:

- `rLHD`: Generate a random Latin hypercube design
- `GLPdesign`: Generate a good lattice point design
- `OLHD_Butler01`: Orthogonal Latin hypercube design. Based on the construction method of Butler (2001)
- `OLHD_Cioppa07`: Orthogonal Latin hypercube design. Based on the construction method of Cioppa and Lucas (2007)
- `OLHD_Lin09`: Orthogonal Latin hypercube design. Based on the construction method of Lin et al. (2009)
- `OLHD_Sun10`: Orthogonal Latin hypercube design. Based on the construction method of Sun et al. (2010)
- `OLHD_Ye98`: Orthogonal Latin hypercube design. Based on the construction method of Ye (1998)

Calculate design properties such as:

- `MaxAbsCor`: Calculate the maximum absolute correlation
- `AvgAbsCor`: Calculate the average absolute correlation
- `MaxProCriterion`: Calculate the maximum projection criterion
- `dij`: Calculate the Inter-site Distance (rectangular/Euclidean) between the *ith* and *jth* row
- `phi_p`: Calculate the phi_p criterion

Other functionality includes:

- `exchange`: Exchange two random elements from a specified column or row in a matrix
- `OA2LHD`: Convert an orthogonal array (OA) into a Latin hypercube design (LHD)
- `williams_transform`: Apply Williams transformation to a specified design

For further details on any of the above function(s) check the corresponding reference(s) under [REFERENCES.md](https://github.com/toledo60/pyLHD/blob/main/REFERENCES.md). 

## Installation

`pyLHD` can be installed from PyPI:

```
pip install pyLHD
```

The latest development version can be installed from the main branch using pip:

```
pip install git+https://github.com/toledo60/pyLHD.git
```

The main dependency for `pyLHD` is [NumPy](https://numpy.org/) and currently tested on Python 3.6+


## Future Work

Search algorithms for efficient LHDs have not yet been implemented. These algorithms include: simulated annealing, particle swarm optimization, and genetic algorithms.

Until these algorithms have been implemented, refer to the R package [LHD](https://cran.r-project.org/web/packages/LHD/index.html) to utilize the search algorithms described above. Lastly, for a good overview of the methods being implemented check out [Musings about Constructions of Efficient Latin Hypercube Designs with Flexible Run-sizes](https://arxiv.org/abs/2010.09154) (R version)

Quick tutorials in Jupyter notebooks demonstrating the functionality of `pyLHD` in Python will be created in future updates