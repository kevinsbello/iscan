# ![iSCAN](https://raw.githubusercontent.com/kevinsbello/iscan/master/logo/iscan.png)

<div align=center>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/v/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/pyversions/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/wheel/iscan-dag"></a>
  <a href="https://pypistats.org/packages/iscan-dag"><img src="https://img.shields.io/pypi/dm/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/l/iscan-dag"></a>
</div>


The `iscan-dag` library is a Python 3 package designed for detecting which variables, if any, have undergo a casual mechanism shift *given multiple datasets*. 

iSCAN operates through a systematic process:

1. For each dataset, iSCAN initially evaluates at each sample the Hessian of the data distribution. This step is helpful in identifying the leaf variables (nodes) for all the datasets.
2. Subsequently, for the identified leaf variable, iSCAN evaluates at each sample the Hessian of the data distribution for the pooled data (resembling a mixture distribution). Then, based on the variance of the Hessian values, iSCAN determines if the given leaf node has undergone a mechanism shift (termed **shifted node**).

The steps above are applied iteratively, eliminating the identified leaf variable across all datasets at each iteration. See [`iscan.est_node_shifts`](https://iscan-dag.readthedocs.io/en/latest/api/iscan/shifted_nodes/est_node_shifts/) for more details.

As an optional step, the library also includes a function to detect structural changes (termed **shifted edges**). As a by-product of detecting shifted nodes, iSCAN also estimates a topological ordering of the causal variables. Thus, allowing for the use of recent methods on variable (parents) selection. The current implementation of iSCAN employs  [`FOCI`](https://cran.r-project.org/web/packages/FOCI/index.html)  to identify the parent set of shifted nodes in each dataset. See [`iscan.est_struct_shifts`](https://iscan-dag.readthedocs.io/en/latest/api/iscan/shifted_edges/est_struct_shifts/) for more details.


## Citation

This is an implementation of the following paper:

[1] Chen T., Bello K., Aragam B., Ravikumar P. (2023). ["iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models"][iscan]. Neural Information Processing Systems ([NeurIPS](https://nips.cc/Conferences/2023/)). 

[iscan]: https://arxiv.org/abs/2306.17361

If you find this code useful, please consider citing:

### BibTeX

```bibtex
@article{chen2023iscan,
    author = {Chen, Tianyu and Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
    title = {{iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models}},
    journal = {arXiv preprint arXiv:2306.17361},
    year = {2023}
}
```

## Features

- Shifted nodes are detected without the need to estimate the DAG structure for each dataset.
- iSCAN is agnostic to the type of score's Jacobian estimator. The current implementation is based on a kernelized Stein's estimator. See [`stein_hess`](https://iscan-dag.readthedocs.io/en/latest/api/iscan/score_estimator/stein_hess/) for details.
- iSCAN's time complexity is not influenced by the underlying graph density, and will run faster than methods such as DCI or UT-IGSP for large number of variables due to its omission of (non)parametric conditional independence tests.

## Getting Started

### Install the package

We recommend using a virtual environment via `virtualenv` or `conda`, and use `pip` to install the `iscan-dag` package.
```bash
$ pip install -U iscan-dag
```

### Using iSCAN

See an example on how to use iSCAN in this [iPython notebook][example].

[example]: https://github.com/kevinsbello/iscan/blob/master/example/example.ipynb

## An Overview of iSCAN

We propose a new method for directly identifying changes (shifts) of causal mechanisms from multiple heterogeneous datasets, which are assumed to be originated by related structural causal models (SCMs) over the same set of variables. 

iSCAN considers that each **SCM belongs to the general class of nonlinear additive noise models** (ANMs), thus, generalizing prior work that assumed linear models. We assume that each dataset is generated from an interventional (observational if no variables are intervened) distribution of an underlying graph $G^*$. See the figure below.

<img width="1335" alt="" src="https://github.com/kevinsbello/iscan/assets/6846921/ecebed13-8968-4a5e-a404-4b110b5eefd6">


In [[1]][iscan], we prove that the Hessian of the log-density function of the **mixture distribution** reveals information about changes (shifts) in general non-parametric functional mechanisms for the leaf variables. Thus, allowing for the detection of shifted nodes. Our method leads to significant improvements in identifying shifted nodes.

**Theorem 1 (see [[1]][iscan]).** 
Let $h$ be the index of the environment (dataset), and $p^h(x)$ denote the pdf of the $h$-th environment. Let $q(x)$ be the pdf of the mixture distribution of the all $H$ environments such that $q(x) = \sum_h w_h p^h(x)$. Also, let $s(x) = \nabla \log q(x)$ be the associated score function. Then, under mild assumptions, if $j$ is a leaf variable in all environments, we have:

$$ 
j \text{ is a shifted node } \iff  \text{Var}_{q}\left[ \frac{\partial s_j(X)}{\partial x_j} \right] > 0.
$$


## Requirements

- Python 3.6+
- `numpy`
- `igraph`
- `torch`
- `scikit-learn`
- `rpy2` (R interface to use the `FOCI` library).
- `GPy` (Library to sample from Gaussian processes)
- `kneed` (Used for the elbow heuristic)
- `pandas`

## Contents

- `score_estimator.py`:  Estimates the diagonal of the Hessian of $\log p(x)$ at the provided samples points.
- `utils.py`: Utility functions for generating synthetic data, and evaluate the results
- `shifted_nodes.py`: Implements iSCAN, providing detected shifted nodes.
- `shifted_edges.py`: Implements the discovery of structural changes (shifted edges).
- `my_foci.R`: R implementation that uses `FOCI` for finding parents based on given nodes and topological order.

## Acknowledgements

We thank the authors of the [SCORE](https://github.com/paulrolland1307/SCORE/tree/main) for making their code available. Part of our code is based on their implementation, specially the `score_estimator.py` file.
