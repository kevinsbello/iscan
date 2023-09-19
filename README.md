# ![iSCAN](https://raw.githubusercontent.com/kevinsbello/iscan/master/logo/iscan.png)

<div align=center>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/v/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/pyversions/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/wheel/iscan-dag"></a>
  <!-- <a href="https://pypistats.org/packages/iscan-dag"><img src="https://img.shields.io/pypi/dm/iscan-dag"></a> -->
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/l/iscan-dag"></a>
</div>


The `iscan-dag` library is a Python 3 package designed for the direct detection of shifted nodes and structural shifted edges across multiple DAGs originating from distinct environments.

iSCAN-dag operates through a systematic process:

1. It initially calculates the derivatives of the score function, a key step in identifying the leaf nodes for all environments.
2. Subsequently, it computes the derivative of the score function from the mixture distribution, then evaluating the variance of these derivatives. If the variance exceeds a threshold of zero, it designates these leaf nodes as shifted nodes.

This process is iteratively applied, eliminating the identified leaf nodes across all environments and repeating the procedure to uncover all shifted nodes.

To detect structural shifted edges, the library leverages the by-products of the prior steps, the topological order. It employs  [`FOCI`](https://cran.r-project.org/web/packages/FOCI/index.html)  to identify discrepancies in parental relationships.


## Citation

This is an implementation of the following paper:

[1] Chen T., Bello K., Aragam B., Ravikumar P. (2023). ["iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models"][iscan]. 

[iscan]: https://arxiv.org/abs/2306.17361

If you find this code useful, please consider citing:

### BibTeX

```bibtex
@article{chen2023iscan,
    author = {Chen, Tianyu and Bello, Kevin and Aragam, Bryon and Ravikumar, Pradeep},
    journal = {ArXiv Preprint 2306.17361},
    title = {{iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models}},
    year = {2023}
}
```

## Features

- Detecting shifted nodes without the need for separate DAG estimations.
- Accommodates any score estimators that can seamlessly integrate into this versatile framework.
- Unlike DCI and UT-IGSP, iSCAN's time complexity is not influenced by graph density and runs faster in larger networks due to its omission of non-parametric conditional independence tests.

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

We propose a new method of  identifying changes (shifts) in causal mechanisms between related Structure causal models  (SCMs) directly, without  recovering the entire underlying DAG structure. This paper focuses on identifying mechanism shifts in two or more related SCMs over the same set of variables---$\textit{without estimating the entire DAG structure of each SCM}$. Prior work under this setting assumed linear models with Gaussian noises; instead, in this work we assume that each SCM belongs to the more general class of nonlinear additive noise models (ANMs). We prove a surprising result where the Jacobian of the score function for the $\textit{mixture distribution}$ reveals information about shifts in general non-parametric functional mechanisms. Once the shifted variables are identified, we leverage recent work to estimate the structural differences (if any) for the shifted variables.  The advantages of our method is it is easy to understand and implement, lead to significant improvement in identifying shifted nodes (e.g F1, Recall, Precision).

### Identifying the shifted leaf nodes

Let $h$ be the index of the environment, and $p^h(x)$ denote the pdf of the $h$-th environment. Let $q(x)$ be the pdf of the mixture distribution of the all $H$ environments such that $q(x) = \sum_{h=1}^H w_h p^h(x)$.
Also, let $s(x) = \nabla \log q(x)$ be the associated score function. 
Then, under with nonlinear assumption and additive noise assumption, we have:

[$$
(i) \text{ If } j \text{ is a leaf in all DAGs } G^h, \text{ then } j \text{ is a shifted node if and only if }  \text{Var}_X\left[ \frac{\partial s_j(X)}{\partial x_j} \right] > 0 \\
(ii) \text{ If } j \text{ is not a leaf in at least one DAG } G^h, \text{ then } \text{Var}_X\left[ \frac{\partial s_j(X)}{\partial x_j} \right] > 0
$$](https://latex.codecogs.com/svg.image?\inline&space;(i)\text{If}\,j\,\text{is&space;a&space;leaf&space;in&space;all&space;DAGs}\,G^h,\text{then}\,j\,\text{is&space;a&space;shifted&space;node&space;if&space;and&space;only&space;if}\,\text{Var}_X\left[\frac{\partial&space;s_j(X)}{\partial&space;x_j}\right]>0\\(ii)\text{If}\,j\,\text{is&space;not&space;a&space;leaf&space;in&space;at&space;least&space;one&space;DAG}\,G^h,\text{then}\,\text{Var}_X\left[\frac{\partial&space;s_j(X)}{\partial&space;x_j}\right]>0&space;)

### Identifying the shifted edges

By utilizing a common estimated topological order across all environments, individuals can customize their definition of functional shifted edges and apply any available (non)parametric statistical technique to identify these edges based on the detected shifted nodes. This approach can significantly expedite the process, particularly when the occurrence of shifted nodes is sparse, obviating the need for exhaustive edge comparisons across all nodes and environments.

Regarding structural shifted edges, we have established a connection between FOCI and our method, enabling the detection of such edges with a time complexity of $\mathcal{O}(n\log n)$.

## Requirements

- Python 3.6+
- `numpy`
- `igraph`
- `torch`

## Contents

- `score_estimator.py`:  Estimates the diagonal of the Hessian of $\log p(x)$ at the provided samples points.
- `utils.py`
  - `set_seed`: Manually sets the random seed.
  - `node_metrics`, `ddag_metrics`: Metrics for identifying shifted nodes and structural shifted edges.
  - `DataGenerator`: Generates data for testing purposes.
- `shifted_nodes.py`: Implements iSCAN, providing detected shifted nodes and test cases.
- `shifted_edges.py`: Implements the discovery of structural shifted edges using FOCI, along with test cases.
- `my_foci.R`: Implements FOCI for finding parents based on given nodes and topological order.

## Acknowledgements

We thank the authors of the [SCORE](https://github.com/paulrolland1307/SCORE/tree/main) for making their code available. Part of our code is based on their implementation, specially the `score_estimator.py` file.
