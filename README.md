# ![iSCAN](https://raw.githubusercontent.com/kevinsbello/iscan/main/logo/iscan.png)

<div align=center>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/v/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/pyversions/iscan-dag"></a>
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/wheel/iscan-dag"></a>
  <!-- <a href="https://pypistats.org/packages/iscan-dag"><img src="https://img.shields.io/pypi/dm/iscan-dag"></a> -->
  <a href="https://pypi.org/project/iscan-dag"><img src="https://img.shields.io/pypi/l/iscan-dag"></a>
</div>


The `iscan-dag` library is a Python 3 package for ...


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

- 

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



## Requirements

- Python 3.6+
- `numpy`
- `igraph`
- `torch`

## Acknowledgements