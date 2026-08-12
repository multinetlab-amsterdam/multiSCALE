# multiSCALE

MultiSCALE (Multilayer Systems for Cross-modal Analysis and Linking of Entities) is a Python toolbox for modelling, computing, and analyzing multilayer networks. It can accept a supra-adjacency matrix as input, as well as tabular or network data, in which cases inter-layer links are modelled to create a multilayer network. Multilayer network metrics were implemented based on the [MuxVizPy](https://github.com/CoMuNeLab/MuxVizPy/tree/master) Python package, which is backed by [`graph-tool`](https://graph-tool.skewed.de/). It is installed as a submodule here. 

---

## Installation 

To install MultiSCALE, conda and [uv](https://docs.astral.sh/uv/) are needed to install `graph-tool` and Python dependencies, respectively. 

1. Create a conda environment with graph-tool and activate it

```bash
conda create -n multiscale python=3.12 graph-tool -c conda-forge
conda activate multiscale
```

2. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install the package

```bash
git clone https://github.com/multinetlab-amsterdam/multiSCALE.git
cd multiSCALE

uv pip install -e .
```

## Dependencies

The minimum required dependencies to run MultiSCALE are:

* numpy>=1.24
* scipy>=1.10
* pandas>=2.3.3
* polars>=0.20
* pyarrow>=14.0
* sparse>=0.15.4
* numba>=0.59
* tensorly>=0.8
* matplotlib>=3.4
* tqdm>=4.60
* scikit-learn>=1.7.0
* gglasso>=0.1.7 

## Input

MultiSCALE can accept as input: 
* a supra-adjacency matrix as a `.mat`
* one or multiple `.csv` where the rows are observations and columns are variables
* one or multiple folders which have `.csv` corresponding to connectivity matrices of individual subjects

## License

MultiSCALE is licensed under the GPL-3.0 license.

## Acknowledgements

* The multilayer network metrics are based on the [MuxVizPy](https://github.com/CoMuNeLab/MuxVizPy/tree/master) Python package.
