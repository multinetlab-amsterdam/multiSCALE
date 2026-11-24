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

## Usage

One can use the followng command to display the arguments: `python -m src.workflow -h`. 

Based on the input data:
* A supra-adjacency matrix `.mat`
One can directly compute a multilayer network metric, e.g., bridge strength, in the folder `results` as an output file `bridge_strength_rand` as: `python -m src.workflow --function bridge_strength -fname supra_randmst.mat -mn supra -o bridge_strength_rand -od results -l 5 -s 210 210 210 210 210`. 
* Two `.csv`: 
One can choose different values for the following arguments: `-corr`, `-omst`, `-thr`.
One important argument is `-l_RCCA`. The user will need to design a JSON file containing the regularization parameter lambda for each pairwise input data when computing RCCA, see -help for more details, and the example file `RCCA_params.json`.

Once all arguments have been determined, the user can adapt the following command for his/her own usage: `python -m src.workflow -fname file1.csv file2.csv -l 2 -s 210 210 -omst 0 1 1 1 1 1 -thr 0.1 -l_RCCA RCCA_params.json -l1_rsa 10 -pol pos`. 

If the user wants to directly manipulate the functions from the toolbox in an IDE, one can take a look at the tutorial.ipynb for an example on how to use them based on synthetic data. 
## License

MultiSCALE is licensed under the GPL-3.0 license.

## Acknowledgements

* The multilayer network metrics are based on the [MuxVizPy](https://github.com/CoMuNeLab/MuxVizPy/tree/master) Python package.
