import argparse

def _get_parser():
    """Parse command line inputs for this function.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(
        description=(
            "MultiSCALE, a toolbox for multilayer network analyses. "
            "It uses a supra-adjacency matrix "
            "(generated in MATLAB), tabular data, "
            "or network data as input, "
            "and creates a multilayer network. "
            "For privacy reasons, we provide a random MST file. \n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    opt_in = parser.add_argument_group("Optional Arguments for Input and Output")
    opt_in.add_argument(
        "-fname",
        "--input-func",
        dest="filename",
        type=str,
        nargs ="+",
        help=(
            "Filename(s) of the input data. "
            "The input data can be a:\n " 
            "- single .mat (e.g., a supra_adjacency matrix);\n "
            "- .csv (e.g., a data matrix of shape N (# subjects) x P (# variables));\n "
            "- path of a folder containing .csv (e.g., a network/connectivity matrix).\n "
            "If .csv and/or a folder of .csv are provided, there should at least be 2 of them, and a supra-adjacency matrix will be modelled subsequently. "
            "Default is supra_randmst.mat"
        ),
        default=["supra_randmst.mat"],
    )
    opt_in.add_argument(
        "-o",
        "--output-func",
        dest="output_filename",
        type=str,
        help=(
            "Filename of the output data. "
            "Default is ''. "
        ),
        default="",
    )
    opt_in.add_argument(
        "-od",
        "--output-dir",
        dest="output_directory",
        type=str,
        help=(
            "Directory of the output data. "
            "Default is 'results'."
        ),
        default="results",
    )
    
    opt_in_proc = parser.add_argument_group("Optional Arguments for Input Processing")
    opt_in_proc.add_argument(
        "-corr",
        "--correlation",
        dest="corr",
        type=str,
        help=(
            "Specify which type of correlation should be used to compute the covariance matrix, either `cor`=Pearson correlation or `par_cor`=Pearson partial correlation. " 
            "Default is `par_cor`."
        ),
        default="par_cor",
    )
    opt_in_proc.add_argument(
        "-omst",
        "--OMST",
        dest="omst",
        type=int,
        nargs="*",
        help=(
            "Specify on which layers should be applied Orthogonal Minimal Spanning Tree (OMST). OMST should be applied to brain networks. " 
            "One can enter a list as -omst 0 1 0 etc. "
            "Default is 0 1."
        ),
        default=[0, 1],
    )
    opt_in_proc.add_argument(
        "-thr",
        "--threshold",
        dest="consensus_thr",
        type=float,
        help=(
            "Specify the threshold for building the consensus matrix across subjects, for network matrices. Only the links which appear in at least `threshold` proportion of subjects are kept. The consensus matrix is then computed by averaging the edges which survived across subjects. A lower threshold value means a higher restriction on the edges, hence a sparser consensus matrix. "
            "Default is 0.1. "
        ),
        default=0.1,
    )
    
    opt_params = parser.add_argument_group("Optional Arguments for Layer Parametrization")
    opt_params.add_argument(
        "-l",
        "--input-layer",
        dest="layer_number",
        type=int,
        help=(
            "Specify the number of layers. "
            "Default is 8."
        ),
        default=8,
    )
    opt_params.add_argument(
        "-s",
        "--input-size",
        dest="layer_size",
        type=int,
        nargs="*",
        help=(
            "Specify the number of regions/nodes per layer. " 
            "If different number of nodes, one can enter a list as -s 20 15 30 etc. "
            "Default is 197."
        ),
        default=197,
    )
    opt_params.add_argument(
        "-f",
        "--function-multilayer",
        dest="function",
        type=str,
        help=(
            "Multilayer function for the "
            "calculation of multilayer network metrics. "
            "Default is bridge_strength"
        ),
        default="bridge_strength",
    )
    
    opt_model = parser.add_argument_group("Optional Arguments for Modelling Inter-Layer Links")
    opt_model.add_argument(
        "-l_RCCA",
        "--lambdas_RCCA",
        dest="lambdas_RCCA",
        type=str,
        help=(
            "Path to a JSON file containing the regularization parameter lambda for each pairwise input data when computing RCCA. This file takes the structure of a 2D matrix where each entry is a list of two components: λ1 and λ2, which correspond to the regularization applied to the datasets. A higher value means a stronger regularization.\n "
            "For example, for two datasets with 28 subjects and 5 variables, and 44 subjects and 210 variables, respectively, we apply λ1=0 (since the first dataset has more subjects than variables and λ2=10). "
            "An example JSON file is provided in RCCA_params.json. "
        ),
    )
    opt_model.add_argument(
        "-l1_RSA",
        "--lambda1_RSA",
        dest="lambda1_RSA",
        type=float,
        help=(
            "Specify the value of lambda1 when building the inter-layer links, specifically the partial correlations between the RSA components. A lower value means a sparser correlation matrix. "
            "Default is 0.01. "
        ),
        default=0.01,
    )
    opt_model.add_argument(
        "-pol",
        "--polarity",
        dest="polarity",
        type=str,
        help=(
            "Specify the polarity of the inter-layer links as `pos` (positive) or `neg` (negative). "
            "Because the directionality of the covariation is meaningful, the built supra adjacency matrix considers positive or negative inter-layer links "
            "Default is pos. "
        ),
        default='pos',
    )

    optional = parser.add_argument_group("Other Optional Arguments")

    optional.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )
    return parser


if __name__ == "__main__":
    print("Running MULTINET Multilayer.")