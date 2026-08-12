import sys
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm
import json

from . import model
from . import utils

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MuxVizPy', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'omst')))

from MuxVizPy import mesoscale, topology, versatility
import graph_tool as gt
from threshold_omst_gce_wu import threshold_omst_gce_wu

FUNCTIONS = {
    "modularity": mesoscale.get_mod,
    "clustering_coefficient": mesoscale.compute_local_clustering_coefficient_vector, 
    "eccentricity": topology.get_eccentricity,
    "strength": versatility.compute_multi_strength_vector,
    "eigenvector_centrality": versatility.compute_eigenvector_centrality,
    "bridge_strength": versatility.get_bridge_strength,
}

def process_input_data(input_data):
    if len(input_data) == 1 and ".mat" in input_data[0]: 
        return scipy.io.loadmat(input_data[0])
    else: 
        data = []
        # only .csv (data matrices)
        if all(fname.endswith(".csv") for fname in input_data):
            for fname in input_data:
                data_matrix = pd.read_csv(fname)
                data.append(data_matrix)
        # only folder of .csv (network matrices)
        elif not any(fname.endswith(".csv") for fname in input_data):
            for fname in input_data:
                network_matrices = []
                for file in sorted(os.listdir(fname)):
                    if os.path.isdir(file):
                        continue
                    if file.isascii():
                        network_matrix = (pd.read_csv(os.path.join(fname, file)).to_numpy())
                        network_matrices.append(network_matrix)
                data.append(network_matrices)
        # .csv and folder of .csv (some data and network matrices)
        elif not all(fname.endswith(".csv") for fname in input_data):
            for fname in input_data:
                print("Loading ", fname)
                if ".csv" in fname:
                    data_matrix = pd.read_csv(fname)
                    data.append(data_matrix)
                else:
                    network_matrices = []
                    for file in sorted(os.listdir(fname)):
                        if os.path.isdir(file):
                            continue
                        if file.isascii():
                            network_matrix = (pd.read_csv(os.path.join(fname, file)).to_numpy())
                            network_matrices.append(network_matrix)
                    data.append(network_matrices)
        else: 
            raise ValueError("Error in input data, use help to see the types of input data accepted.")
            
        return data
    
def load_matrix(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data)  # shape: (n, n, 2)

def multilayer(function, data, filename, output_directory, layers, N):
    """
    Wrapper function which calls the corresponding function based on the multilayer network metric input. 
    
    Parameters
    ----------
    function: One of the functions developed in this code for Multilayer Networks
    data: The data we want to use: e.g., supra_mst
    filename: The name of the file you want to save
    output_directory: The name of the directory you want to save
        Default: "results/"
    layers: number of layers
    N: number of nodes per layer
    """
    layer = []
    local_node = []
    phys_node = []
    
    # In the case where the first layer has its unique nodes and the other layers  have shared nodes, then the offset starts at the index corresponding to the number of nodes in the first layer
    shared_offset = N[0]
    
    for i, n in enumerate(N):
        layer.extend([i] * n)
        local_node.extend(range(n))
        if i == 0:
            # layer 0: unique nodes, not shared
            phys_node.extend(range(0, N[0]))
        else:
            # other layers: share the same nodes 
            phys_node.extend(range(shared_offset, shared_offset + n))
    
    node_table = pd.DataFrame({
        "layer": layer,
        "local_node": local_node,
        "phys_node": phys_node,
    })
    
    if function.__name__ == "get_mod": 
        adj = np.tril(data)
        idx = adj.nonzero()
        weights = adj[idx]
        g = gt.Graph(directed=False)
        g.add_edge_list(np.transpose(idx))
        
        #add weights as an edge propetyMap
        ew = g.new_edge_property("double")
        ew.a = weights 
        g.ep['weight'] = ew
        temp = list(function(g, layers))
    elif function.__name__ == "get_multi_path_statistics":
        temp = function(data, layers, N, node_table)
        if isinstance(temp, dict):
            for metric, value in temp.items():
                basename, ext = filename.rsplit("_", maxsplit=1)
                new_filename = metric + "_" + ext
                utils.save_csv(value, new_filename, output_directory)
            return temp
    elif function.__name__ == "compute_eigenvector_centrality" or \
    function.__name__ == "compute_multi_strength_vector" or \
    function.__name__ == "compute_local_clustering_coefficient_vector" or \
    function.__name__ == "get_eccentricity":
        temp = function(data, layers, N, node_table)
    else:
        temp = list(function(data, layers, N))
        
    # utils.save_csv(temp, filename, output_directory)
    return temp

def main(argv=None):
    from .cli.run import _get_parser
    import graph_tool.draw as gtdraw
    
    options = _get_parser().parse_args(argv)
    
    if len(options.layer_size) == 1:
        options.layer_size = options.layer_size[0]
    
    print(f"Computing {options.function} for {options.filename} with {options.layer_number} layers and {options.layer_size} nodes:\n")
    
    # Process input data for multilayer analysis
    print("Number of input data:", len(options.filename))
    data = process_input_data(options.filename)
    
    ## supra adjacency matrix 
    if len(options.filename) == 1 and ".mat" in options.filename[0]: 
        name = list(data.keys())[-1]
        supra = data[name]
    ## RCCA + RSA
    else:
        ### OMST, consensus, partial correlations
        data_type = []
        
        intra_layers = []
        data_matrices = []
        data_omst = []
        consensus_matrices = []
        for matrices in data: 
            # OMST if network matrices
            if isinstance(matrices, list): 
                omst_graphs = []
                for network_matrices in tqdm(matrices, desc="Computing OMST..."):
                    _, omst_graph, _, _, _, _ = threshold_omst_gce_wu(network_matrices, flag=0)
                    omst_graphs.append(omst_graph)
                data_omst.append(omst_graphs)
                # Consensus
                consensus_mat = model.compute_consensus(omst_graphs, threshold=options.consensus_thr)
                consensus_mat = model.normalize_0_1_matrix(consensus_mat)
                consensus_matrices.append(consensus_mat)
                
                data_type.append("network")
                intra_layers.append(consensus_mat)
                data_matrices.append(0) # dummy for correct indexing
            # Partial correlations if .csv
            elif isinstance(matrices, pd.DataFrame): 
                print("computing par cor")
                print(matrices.shape)
                print(matrices)
                if options.correlation == "cor":
                    cor = matrices.corr()
                    np.fill_diagonal(cor.values, 0)
                    # FIXME: absolutize or divide into + and - ?
                    cor = model.normalize_0_1_matrix(np.abs(cor))
                elif options.correlation =="par_cor":
                    cor = model.ebic_glasso(matrices, gamma=0.5, max_iter=10000)["network"]
                    print(cor)
                    cor = model.normalize_0_1_matrix(cor)
                data_matrices.append(matrices)
                data_type.append("data")
                intra_layers.append(cor)
                data_omst.append(0) # dummy for correct indexing
                
        ### Inter-layer links
        interlayer_links = []
        for i in tqdm(range(options.layer_number), desc="Modelling inter-layer links...\n"):
            for j in range(i+1, options.layer_number):
                if i == j:
                    continue
                
                # Check data type 
                if data_type[i] == "network":
                    X = np.vstack([np.sum(data_omst[i][k], axis=0) for k in range(len(data_omst[i]))])
                else: 
                    X = data_matrices[i].to_numpy()
                    
                if data_type[j] == "network":
                    Y = np.vstack([np.sum(data_omst[j][k], axis=0) for k in range(len(data_omst[j]))])
                else:
                    Y = data_matrices[j].to_numpy()
                
                # RCCA
                X_loadings, Y_loadings = model.RCCA_multi(
                    X=X, 
                    Y=Y,
                    lambda1=options.lambdas_RCCA[i, j][0],
                    lambda2=options.lambdas_RCCA[i, j][1]
                )
                
                # Inter-layer links
                if data_matrices:
                    N = len(data_matrices[0])
                else:
                    N = len(omst_graphs)
                inter_layers = model.partial_interlayer(
                    X_loadings=X_loadings, 
                    Y_loadings=Y_loadings, 
                    N=N,
                    lambda1=options.lambda1_RSA
                )
                
                if inter_layers.shape != (X_loadings.shape[0], Y_loadings.shape[0]):
                    inter_layers = inter_layers.T
                
                inter_layers_split = model.split_interlayer_links(inter_layers)
                
                interlayer_links.append(inter_layers_split[options.polarity])
          
        ### Supra-adjacency matrix
        supra = model.build_supra_adjacency(intra_layers, interlayer_links, options.layer_size)
    
    filename=options.function + "_" + options.output_filename
    print("Saving", filename)
    if len(supra.shape)==2:
        Data=supra
    else:
        Data=supra[..., 0]
    
    result = multilayer(FUNCTIONS[options.function], Data, filename, options.output_directory, options.layer_number, options.layer_size)
    
    # print("Results: {}".format('\n '.join(map(str, result))))

if __name__ == "__main__":
    main()