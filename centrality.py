from networkx.algorithms.shortest_paths import weighted
import numpy as np
import networkx as nx
import torch

def eigen_centrality(data):
    """Calculates the eigenvector centrality for the given graph

    Parameters:
    data (np.array): Adjacency matrix representation of the graph

    Returns:
    np.array: Array of eigenvector centralities
    """
    g = nx.from_numpy_matrix(data)

    # # compute egeinvector centrality and transform the output to vector
    ec = nx.eigenvector_centrality_numpy(g, weight="weight")
    eigenvector_centralities = np.array([ec[g] for g in ec])

    return eigenvector_centralities


def pagerank_centrality(data):
    """Calculates the pagerank centrality for the given graph

    Parameters:
    data (np.array): Adjacency matrix representation of the graph

    Returns:
    np.array: Array of pagerank centralities
    """
    g = nx.from_numpy_matrix(data)

    # # compute egeinvector centrality and transform the output to vector
    cc = nx.pagerank(g, weight="weight")
    pagerank_centralities = np.array([cc[g] for g in cc])

    return pagerank_centralities


def betweenness_centrality(data):
    """Calculates the betweenness centrality for the given graph

    Parameters:
    data (np.array): Adjacency matrix representation of the graph

    Returns:
    np.array: Array of betweenness centralities
    """
    g = nx.from_numpy_matrix(data)

    # # compute egeinvector centrality and transform the output to vector
    bc = nx.betweenness_centrality(g, weight="weight")
    betweenness_centralities = np.array([bc[g] for g in bc])

    return betweenness_centralities

def node_strength(data):
    """Calculates the node strenght for the given graph
    
    Parameters:
    data (np.array): Adjacency matrix represntation of the graph

    Returns:
    np.array: Array of node strenghts
    
    """
    g = nx.from_numpy_array(data)
    ns = g.degree(weight='weight')
    ns = dict(ns)
    node_strengties = np.array([ns[g] for g in ns])
    return node_strengties

def closeness_centrality(data):
    """Calculates the closeness centrality for the given graph
    
    Parameters:
    data (np.array): Adjacency matrix represntation of the graph

    Returns:
    np.array: Array of closeness centrality
    
    """
    g = nx.from_numpy_array(data)
    cc =nx.closeness_centrality(g,distance='weight')
    closeness_centralities = np.array([cc[g] for g in cc])
    return closeness_centralities

def cluster_coeff(data):
    """Calculates the cluster coeffients for the given graph
    
    Parameters:
    data (np.array): Adjacency matrix represntation of the graph

    Returns:
    np.array: Array of cluster coeffients
    """

    g = nx.from_numpy_array(data)
    cc =nx.clustering(g,weight='weight')
    cluster_coeffs = np.array([cc[g] for g in cc])
    return cluster_coeffs

def topological_measures(data):
    """Returns the topological measures for the given graph

    Parameters:
    data (np.array): Adjacency matrix representation of the graph

    Returns:
    np.array: Array with first element pagerank centrality,
                         second element betweenness centrality
                         third element eigenvector centrality
                         fourd element node strenght
                         fifth element closeness_centrality
                         sixth element clustering coeficients
    """
    topology = []
    topology.append(torch.tensor(pagerank_centrality(data)))  # 0
    topology.append(torch.tensor(betweenness_centrality(data)))  # 1
    topology.append(torch.tensor(eigen_centrality(data)))  # 2
    topology.append(torch.tensor(node_strength(data))) # 3
    topology.append(torch.tensor(closeness_centrality(data))) # 4
    topology.append(torch.tensor(cluster_coeff(data))) # 5
    return topology