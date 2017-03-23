#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Empirical community detection using spectral methods
Generate NCP plots and more using PageRank Nibble
Leon Nguyen (A10953601)
Python 2.7.13

Note that this requires networkx-1.9.1 (there is a bug in reading .gml files in newer versions) and that this was written for Python 2.7
"""
from __future__ import division
import networkx as nx
import numpy as np
from math import log, sqrt
from collections import deque
from random import sample
import matplotlib.pyplot as plt
import pickle

def vol(G, S):
    """Compute volume of a subset S. For debugging only."""
    return sum(G.degree(S).values())

def conductance(G, S, T = None):
    """Compute conductance between two subsets S and T. For debugging only."""
    volS = vol(G, S)
    m = G.number_of_edges()
    return len(nx.edge_boundary(G, S, T)) / min(volS, 2 * m - volS)

def approximatePageRank(G, s, alpha, eps):
    """Returns an approximate personalized PageRank (APR) vector
    
    Input parameters:
        G: [NetworkX graph object] G = (V, E)
        s: [int] seed vertex around which APR vector is computed
        alpha: [float] teleportation constant in (0, 1]
        eps: [float] precision parameter inversely proportional to runtime controlling inclusion of vertices in APR vector (the lower the eps, the more vertices included)
    
    Output:
        p: [dict] the APR vector
    
    Description:
        Determine which vertex to push at each step by maintaining a queue containing those vertices u with r(u) >= eps*d(u). At each step, pop the first vertex from the queue and perform a push operation on u. If r(u) is still at least eps*d(u) after the push, then append u to the queue. If a push operation raises the value of r(v) above eps*d(v) for some neighbor v of u, then append v to the queue if it is not already in the queue. This continues until the queue is empty, at which point every vertex u has r(u) < eps*d(u).
    
    Reference:
        Using PageRank to Locally Partition a Graph by R. Andersen, F. Chung, and K. Lang [2007]
    """
    p = {vertex: 0 for vertex in nx.nodes_iter(G)}
    r = p.copy()
    r[s] = 1
    
    queue = deque([s])
    
    while len(queue) > 0:
        u = queue.popleft()
        
        # The push algorithm:
        p[u] += alpha * r[u]
        r[u] = (1 - alpha) * r[u] / 2
        if r[u] > eps * G.degree(u):
            queue.append(u)
        for v in nx.all_neighbors(G, u):
            r[v] += (1 - alpha) * r[u] / (2 * G.degree(u))
            if r[v] > eps * G.degree(v) and queue.count(v) is not 1:
                queue.append(v)
    
#    test = all([r[node]/G.degree(node) < eps for node in nx.nodes_iter(G)])
    
    return p

def pageRankNibble(G, s, phi, b, NCP = False):
    """Returns a cluster of vertices S with conductance less than phi
    
    Input parameters:
        G: [NetworkX graph object] G = (V, E)
        s: [int] seed vertex around which S is found
        phi: [float] target conductance for S, in (0, 1]
        b: [float] precision parameter in [1, ln(m)], runtime proportional to 2^b
        NCP: if False, implement the algorithm as in the reference. if True, perform the modified version to make the NCP plot
    
    Output:
        S: [list] cluster of vertices that represent a community around seed vertex
        Phi: [float] conductance of minimal conductance community
    
    Description:
        Remove the seed vertex s and any vertices corresponding to zero values in the APR vector. Sort the remaining vertices by their APR value in descending order. Sweep through these vertices in this order, adding them to the set S, which initially contains s. Stop when the conductance of S becomes less than phi and the volume of S is between 2^(b-1) and 4m/3.
        
    Reference:
        Using PageRank to Locally Partition a Graph by R. Andersen, F. Chung, and K. Lang [2007]
    """
    m = G.number_of_edges()
    
    alpha = phi ** 2 / (225 * log(100 * sqrt(m)))
    eps = 1 / (2 ** b * 48 * log(m))
    
    p = approximatePageRank(G, s, alpha, eps)
    
    del p[s]
    p = {k: v / G.degree(k) for k, v in p.iteritems() if v > 0}
    pSort = deque([k for (k, v) in sorted(p.items(), key = lambda (k, v): v, reverse = True)])
    
    volS = sum(G.degree([s]).values())
    volSc = 2 * m - volS
    numEdgesOnSetBoundary = len(nx.edge_boundary(G, [s]))
    S = deque([s])
    
    if NCP is False:
        Phi = numEdgesOnSetBoundary / volS
        while not (Phi < phi and volS > 2 ** (b - 1) and volS < 4 * m / 3):
            u = pSort.popleft()
            S.append(u)
            
            volS += G.degree(u)
            volSc -= G.degree(u)
            
            for v in nx.all_neighbors(G, u):
                if v in S:
                    numEdgesOnSetBoundary -= 1
                else:
                    numEdgesOnSetBoundary += 1
            
            if len(pSort) is 0:
                Phi = numEdgesOnSetBoundary / volS
                break
            
            Phi = numEdgesOnSetBoundary / min(volS, volSc)
        
        return list(S), Phi
    
    else:
        Phi = [[] for i in xrange(len(pSort))]
        S = deque([s])
        Phi[0] = numEdgesOnSetBoundary / volS
        for i in range(1, len(pSort)):
            u = pSort.popleft()
            S.append(u)
            
            volS += G.degree(u)
            volSc -= G.degree(u)
            
            for v in nx.all_neighbors(G, u):
                if v in S:
                    numEdgesOnSetBoundary -= 1
                else:
                    numEdgesOnSetBoundary += 1
            
            Phi[i] = numEdgesOnSetBoundary / min(volS, volSc)
        
        return list(S), np.array(Phi)

def findBestCuts(G, phi, b, title):
    """For each community size, find the community with the lowest conductance
    
    Description:
        Use a rough heuristic of sampling half of the graph vertices to perform PageRank Nibble on. Keep track of the communities with the lowest conductance values for each community size k. The idea of randomly sampling nodes is inspired from Spielman and Teng in the reference, but we only implement a uniform sampling without replacement, whereas they draw from a distribution based on the degrees of the vertices. The implementation is roughly inspired by what they do in the "Local_Spectral_Clustering_algorithm" of the SNAP C++ package.
    
    Reference:
        A LOCAL CLUSTERING ALGORITHM FOR MASSIVE GRAPHS AND ITS APPLICATION TO NEARLY LINEAR TIME GRAPH PARTITIONING by D. Spielman and S. Teng
        SNAP: http://snap.stanford.edu/snap/index.html
    """
    n = G.number_of_nodes()
    seedSet = sample(xrange(n), n // 2)
    bestSets = None
    
    try:
        for count, s in enumerate(seedSet):
            S, Phi = pageRankNibble(G, s, phi, b, NCP = True)
            
            if bestSets is None:
                bestSets = {k: S[:k + 1] for k in xrange(len(S))}
                bestPhis = Phi
                continue
            
            j = min(bestPhis.size, Phi.size)
            
            for k in np.where(Phi[:j] < bestPhis[:j])[0]:
                bestSets[k] = S[:k + 1]
            bestPhis[:j] = np.fmin(bestPhis[:j], Phi[:j])
            
            if Phi.size > bestPhis.size:
                for k in xrange(bestPhis.size + 1, Phi.size + 1):
                    bestSets[k] = S[:k + 1]
                bestPhis = np.append(bestPhis, Phi[j:])
            
            print 'Iteration %d of %d: lowest conductance is %f for community of size %d' % (count + 1, n // 2, np.amin(bestPhis), np.nonzero(bestPhis == np.amin(bestPhis))[0][0] + 1)
    except KeyboardInterrupt:
        plotNCP(bestPhis, title)
        return bestPhis, bestSets
    
    return bestPhis, bestSets

def plotNCP(Phi, title):
    """Plot the network community profile (NCP)"""
    minConductance = np.amin(Phi)
    communitySize = np.argmin(Phi) + 1
    titleFrmt = title + ': $\Phi$ = %f for k = %d' % (minConductance, communitySize)
    
    plt.figure(1)
    plt.semilogy(xrange(1, Phi.size + 1), Phi)
    plt.grid(True, which = 'both')
    plt.title(titleFrmt)
    plt.xlabel('k vertices in the community')
    plt.ylabel('Conductance ($\Phi$)')
    plt.tight_layout()
    plt.savefig(title + '.eps', format = 'eps', dpi = 1000)
    plt.show()

def drawGraph(G, Phi, communities, title, vertexList = None, labels = False):
    """Draw a graph of G with the minimum conductance community highlighted"""
    if vertexList == None:
        vertexList = G.nodes()
    plt.figure(2)
    plt.axis('off')
    vertexColors = ['blue' if vertex in communities[np.argmin(Phi)] else 'red' for vertex in vertexList]
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos = pos, node_color = vertexColors, nodelist = vertexList)
    nx.draw_networkx_edges(G, pos = pos, edgelist = nx.edges(G, vertexList))
    if labels == True:
        nx.draw_networkx_labels(G, pos = pos, font_color = 'white')
    plt.tight_layout()
    plt.savefig(title + 'Graph.eps', format = 'eps', dpi = 1000)
    plt.show()

def saveData(Phi, communities, title):
    """Save conductance and community set data"""
    np.save(title + 'Phi.npy', Phi)
    with open(title + 'Communities.pkl', 'wb') as fd:
        pickle.dump(communities, fd)

def loadData(phiFname, communitiesFname):
    """Load conductance and community set data"""
    Phi = np.load(phiFname)
    with open(communitiesFname, 'rb') as fd:
        communities = pickle.load(fd)
    return Phi, communities

# Karate:
G = nx.read_gml('karate.gml')

# Facebook:
#G = nx.read_edgelist('facebook_combined.txt')
G = nx.convert_node_labels_to_integers(G)
m = G.number_of_edges()

phi = 0.1
b = log(m)
title = 'Karate'

Phi, communities = findBestCuts(G, phi, b, title)
plotNCP(Phi, title)
drawGraph(G, Phi, communities, title)
#Phi, communities = loadData('savedData/facebookPhi.npy', 'savedData/facebookCommunities.pkl')

#bestComm = communities[np.nonzero(Phi == np.partition(Phi, 2)[2])[0][0]]
#plotComm = nx.node_boundary(G, bestComm)
#
#for vertex in G.nodes():
#    if vertex in bestComm:
#        G.node[vertex]['color'] = 'blue'
#        G.node[vertex]['style'] = 'filled'
#        G.node[vertex]['fillcolor'] = 'blue'
#    elif vertex in plotComm:
#        G.node[vertex]['color'] = 'red'
#        G.node[vertex]['style'] = 'filled'
#        G.node[vertex]['fillcolor'] = 'red'
#    else:
##        G.node[vertex]['color'] = '#D3D3D380'
##        G.node[vertex]['style'] = 'filled'
##        G.node[vertex]['fillcolor'] = '#D3D3D380'
#        G.remove_node(vertex)
#
#nx.write_dot(G, 'facebookDot')
#drawGraph(G, Phi, communities, title, vertexList = plotComm)