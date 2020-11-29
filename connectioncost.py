from Graph import Graph
from mutations import *
from parameters import *
import networkx as nx
import activations as activations

import numpy as np
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
'''
we minimize twice to ensure that neurons in the same layer do not overlap. It could be enoug to minimze one time, i only do it for better visualization

'''
def objective(r):
    return r.dot(np.squeeze(np.asarray(L.dot(r.transpose()))))

def constraint1(r):
    return np.dot(r,r.transpose()) -1

def constraint2(r):
    return np.sum(r) 

def constraint3(r):
    #set minimial euclidian distance between nodes
    minDist = 0.2
    posis[:,0] = r
    x = np.array([pt[0] for pt in posis])
    y = np.array([pt[1] for pt in posis])
    dists = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
    return dists.flatten() - minDist

def getCost(WG):
    global L ,posis
    graphCost = Graph()
    graphCost.DG = WG.copy()
    graphCost.WG = WG.copy()

    #need positive weigh values to minimize
    weights = nx.get_edge_attributes(graphCost.WG, "weight")
    absweights = np.absolute(list(weights.values()))
    weights.update(zip(weights,absweights))
    nx.set_edge_attributes(graphCost.WG, weights, "weight")

    #make undirected graph becouse directions dosnt matter and the adjancency matrix needs to be symetric ( Laplacian depends on adjancency)
    undirectedWG= graphCost.WG.to_undirected()
    L = nx.laplacian_matrix(undirectedWG)
    L = L.todense()

    position = nx.get_node_attributes(graphCost.WG, "pos")
    position = list(position.values())
    posis = np.float_(np.array(position))

    #starting condition for optimizer
    pos0 = posis[:,0]           ## we only optimze x-value
    pos0 = pos0[np.newaxis,:]

    #set conditions
    con1 = {'type': 'eq', 'fun': constraint1}
    con2 = {'type': 'eq', 'fun': constraint2}
    con3 = {'type': 'ineq', 'fun': constraint3}
    cons = [con1,con2] #only first two for first optimization, following the paper
    #run first optimisation
    sol = minimize(objective,pos0, method='SLSQP', constraints = cons)

    #set new position
    posis[:,0] = sol.x
    posis = posis.round(2)

    graphCost.setNodeAttribute(posis,"pos")
    pos = nx.get_node_attributes(graphCost.WG, "pos")
    #graphCost.printGraph()
    #find nodes at identical pos
    unique, index, count = np.unique(posis, axis=0, 
                            return_index=True,
                            return_counts=True)

    # add 0.3 to x-coord if at same pos, as long as nodes at same pos
    while(True):
        for idx,i in enumerate (count):
            if i >1:
                need = posis[index[idx]]
                rowIdx = np.argwhere((posis[:,0] == need[0]) * (posis[:,1] == need[1]))
                posis[rowIdx[0],0] += 0.3
        unique, index, count = np.unique(posis, axis=0, 
                            return_index=True,
                            return_counts=True)    
        if all(count ==1):
            break

    # 2nd optimisation to get  optimzied values after manuel changes in c-coord
    cons = [con1,con2,con3]
    pos0 = posis[:,0]
    pos0 = pos0[np.newaxis,:]
    sol = minimize(objective,pos0, method='SLSQP', constraints = cons)

    posis[:,0] = sol.x
    posis = posis.round(2)
    graphCost.setNodeAttribute(posis,"pos")
    pos = nx.get_node_attributes(graphCost.WG, "pos")
    #graphCost.printGraph()
    return sol.fun/MAXCC