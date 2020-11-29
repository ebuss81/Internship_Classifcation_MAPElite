import networkx as nx

from modularity_maximization import partition
from modularity_maximization.utils import get_modularity
import activations as activations
import matplotlib.pyplot as plt
from mutations import*
import numpy as np
import random

from parameters import *

class Graph:
    def __init__(self, DG = nx.DiGraph()):
        # build Graphs
        self.DG = DG                # fully Connected, full size
        self.WG = DG.copy           # working grapgh
        self.printGraph = self._printGraph
        self.NeuronsperLayer = NPL
        self.NumBiasedLayers =  NBL
        self.StartNeuronsPerLayer = SNPL
        self.biases = []
        self.weightsRange = WR
        self.biasRange = BR

        

    def removeNodes(self):
        activeNodes = nx.get_node_attributes(self.WG, "active")
        [self.WG.remove_node(node) for node in activeNodes if activeNodes.get(node) == False]
        return None
    def removeParentNodes(self):
        activeNodes = nx.get_node_attributes(self.DG, "active")
        [self.DG.remove_node(node) for node in activeNodes if activeNodes.get(node) == False]
        return None

    def removeEdge(self):
        activeEdges = nx.get_edge_attributes(self.WG, "active")
        [self.WG.remove_edge(*edge) for edge in activeEdges if activeEdges.get(edge) == False]
        return None

    def addNodeFromParent(self,node):
        dic= self.DG.nodes(data=True)
        self.WG.add_nodes_from([(node,dic[node])])

    def addEdgeFromParent(self,connection):
        #needs ad input (x,v)
        self.WG.add_edges_from([(*connection,self.DG[connection[0]][connection[1]])])

    def getWeights(self):
        return nx.get_edge_attributes(self.WG, "weight")

    def getActivation(self):
        return nx.get_node_attributes(self.WG, "ActFunc")
    
    def setNodeAttribute(self, attribute, attributeName):
        NodeAtrributes = {n: attribute[idx] for idx,n in enumerate(self.WG.nodes)}
        nx.set_node_attributes(self.WG, NodeAtrributes, attributeName)
        nx.set_node_attributes(self.DG, NodeAtrributes, attributeName)

    def setWeights(self,weights):
        edge_weights = {e: weights[idx] for idx,e in enumerate(self.WG.edges)}
        nx.set_edge_attributes(self.WG, edge_weights,"weight")
        if self.NumBiasedLayers >=1:
            self.biases[-self.NumBiasedLayers:] = weights[-self.NumBiasedLayers:]


    def getCommunitiesModularity(self):
        comm_dict = partition(self.WG)
        return get_modularity(self.WG, comm_dict)

    def _printGraph(self):
        plt.subplot(121)
        pos = nx.get_node_attributes(self.WG, "pos")
        nx.draw_networkx(self.WG, pos = pos, with_labels=True, font_weight='bold')
        plt.draw()
        plt.show()

    def randomIndi(self):
        #random connections
        self.WG = self.DG.copy()
        activeEdges = nx.get_edge_attributes(self.DG, "active")
        acts= np.random.choice(a=[False, True], size=(1,self.WG.number_of_edges()),p = [0.1,0.9])
        activeEdges.update(zip(activeEdges,acts[0])) 
        nx.set_edge_attributes(self.DG,activeEdges,"active")
        nx.set_edge_attributes(self.WG,activeEdges,"active")
        self.removeEdge()
        #print(nx.get_edge_attributes(self.DG, "active"))
        
        #random bias
        BiasValue = nx.get_node_attributes(self.WG, "bias")
        acts = np.random.choice(self.biasRange, size=(1,self.WG.number_of_nodes()-NPL[0]))
        acts = np.insert(np.zeros(NPL[0]),2,acts[0])
        BiasValue.update(zip(BiasValue,acts)) 
        nx.set_node_attributes(self.WG,BiasValue,"bias")
        #print(nx.get_node_attributes(self.WG, "bias"))
        
        #random weights
        weightsValues = nx.get_edge_attributes(self.WG, "weight")
        acts = np.random.choice(self.weightsRange, size=(1,self.WG.number_of_edges()))
        weightsValues.update(zip(weightsValues,acts[0])) 
        nx.set_edge_attributes(self.WG,weightsValues,"weight")
        #print(nx.get_edge_attributes(self.WG, "weight"))
        
    def predict(self,x):
        A=nx.adjacency_matrix(self.WG)
        A=A.todense()
        A=np.c_[A,np.zeros(len(A))]

        self.biases = nx.get_node_attributes(self.WG,"bias")
        self.biases = list(self.biases.values())


        x=x.transpose()


        numRow = 0
        numColl = self.NeuronsperLayer[0]

        #take the respective weights of the layers in the neighborhood matrix and propagate forward
        for idx,layer in enumerate(self.NeuronsperLayer):
            if(idx == len(self.NeuronsperLayer)-1):
                break 
            #get weights out of A
            weightMat= np.array(A[numRow:np.sum(self.NeuronsperLayer[:idx+1]),numColl: numColl+self.NeuronsperLayer[idx+1]])
            #get activationfunction of layer
            acts= self.getActivation()[self.StartNeuronsPerLayer[idx+1]]
            numRow = np.sum(self.NeuronsperLayer[:idx+1])
            bias = self.biases[numRow:np.sum(self.NeuronsperLayer[:idx+2])]
            numColl += self.NeuronsperLayer[idx+1]
            x = np.dot(x,weightMat) +bias
            x = acts(x)
        return(x)