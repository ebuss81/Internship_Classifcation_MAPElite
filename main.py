
'''
nn.DG is the parent graph, which means that no neurons are added or deleted in this graph. 
Only the weights and the bias are changed here.


The existence of neurons and connections are only changed in the working graph (WG). 
If a neuron or a connection is added in the WG, it can be copied from the parent graph. 


maxcc (connection cost) was tested with a fully connected layer with max weights
'''

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#from Parameter import *
from mutations import *
from Graph import Graph
from MAPElite import MapElite
from connectioncost import *
import activations as activations
from parameters import *
#TODO plot results

def loadData(filePath):
    fileData = np.genfromtxt(filePath, dtype=None)
    data = np.array([fileData["f2"], fileData["f3"]])
    result = np.array([fileData["f1"]])

    return data, result
  
def evaluate2(prediction):
    return (result == prediction.transpose()).sum() / len(data[0]),

def plotNN(nn, best):
    # Plot Neural Network output as colored plane
    xvalues = np.arange(data[0].min() - 0.1, data[0].max() + 0.1, 0.005)
    yvalues = np.arange(data[1].min() - 0.1, data[1].max() + 0.1, 0.005)
    icoords, jcoords = np.meshgrid(xvalues, yvalues)
    testdata = np.array([icoords.flatten(), jcoords.flatten()])
    Z = nn.predict(testdata).reshape(icoords.shape)
    plt.pcolormesh(icoords, jcoords, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))

def plottNNv2(nn):
    # Plot Neural Network output as colored plane
    xvalues = np.arange(data[0].min() - 0.1, data[0].max() + 0.1, 0.005)
    yvalues = np.arange(data[1].min() - 0.1, data[1].max() + 0.1, 0.005)
    icoords, jcoords = np.meshgrid(xvalues, yvalues)
    testdata = np.array([icoords.flatten(), jcoords.flatten()])
    Z = nn.predict(testdata).reshape(icoords.shape)
    plt.pcolormesh(icoords, jcoords, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
    

def plotResults(datas, results, predictions):
    for i in range(len(predictions)):
        data = datas[:, i]
        result = results[0][i]
        prediction = predictions[i]

        marker = "o" if result == 0 else "s"
        color = "r" if result == 0 else "b"
        faceColor = color if result != prediction else "w"
        plt.scatter(*data, marker=marker, facecolor=faceColor, edgecolors=color)

    plt.xlabel("x")
    plt.ylabel("y")


def initialzePopulation(numIter):
    for i in range(numIter):
        print(i)
        nn.randomIndi()
        fit = evaluate2(nn.predict(data))
        modularity = nn.getCommunitiesModularity()
        ccost = getCost(nn.WG)
        #print("mod: ", modularity, " cc: ",ccost)
        try:
            ME.archiv(modularity,ccost, nn.WG,fit)
            ME.insertIndividuum(modularity,ccost,fit)
        except:
            print("Somethin went wron in initialzePopulation")            
    ME.plotGrid()
    ME.saveInFile(0)

def run(numIter):
    ME.readFile(1)
    for i in range(numIter):
        print("Iteration:",i)
        #if(i%1000 == 0):
        #    ME.plotGrid()
        nn.WG = ME.getRandomIndi()

        activeEdges = nx.get_edge_attributes(nn.WG,"active")
        nx.set_edge_attributes(nn.DG, activeEdges, "active")

        WightEdges = nx.get_edge_attributes(nn.WG,"weight")
        nx.set_edge_attributes(nn.DG, WightEdges, "weight")

        biasNodes = nx.get_node_attributes(nn.WG,"bias")
        nx.set_node_attributes(nn.DG, biasNodes, "bias")


        nn.WG,nn.DG = changeBias(nn.WG,BR, nn.DG)
        nn.DG,edge = addEdge(nn.WG,nn.DG)
        if edge != 0:
            nn.addEdgeFromParent(edge)
        nn.WG, nn.DG = deleteEdge(nn.WG, nn.DG) 
        nn.removeEdge()

        nn.WG, nn.DG = changeWeights(nn.WG,WR, nn.DG)   
        modularity = 0
        try:
            modularity = nn.getCommunitiesModularity() 
        except:
            print("no mudularity, becouse no edges or nodes")  

        fit = evaluate2(nn.predict(data))
        ccost = getCost(nn.WG)
        #print("Mod: ", modularity," ccost: ", ccost)
        try:
            ME.archiv(modularity,ccost, nn.WG,fit)
            ME.insertIndividuum(modularity,ccost,fit)
        except:
            print("not iserted, out of range, correct normalization?")
            
        #reset activity of DG to false, to overwrite it later with TRUE after mutatinos
        activeEdges = nx.get_edge_attributes(nn.DG,"active")
        setactivieEdges = [False, False, False, False, False, False, False, False, False,False,False,False]
        activeEdges.update(zip(activeEdges,setactivieEdges))
        nx.set_edge_attributes(nn.DG, activeEdges, "active")
    ME.plotGrid()
    ME.saveInFile(1)



def main():
    global nn, data, result, ME

    dataPaths = ["/home/ed/Dokumente/Codes/Classification_Network_Deap/data2.txt"]
    titles = ["Neural Network Classification"]
    data, result = loadData(dataPaths[0])
    
    #build Graphs
    DG = nx.DiGraph()

    #setlayers
    DG.add_nodes_from([1, 2, 3, 4], ActFunc = activations.noFunc, active = True, pos = (0,0),bias=0)
    DG.add_nodes_from([5, 6, 7, 8], ActFunc = activations.modSigmoid, active = True, pos = (0,0),bias=1)
    DG.add_nodes_from([9, 10, 11,12], ActFunc = activations.modSign, active = True, pos = (0,0),bias=1)
    #set edges
    DG.add_edges_from([(1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 7), (2, 8),(3, 5), (3, 6), (3, 7), (3, 8),(4, 5), (4, 6), (4, 7), (4, 8)], weight = 10, active = False)
    DG.add_edges_from([(5, 9), (5, 10), (5, 11), (5, 12),(6, 9), (6, 10), (6, 11), (6, 12),(7, 9), (7, 10), (7, 11), (7, 12),(8, 9), (8, 10), (8, 11), (8, 12)], weight = 10, active = False)    
    pos= [None] *12   

    nn = Graph()
    nn.DG = DG  
    nn.WG = DG.copy()
    
    c=0 # set positions of neurons
    for i in range(NUMLAYER):
        for j in range(NUMNEURONS):
            pos[c]= np.array([j,i])
            c+=1
    nn.setNodeAttribute(pos,"pos")


    #set active nodes, especially remove not needed In- and Outputneurons
    activeNodes = nx.get_node_attributes(nn.WG, "active")
    setactivieNodes = [True, False, False, True, True, True, True, True,False, True, False, False]
    activeNodes.update(zip(activeNodes,setactivieNodes))
    nx.set_node_attributes(nn.WG, activeNodes, "active")
    nx.set_node_attributes(nn.DG, activeNodes, "active")
    nn.removeNodes()
    nn.removeParentNodes()

    #get biases for prediction
    biases = nx.get_node_attributes(nn.WG,"bias")
    nn.biases = list(biases.values())
    
    #size = np.array([10,10])
    size = np.array([50,50])
    ME = MapElite(size)


    #initialzePopulation(200)
    #run(15000)

    #Analyze
    '''
    ME.readFile(1)
    nn.WG = ME.getBest(1)
    #nn.printGraph()
    ME.plotGrid()
    modularity = nn.getCommunitiesModularity()
    ccost = getCost(nn.WG)
    print("Fit :",evaluate2(nn.predict(data)), " Mod: ", modularity," ccost: ", ccost)
    print(nx.get_edge_attributes(nn.WG,"weight"))
    print(nx.get_node_attributes(nn.WG,"bias"))
    plotNN(nn, nn.WG)
    plotResults(data, result, nn.predict(data))
    plt.show()
    nn.printGraph()
    '''
    '''
    ME.readFile(1)
    nn.WG = ME.getIndi(5,5)
    print(evaluate2(nn.predict(data)))
    nn.printGraph()
    ME.plotGrid()
    '''
    
if __name__ == "__main__":
    main()
