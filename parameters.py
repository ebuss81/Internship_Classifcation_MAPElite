import numpy as np

## used in graph.py and main.py

NPL = np.array([2,4,1])     #   NeuronsperLayer, to iterate in prediction function
NBL =  5                    #   NumBiasedLayers, Here only in OutputLayer, important to get Biased weight out of genome
SNPL = np.array([1,5,10])   #   StartNeuronsPerLayer, only to get activiationsfunction of layers

WR = np.arange(-10,10.1,0.1) #  weights range
BR = np.arange(-10,10.1,0.1) #  bias range

# next parameter for parent graph!
NUMLAYER = 3
NUMNEURONS = 4

# to normalize Connection Cost
# maxcc (connection cost) was tested with a fully connected layer with max weights in Mod_Cost_test.py
MAXCC = 33.2