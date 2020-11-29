import numpy as np
import random
import matplotlib.pyplot as plt
from Graph import Graph
import pickle

from connectioncost import *

class MapElite:
    def __init__(self, gridsize = np.array([10,10]) ):
        self.size = gridsize
        self.Grid = np.zeros(self.size)
        self.archive  = [[None for i in range(self.size[0])] for j in range(self.size[1])] 
        self.nn = Graph()
        self.gss = 0.02    #gridstepsize, depending on resolution. for 10x10 in [0,1] gss =0.1 for 50x50 in [0,1] gss = 0.02


    def plotGrid(self):
        min_val, max_val = 0, 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(self.Grid,cmap=plt.cm.Blues,extent=[min_val, max_val, max_val, min_val], origin="upper")

        fig.colorbar(cax)
        cax.set_clim(0, 1)

        '''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                c = self.Grid.transpose()[i][j]
                ax.text((i+0.5)/self.size[0], (j+0.5)/self.size[1], str(c), va='center', ha='center')
        '''
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(max_val, min_val)
        ax.set_xticks(np.arange(10.)/10)
        ax.set_yticks(np.arange(10.)/10)
        plt.xlabel("ConnectionCost")
        plt.ylabel("Modularity")

        plt.show()

        
    def insertIndividuum(self, f1,f2,fitness):
        fitness = np.round(fitness[0],3)
        f1 = int(f1/self.gss)
        f2 = int(f2/self.gss)
        if self.Grid[f1,f2] < fitness:
            self.Grid[f1,f2] = fitness
            print("fit insert")


    def archiv (self, f1,f2,WG, fitness):
        fitness = np.round(fitness[0],3)
        f1 = int(f1/self.gss)
        f2 = int(f2/self.gss)
        #print("f1",f1,"f2",f2)
        if self.Grid[f1,f2] < fitness:
            self.archive[f1][f2] = WG
            print("graph insered")

    def getArchiv(self):
        return self.archive

    def getIndi(self,f1,f2):
        try:
            indi = self.archive[f1][f2]  
            return indi.copy()
        except:
            print("no Graph in Slot")
            return None

    def largest_indices(ary, n):
        #"""Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def getBest(self,x):
        maxidx = np.argmax(self.Grid)
        print(maxidx)
        #print((-self.Grid.flatten()).argsort()[:4])
        maxidx = (-self.Grid.flatten()).argsort()[:10]
        print(maxidx)
        row = int(maxidx[x]/self.size[0])
        col = maxidx[x] % self.size [1]
        return self.archive[row][col].copy()


    def saveInFile(self,num):
        if num == 0:
            pickle_out = open("fit0.pickle","wb")
            pickle.dump(self.Grid,pickle_out)
            pickle_out.close()

            pickle_out1 = open("archiv0.pickle","wb")
            pickle.dump(self.archive,pickle_out1)
            pickle_out1.close()
        if num == 1:
            pickle_out = open("fit1.pickle","wb")
            pickle.dump(self.Grid,pickle_out)
            pickle_out.close()

            pickle_out1 = open("archiv1.pickle","wb")
            pickle.dump(self.archive,pickle_out1)
            pickle_out1.close()

    def readFile(self,num):
        if num == 0:
            pickle_in = open("fit0.pickle","rb")
            self.Grid = pickle.load(pickle_in)
            
            pickle_in1 = open("archiv0.pickle","rb")
            self.archive = pickle.load(pickle_in1)       
        if num == 1:
            pickle_in = open("fit1.pickle","rb")
            self.Grid = pickle.load(pickle_in)
            
            pickle_in1 = open("archiv1.pickle","rb")
            self.archive = pickle.load(pickle_in1)   

    def getRandomIndi(self):
        allIndis = np.array(np.where(self.Grid >0))
        allIndis = allIndis.transpose()
        randomIdx = allIndis[np.random.choice(allIndis.shape[0], 1, replace=False)]
        return self.archive [randomIdx[0][0]][randomIdx[0][1]].copy()

