#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: konstantinos
"""
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
        
class UnitCell:

    def __init__(self, N):
        """
        General class for unit cell on a square fixed-size grid
        N: dimension of the grid
        """
        self.N = N
        self.G = nx.Graph()
        self.storeFullConnectedLattice()

#---------------------------------------------------------------------------------------------------
        
    def storeFullConnectedLattice(self):
        """
        Creates a networkx graph with all possible nearest neighbor edges
        making sure that there are no edge crossings where nodes do not exist
        """
        # Initialize

        self.G.add_nodes_from(range(self.N**2))
        
        # Interior quad connectivity
        for i in range(self.N-1):
            for j in range(self.N-1):
                idxIJ = j*self.N+i
                self.G.add_edge(idxIJ, idxIJ+1)
                self.G.add_edge(idxIJ, idxIJ+self.N)

        # Edge connectivity
        for i in range(self.N-1):
            idxIJ = (self.N-1)*self.N+i
            self.G.add_edge(idxIJ, idxIJ+1)
        for j in range(self.N-1):
            idxIJ = j*self.N+self.N-1
            self.G.add_edge(idxIJ, idxIJ+self.N)
                
        # Interior cross connectivity
        for i in range(self.N-1):
            for j in range(self.N-1):
                if i%2==0 and j%2==0:
                    idxC1 = j*self.N+i
                    idxC2 = (j+1)*self.N+i+1
                    self.G.add_edge(idxC1,idxC2)
                elif i%2==1 and j%2==1:
                    idxC1 = j*self.N+i
                    idxC2 = (j+1)*self.N+i+1
                    self.G.add_edge(idxC1,idxC2)
                if i%2==0 and j%2==1:
                    idxC3 = j*self.N+i+1
                    idxC4 = (j+1)*self.N+i
                    self.G.add_edge(idxC3,idxC4)
                elif i%2==1 and j%2==0:
                    idxC3 = j*self.N+i+1
                    idxC4 = (j+1)*self.N+i
                    self.G.add_edge(idxC3,idxC4)

#---------------------------------------------------------------------------------------------------
        
    def checkConnectedness(self):
        """
        Returns true if graph is connected
        """
        return nx.is_connected(self.G)

#---------------------------------------------------------------------------------------------------
                
    def checkHangingBeams(self):
        """
        Returns false if graph has a hanging beam
        """
        minZ = min([d[1] for d in self.G.degree()])
        if minZ < 2:
            return False
        return True

#---------------------------------------------------------------------------------------------------

    def plot(self, reward, stiffness):
        """
        Plot unit cell
        """
        x = np.arange(self.N)
        y = np.arange(self.N)
        xG,yG = np.meshgrid(x,y)
        xG = xG.ravel()
        yG = yG.ravel()
        
        # Plot
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        A = nx.adjacency_matrix(self.G).todense()
        for idx1 in range(len(A)):
            for idx2 in range(len(A)):
                if A[idx1,idx2] == 0: continue
                ax.plot([xG[idx1],xG[idx2]],
                        [yG[idx1],yG[idx2]],'-o',
                    markersize=5, color='black',lw=1)
        plt.axis('off')
        plt.title(f'rew: {reward.item(): .3f}, {stiffness: .5f}')
        plt.show()

#---------------------------------------------------------------------------------------------------

    def save(self, filename):
        """
        Save to a mesh file for downstream FEM
        """
        x = np.arange(self.N)
        y = np.arange(self.N)
        xG,yG = np.meshgrid(x,y)
        xG = xG.ravel()
        yG = yG.ravel()
        
        # Write to file
        f = open(filename,'w')
        for xGi, yGi in zip(xG,yG):
            f.write('%.4f,%.4f\n' % (xGi, yGi)) 
        f.write('connectivity\n')
        for u,v in self.G.edges():
            f.write('%d,%d\n' % (u, v)) 
        f.close()

    # from Johannes
    def transform_to_connectivity(self, N):
        connect_ones = torch.zeros(N * N)
        connectivity = torch.diag(connect_ones)
        for u, v in self.G.edges():
            connectivity[u, v] = 1.0
            if u == v:
                connectivity[u, v] = 0.0

        for x in range(N * N):
            for y in range(N * N):
                 connectivity[y, x] = connectivity[x, y]
        return connectivity

    def removeEdge(self, col, row):
        self.G.remove_edge(col, row)

    def has_edge(self, u, v):
        return self.G.has_edge(u, v)

    def transform_to_triu(self, N):
        x = self.transform_to_connectivity(N)
        row_idx, col_idx = np.triu_indices(x.shape[1], k=1)
        row_idx = torch.LongTensor(row_idx)
        col_idx = torch.LongTensor(col_idx)
        x = x[row_idx, col_idx]
        return x

    def count_edges(self):
        return self.G.number_of_edges()

    def plotsave(self, pltname, reward, stiffness):
        """
        Plot unit cell
        """
        x = np.arange(self.N)
        y = np.arange(self.N)
        xG, yG = np.meshgrid(x, y)
        xG = xG.ravel()
        yG = yG.ravel()

        # Plot
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        A = nx.adjacency_matrix(self.G).todense()
        for idx1 in range(len(A)):
            for idx2 in range(len(A)):
                if A[idx1, idx2] == 0: continue
                ax.plot([xG[idx1], xG[idx2]],
                        [yG[idx1], yG[idx2]], '-o',
                        markersize=5, color='black', lw=1)
        plt.axis('off')
        plt.title(f'rew: {reward.item(): .3f}, {stiffness: .2f}')
        pltname = str(pltname)
        plt.savefig(pltname)

    # count edges in vertical direction
    def hor_count(self):
        count = 0
        for u, v in self.G.edges():
            if abs(u-v) == 1:
                count += 1
        return count

    def transform_to_bars(self, N):
        full_truss = UnitCell(N)
        full_truss_con = full_truss.transform_to_triu(N)
        non_zero_values = full_truss_con.nonzero()
        x = self.transform_to_triu(N)
        return x[non_zero_values].flatten()

    def edges(self):
        return self.G.edges()

    def bar_removed(self, a, N):
        W = UnitCell(N)
        # N*N nodes in unitcell
        connect_ones = torch.zeros(N * N)
        # N*N nodes = connectivity matrix
        connectivity = torch.diag(connect_ones)
        for u, v in W.edges():
            connectivity[u, v] = 1.0
            if u == v:
                connectivity[u, v] = 0.0
        count = 0
        a += 1
        # shooting with canons on birds
        for x in range(N * N):
            for y in range(N * N):
                if connectivity[x, y] == 1:
                    count += 1
                if (count - a) == 0:
                    self.G.remove_edge(x, y)
                    return self.transform_to_bars(N)
        print("action wrong error", a)
        return









