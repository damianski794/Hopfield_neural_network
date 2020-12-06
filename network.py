# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

class HopfieldNetwork(object):      
    def train_weights(self, train_data, train_method='Hebb'):
        print("Start to train weights...")
        num_data =  len(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        #W = np.random.rand(self.num_neuron, self.num_neuron)
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)

        if train_method == 'Hebb':
            # Hebb rule
            for i in tqdm(range(num_data)):
                t = train_data[i] - rho
                W += np.outer(t, t)

        elif train_method == 'Oja':
            ######################
            # u = 0.01
            # V = np.dot(self.weight, input_data.T)
            # i = 0

            # for inp in input_data:
            #    v = V[:, i].reshape((n_features, 1))  # n_features is # of columns
            #    self.weight += (inp * v) - u * np.square(v) * self.weight
            #    i += 1
            #######################
            print(f'{train_data=}')
            u = 0.01
            V = np.dot(W, np.array(train_data).T)
            print(f'{np.array(train_data)=}')
            print(f'{np.array(train_data).shape=}')
            print(f'{V=}')
            print(f'{V.shape=}')
            i = 0

            for inp in tqdm(train_data):
                v = V[:, i].reshape((-1, 1))  # n_features is # of columns
                W += (inp * v) - u * np.square(v) * W
                i += 1
        else:
            raise ValueError(f'train method must be either Hebb or Oja. Value used -> {train_method}')

        print(f'{W=}')
        print(f'{W.shape=}')
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
    
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn
        
        # Copy to avoid call by reference 
        copied_data = np.copy(data)
        
        # Define predict list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted
    
    def _run(self, init_s):
        if self.asyn==False:
            """
            Synchronous update
            """
            # Compute initial state energy
            s = init_s

            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
        else:
            """
            Asynchronous update
            """
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.num_iter):
                for j in range(100):
                    # Select random neuron
                    idx = np.random.randint(0, self.num_neuron) 
                    # Update s
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                
                # Compute new state energy
                e_new = self.energy(s)
                
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
    
    
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()