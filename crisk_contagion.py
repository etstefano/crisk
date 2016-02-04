# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 06:38:31 2016

main script for crisk 

@author: stef
"""

import os
import itertools

import numpy as np
import pandas as pd

path = "/home/Stef/crisk/"

###############################################################################
#import data and list of impact matrices
data = pd.read_csv(path + "data.csv")

im_list = list()
for file in os.listdir(path):
    if "impact_mat" in file:
        im_list.append(file)

###############################################################################
# model parameters
shocks = [float(x)/1000 for x in range(0,55,5)]
#scenarios = list(itertools.combinations_with_replacement(shocks, 4))
scenarios = list(itertools.product(shocks, repeat = 2))




equity = data.Equity.values
carbon_sectors = data[["ExtrPetr","ElecGas"]].values

###############################################################################
#function that computes contagion
def compute_contagion(phi, equity, impact_mat):
    #create matrix of states that tells us if the bank has already been hit or not
    state_mat = np.zeros([183,1])
    #create matrix of final ratio of equity losses
    losses = np.zeros([183,1])
    #get indices of banks hit by initial shock
    seeds = np.nonzero(phi)[0]
    
    #go on with the contagion until there are no seed nodes
    losses[seeds,0] = phi[seeds] #register initial losses 
    state_mat[seeds] = 1
    
    #while there are nodes active, propagate shock
    while len(seeds) > 0:
        neigh_list = list()
        for seed in seeds:
            #find active nodes
            active = np.where(state_mat < 2)[0]
            #find neighbors who are active
            neigh = np.intersect1d(np.nonzero(impact_mat[seed,:]), active)
            neigh_list.append(neigh.tolist())
            #for each neighbor, unload the stress
            losses[neigh] = losses[neigh] + np.reshape(np.transpose(np.multiply(losses[seed], impact_mat[seed, neigh])), 
                                                        [len(neigh),1])
            losses[losses > 1] = 1
            #make inactive those who have spread shocks
            state_mat[seed] = 2
            
        #change state of those who got shock and put them as seeds for next propagation
        neigh = list(set([item for sublist in neigh_list for item in sublist]))
        state_mat[neigh] = 1
        seeds = neigh
            
    
    return np.reshape(losses,[state_mat.shape[0]])

###############################################################################
#write csv table with all the results
col_names = ['names', 'ExtrPetr','ElecGas'] + [str(n) for n in range(1,101)]


with open(path + "crisk_table3.csv", "a") as fu:
    for name in col_names:
        fu.write(name + ",")
    fu.write("\n")

    n = 1
    #for each scenario, evaluate shocks and store them 
    for sc in scenarios[1:]:
        print str(n)
        #initial shock, phi
        shock_value = np.multiply(np.array(sc),carbon_sectors)
        phi = np.divide(np.sum(shock_value, axis = 1), equity)
        phi[phi>1] = 1 #maximus losses is the amount of equity a bank owns
    
        #initialize stress matrix, number of banks x network realizations
        stress_mat = np.zeros([183,100])
        
        #apply scenario to each net reconstruction
        for f in range(0,len(im_list)):
            impact_mat = np.genfromtxt(path + im_list[f],delimiter=',')
            
            stress_mat[:,f] = compute_contagion(phi, equity, impact_mat)
            
        stress_mat[stress_mat > 1] = 1

        for row in range(0,stress_mat.shape[0]):
            fu.write(data.Name.ix[row].replace(",", " ") + ",") #write name of bank
            for ss in sc:
                fu.write(str(ss) + ",")
            for l in range(0,stress_mat.shape[1]):
                fu.write(str(stress_mat[row,l]) + ",")
            fu.write("\n")
        
        n = n+1
