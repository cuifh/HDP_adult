"""
Generating data

"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
from npl import bootstrap_logreg as bbl
import pickle

def gen_toy(N_data, beta_1, beta_2, seed):
    
    np.random.seed(seed)
    
    K_set=2
    
    X_1 = np.random.multivariate_normal([1,2], [[1,0],[0,1]], N_data)
    X_2 = np.random.multivariate_normal([1,2], [[1,0],[0,1]], N_data)
    
    Linear_part_1 = X_1@beta_1
    Linear_part_2 = X_2@beta_2
    
    eta_1 = 1/(1 + np.exp(-Linear_part_1))
    eta_2 = 1/(1 + np.exp(-Linear_part_2))
    
    y_1 = np.random.binomial(1,eta_1)
    y_2 = np.random.binomial(1,eta_2)
    
    gamma=1/N_data
    
    return X_1, X_2, y_1, y_2, gamma



def main(beta_1, beta_2, B_postsamples):
    n_iter=1

    T_trunc = 100
    a=1
    b = 1 #rate of gamma hyperprior
    
    alph_conc=0
    N_data=100
    D_data=2
    
    
    for i in range(n_iter):

        seed = 100+i
        
        np.random.seed(seed)
        x_1, x_2, y_1, y_2, gamma = gen_toy(N_data, beta_1, beta_2, seed)
        
        #y,x,alph_conc,gamma,N_data,D_data = load_data(dataset,seed)

        start= time.time()
        #carry out posterior bootstrap
        beta_bb, ll_b = bbl.bootstrap_logreg(B_postsamples,alph_conc,T_trunc,y_1,x_1,N_data,D_data,a,b,gamma)
        end = time.time()
        print ('Time elapsed = {}'.format(end - start))

        #convert to dataframe and save
        dict_bb = {'beta': beta_bb, 'll_b': ll_b, 'time': end-start}
        par_bb = pd.Series(data = dict_bb)

        #Polish
        if dataset == 'Polish':
            par_bb.to_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_pol_B{}_seed{}'.format(alph_conc,a,b,B_postsamples,seed))

        #Adult
        if dataset == 'Adult':
            par_bb.to_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_ad_B{}_seed{}'.format(alph_conc,a,b,B_postsamples,seed))

        #Arcene
        if dataset == 'Arcene':
            par_bb.to_pickle('./parameters/par_bb_logreg_c{}_a{}_b{}_gN_ar_B{}_seed{}'.format(alph_conc,a,b,B_postsamples,seed))

if __name__=='__main__':
    main([0,1],[1,2],100)
    