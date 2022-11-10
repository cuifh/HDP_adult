"""function for NPL posterior bootstrap sampler for logistic regression

Parameters
----------
B_postsamples : int 
    Number of posterior samples to generate

alph_conc : float
    Concentration parameter for DP prior

T_trunc: int > 0 
    Number of prior pseudo-samples from DP base measure for truncated sampling

y: array
    Observed classes

x: array
    Covariates

N_data: int
    Number of data points

D_covariate: int
    Dimension of covariates

a,b: floats
    Hyperparameter terms in Student t prior

gamma: float
    Loss scaling term for loss-NPL

n_cores: int
    Number of cores Joblib can parallelize over; set to -1 to use all cores
"""
import numpy as np
import scipy as sp
import pandas as pd
import copy
import time
import npl.maximise_logreg as mlr
from joblib import Parallel, delayed,dump, load
from tqdm import tqdm
import os


def bootstrap_logreg(B_postsamples,alph_conc,alpha_top_layer,T_trunc,y,x,N_data,D_covariate,a,b,gamma,n_cores = -1):   #Bootstrap posterior
    eps_dirichlet=10**(-100)
    
    
    K_set=len(N_data)
    
    y_tol=y
    x_tol=x
    
    y=y_tol[0]
    x=x_tol[0]
    
    for i in range(K_set-1):
        y=np.append(y,y_tol[i+1])
        x=np.concatenate([x,x_tol[i+1]],axis=0)
        
        
        
    
    if alph_conc!=0:
        
        alphas = np.concatenate((np.ones(sum(N_data)),(alph_conc/T_trunc)*np.ones(T_trunc)))
        beta_weights = np.random.dirichlet(alphas,B_postsamples)  
        y_prior,x_prior = mlr.sampleprior(x,sum(N_data),D_covariate,T_trunc,B_postsamples)
        alpha_top_layer_beta=alpha_top_layer*beta_weights
        
        Weights=[]
        Weights.append(np.repeat([np.concatenate((np.ones(N_data[0]), np.zeros(sum(N_data[1:])+T_trunc)))],B_postsamples,axis=0)+alpha_top_layer_beta)
        
        for i in range(K_set-1):
            Weights.append(np.repeat([np.concatenate((np.zeros(sum(N_data[0:(i+1)])),np.ones(N_data[i+1]), np.zeros(sum(N_data[(i+2):])+T_trunc)))],B_postsamples,axis=0)+alpha_top_layer_beta)
            
    else:
        
        beta_weights = np.random.dirichlet(np.ones(sum(N_data)),B_postsamples)
        y_prior = np.zeros(B_postsamples)
        x_prior = np.zeros(B_postsamples)
        alpha_top_layer_beta=alpha_top_layer*beta_weights
            
        Weights=[]
        Weights.append(np.repeat([np.concatenate((np.ones(N_data[0]), np.zeros(sum(N_data[1:]))))],B_postsamples,axis=0)+alpha_top_layer_beta)
        for i in range(K_set-1):
            Weights.append(np.repeat([np.concatenate((np.zeros(sum(N_data[0:(i+1)])),np.ones(N_data[i+1]), np.zeros(sum(N_data[(i+2):]))))],B_postsamples,axis=0)+alpha_top_layer_beta)
            
    
    weights=Weights
    
    for i in range(K_set):
        for b in range(B_postsamples):
            weights[i][b,:]=np.random.dirichlet(Weights[i][b,:]+eps_dirichlet,1)
            
    
    ll_bb=[]
    beta_bb=[]
    #import pdb
    #pdb.set_trace()
    for i in range(K_set):
        ll_bb.append( np.zeros(B_postsamples) )
        beta_bb.append( np.zeros((B_postsamples,D_covariate+1)))
    
    #pdb.set_trace()


    #Initialize RR-NPL with normal(0,1)
    R_restarts = 1
    beta_init = np.random.randn(R_restarts*B_postsamples,D_covariate+1)
    
    temp=[]
    
    #import pdb
    #pdb.set_trace()
    
    for j in range(K_set):
        
        temp_1 = Parallel(n_jobs=n_cores, backend= 'loky')(delayed(mlr.maximise)(y,x,y_prior[i],x_prior[i],sum(N_data),D_covariate,alph_conc,T_trunc,weights[j][i],beta_init[i*R_restarts:(i+1)*R_restarts],a,b,gamma[j],R_restarts) for i in tqdm(range(B_postsamples)))
        temp.append(temp_1)
    
    #pdb.set_trace()

    #Convert to numpy array
    
    beta_bb = np.zeros([K_set,B_postsamples])
    ll_bb = np.zeros([K_set,B_postsamples])
    
    #for j in range(K_set):
    #    for i in range(B_postsamples):
    #        aa, bb =temp[j] [i]
    #        beta_bb[j][i] = aa
    #        ll_bb[j][i] = bb
    
    
    
    
    #for i in range(B_postsamples):
    #    aa, bb =temp_1 [i]
    #    beta_bb_1[i] = aa
    #    ll_bb_1[i] = bb
        
        #pdb.set_trace()
        
    #    aa, bb =temp_2 [i]
    #    beta_bb_2[i] = aa
    #    ll_bb_2[i] = bb
        
        
        #beta_bb_1[i],ll_bb_1[i] = temp_1[i] 
        #beta_bb_2[i],ll_bb_2[i] = temp_2[i] 
    


    #return beta_bb, ll_bb   
    return temp
