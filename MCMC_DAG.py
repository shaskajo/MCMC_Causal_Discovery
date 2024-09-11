# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:08:44 2024

@author: jshas
"""

import numpy as np
import random
from scipy.linalg import expm

def generate(sig, power, n ,d):
    X = np.random.normal(0,sig,(n,d))
    Triu = np.triu(np.ones((d,d)),1)
    A = np.random.binomial(1,.5,(d,d)) * Triu
    P = np.zeros((d,d))
    pi = np.random.permutation(d)
    for l in range(d): P[l][pi[l]] = 1
    A = P.T@A@P
    Asym = A+A.T
    B = np.random.random((d,d))
    B[B<.5] = -1
    B[B>=.5] = 1
    #REMEMBER TO UNCOMMENT THIS
    #A *= B
    A *= power
    X = (np.linalg.inv(np.eye(d)-A)@X.T).T
    return X, A

def Cov(X):
    Kxx = (X.T) @ X
    Kxx /= len(X[:,0])
    return Kxx

def generate_neighbors(A, d):
    neighbors = []
    for i in range(d):
        for j in range(i):
            A0 = np.copy(A)
            A1 = np.copy(A)
            if A[i][j] == 1:
                #edge reversal
                A0[i][j]=0 
                A0[j][i]=1
                #edge deletion
                A1[i][j]=0
            elif A[i][j] == 0 and A[j][i] == 0:
                #edge additions
                A0[i][j]=1
                A1[j][i]=1
            elif A[i][j] == 0 and A[j][i] == 1:
                #edge reversal
                A0[i][j]=1
                A0[j][i]=0
                #edge deletion
                A1[j][i]=0
            #Determine if matrices represent DAGs
            if np.trace(expm(A0*A0)) == d: neighbors.append(A0)
            if np.trace(expm(A1*A1)) == d: neighbors.append(A1)
    return neighbors

def get_next_sample(A, lik_A, d, sig, Kx):
    neighbors = generate_neighbors(A, d)
    A_prime = random.choice(neighbors)
    I = np.eye(d)
    lik_prime = (np.trace((I-A_prime).T @ (I-A_prime) @ Kx))
    r_thresh = -(n/2*sig) * (lik_prime - lik_A)
    u = np.random.uniform()
    if np.log(u) <= r_thresh:
        return A_prime, lik_prime
    return A, lik_A

def MCMC(X, d, sig):
    Kx = Cov(X)
    #TODO make the graph generation its own function
    Triu = np.triu(np.ones((d,d)),1)
    A = np.random.binomial(1,.5,(d,d)) * Triu
    P = np.zeros((d,d))
    pi = np.random.permutation(d)
    for l in range(d): P[l][pi[l]] = 1
    A = P.T@A@P
    I = np.eye(d)
    lik_A = (np.trace((I-A).T @ (I-A) @ Kx))
    A_key = tuple(map(tuple, A))
    posterior = {A_key: 1}
    Prob_12 = 0
    N = 100000
    for i in range(N):
        A, lik_A = get_next_sample(A, lik_A, d, sig, Kx)
        if A[1,2]==1 or A[2,1] == 1: Prob_12 += 1
        A_key = tuple(map(tuple, A))
        cur_samples = posterior.get(A_key,0)
        posterior[A_key] = cur_samples + 1
    for graph in posterior:
        posterior[graph] /= N
    return posterior, Prob_12/N
    



sig = 1
power = 1
n = 10
d = 3
X, A = generate(sig, power, n, d)
post, P12 = MCMC(X,d,sig)