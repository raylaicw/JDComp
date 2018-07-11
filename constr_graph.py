# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 03:26:45 2018

@author: keith_work
"""

import numpy as np


########### G1 #######################
def k_th_veh(A,N,F,k,K): #k-th vehicle of K
    if k==1:
        dummy3=np.tile(np.zeros(len(A)),K-1)
        A=np.concatenate((A,dummy3))
    else:
        dummy3=np.tile(np.zeros((N+F+2)*(N+F+2)+N+F+2+N),k-1)
        dummy4=np.tile(np.zeros((N+F+2)*(N+F+2)+N+F+2+N),K-k)
        A=np.concatenate((dummy3,A,dummy4))
    return(A)

def fcn_cscon1(N,F,k,K,i): #k-th vehicle of K; at customer i
    sub_key = np.ones(N)
    sub_key[i-1] = 0 
    key = np.concatenate((np.zeros(1),sub_key,np.zeros(F+1)))
    dummy1=np.tile(np.zeros(N+F+2),i)
    dummy2=np.tile(np.zeros(N+F+2),N+F+2-i-1)
    A = np.concatenate((dummy1,key, dummy2,np.zeros(N+F+2+N)))
    A = k_th_veh(A,N,F,k,K)
    #A = np.tile(A,K)
    #print(A.shape)
    return(A)

def fcn_cscon2(N,F,k,K,j): #k-th vehicle of K; at customer i
    key = np.concatenate((np.zeros(j),np.ones(1),np.zeros(N+F+2-j-1)))
    key = np.tile(key,N)
    key[(j-1)*(N+F+2)+j]=0
    A = np.concatenate((np.zeros(N+F+2),key,np.zeros((N+F+2)*(F+1)+N+F+2+N)))
    A = k_th_veh(A,N,F,k,K)
    #print(A.shape)
    #A = np.tile(A,K)
    return(A)

def fcn_cscon1_v2(N,F,k,K,i):
    A = sum_x(N,F,K,range(1,N+1),[i],[k]).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)
    
def fcn_cscon2_v2(N,F,k,K,i):
    A = sum_x(N,F,K,[i],range(1,N+1),[k]).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)
    
def sum_x(N,F,K,rng_i,rng_j,rng_k):
    A = np.zeros([K,N+F+2,N+F+2])
    for k in rng_k:
        for i in rng_i:
            for j in rng_j:
                A[k,j,i] = 1
    return A
        
def connect_cs(N,F,K): ## equality constraints
    j=0
    B = np.empty([(N*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])    
    for i in range(N):
        for k in range(K):
            B[j,:]=fcn_cscon1_v2(N,F,k,K,i) - fcn_cscon2_v2(N,F,k,K,i)
            j+=1
    return(B)
    
def mayvisit(N,F,K): ## inequality that each i must be visited
    j=0
    B = np.empty([(N*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])
    for i in range(N):
        for k in range(K):
            B[j,:]=fcn_cscon1_v2(N,F,k,K,i)
            j+=1
    return(B)
    
########### End of G1 ################
#%%
########### G2 #######################
def fcn_cscon3(N,F,k,K,i):
    A = sum_x(N,F,K,[i],range(N+1,N+F+2),[k]).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)

def fcn_cscon4(N,F,k,K,i):
    A = sum_x(N,F,K,range(N+1,N+F+2),[i],[k]).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)

def G2(N,F,K):
    j=0
    B = np.empty([(N*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])    
    for i in range(N):
        for k in range(K):
            B[j,:]=fcn_cscon3(N,F,k,K,i) - fcn_cscon4(N,F,k,K,i)
            j+=1
    return(B)
########### End of G2 ################
#%%
################## G3 ################
def fcn_cscon5(N,F,K,i):
    A = sum_x(N,F,K,[i],range(1,N+F+2),range(0,K)).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)

def G3(N,F,K):
    B = np.empty([N,((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])
    j = 0   
    for i in range(N):
        B[j,:] = fcn_cscon5(N,F,K,i)
        j += 1
    return(B)
########## End of G3 #################
#%%   
################## G4 ################
def fcn_cscon6(N,F,K,i):
    A = sum_x(N,F,K,[i],range(1,N+F+2),range(0,K)).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)

def G4(N,F,K):
    B = np.empty([N,((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])
    j = 0    
    for i in range(N):
        B[j,:] = fcn_cscon6(N,F,K,i)
        j += 1
    return(B)
########## End of G4 #################
    
################## G5 ################
def fcn_cscon7(N,F,k,K,i):
    A = sum_x(N,F,K,[i],[0,N+F+1],[k]).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)

def G5(N,F,K):
    B = np.empty([(N*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])
    j = 0   
    for i in range(N):
        for k in range(K):
            B[j,:] = fcn_cscon7(N,F,k,K,i)
            j += 1
    #return(B.reshape((N*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K))
    return B
########## End of G5 #################
    
################## G6 ################
def fcn_cscon8(N,F,k,K,i,j):
    A = sum_x(N,F,K,[i],[j],[k]).reshape(-1)
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for a
    A = np.append(A,np.zeros((N+F+2)*K))    # Insert for y
    A = np.append(A,np.zeros(N*K))          # Insert for w
    return(A)

def G6(N,F,K):
    B = np.empty([(N+F+2)*(N+F+2)*K,((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])
    j = 0    
    for k in range(K):
        for i in range(N+F+2):
            for j in range(N+F+2):
                B[j,:] = fcn_cscon8(N,F,k,K,i,j)
                j += 1
    return(B)
########## End of G6 #################
    
################## T1 ################
def fcn_cscon9(N,F,k,K):
    A = np.zeros([K,(N+F+3),(N+F+3)])
    for i in range(N+1):
        for j in range(1,N+2):
            if not(i == j or (i==0 and j==N+1)):
                A[k,i,j] = 1
    return(A)

def T1(N,F,K):
    #B = np.empty([(N*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K])
    B = []    
    for i in range(N+F+2):
        for j in range(N+F+2):
            for k in range(K):
                B = np.append(B,fcn_cscon9(N,F,k,K,i,j))
    return(B.reshape(((N+F+2)*(N+F+2)*K),((N+F+2)*(N+F+2)+2*(N+F+2)+N)*K))
########## End of T1 #################
    
def constr_graph(N,F,K):
    A = connect_cs(N,F,K)
    A.vstack(mayvisit(N,F,K))
    print('G1')
    A.vstack(G2(N,F,K))
    print('G2')
    A.vstack(G3(N,F,K))
    print('G3')
    A.vstack(G4(N,F,K))
    print('G4')
    A.vstack(G5(N,F,K))
    print('G5')
    A.vstack(G6(N,F,K))
    print('G6')
    print(A)
    
constr_graph(30,5,10)