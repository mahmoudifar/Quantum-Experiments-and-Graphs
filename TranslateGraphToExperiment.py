#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sympy as sp 
import pandas as pd 
import itertools
import csv
from numpy.random import choice
import random 
import json
from sympy import sqrt,pi,I
import ast
from collections import Counter
from functools import reduce


# In[ ]:


def get_num_label(labels):
    num_to_label = dict((num, label) for num, label in enumerate(labels))
    return num_to_label

def encoded_label(nums,labels ):# for transform num to alphabet
    encoded_labels =[labels[num] for num in nums]
    return encoded_labels

def grouper(n, iterable):
    args = [iter(iterable)] * n
    return list(zip(*args))

def SetupToStr(setup):
    yyy ='XXX'
    for element in range(len(setup)-1,-1,-1):
        yyy = yyy.replace('XXX', setup[element])
    return yyy


# In[ ]:


#define optical devices (bs , pbs , hwp , spdc , phase shifter , oamhologram, reflection, absorber) 
Paths = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
colors = ['dodgerblue', 'firebrick', 'limegreen', 'darkorange', 'purple', 'yellow', 'cyan']
a, b, c, d, e, f, g, h, i, j, k = map(sp.IndexedBase,Paths)
zero=sp.Symbol('zero') 
theta, alpha , phi, beta, gamma, eta, ommega = sp.symbols(' theta alpha phi beta  gamma eta , ommega',integer=True )
p, p1, p2 = map(sp.IndexedBase,['p', 'p1', 'p2'])
l,l1, l2, l3, l4, l5, l6, l7, l8,l9 , l10, P, r , t, coeff  =map(sp.Wild, ['l','l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8','l9','l10', 'P', 'r', 't', 'coeff '])
a0 ,a1, a2 , a3, a4 , a5 = sp.symbols('a:6', cls =sp.IndexedBase )
b0 ,b1, b2 , b3, b4 , b5 = sp.symbols('b:6', cls =sp.IndexedBase )
c0 ,c1, c2 , c3, c4 , c5= sp.symbols('c:6', cls =sp.IndexedBase )
d0 ,d1, d2 , d3, d4 , d5 = sp.symbols('d:6', cls =sp.IndexedBase )
e0 ,e1, e2 , e3, e4 , e5 = sp.symbols('e:6', cls =sp.IndexedBase )
f0 ,f1, f2 , f3, f4 , f5 = sp.symbols('f:6', cls =sp.IndexedBase )
# H -> 0 V -> 1
# n = 1 , dim=2 -> HWP: Cyclic_Transformation in 2 dimention 
dim =  [l1 , l2 ,l3 , l4, l5 , l6 , l7 , l8 , l9 , l10]
def HWP(psi,p,n=1,dim=2): 
    psi=psi.replace(p[l],lambda l: p[np.mod((l+n),d)])
    return psi

def Absorber(psi, p):
    psi = psi.replace(p[l], 0)
    return psi

def OAMHolo(psi , p , n):
    psi = psi.replace(p[l], p[l+n])
    return psi 

def BS_Fun(psi, p1, p2):
    if psi.base == p1:
        psi = psi.replace(p1[l], 1/sqrt(2)*(p2[l]+I*p1[l]))
    elif psi.base == p2:
        psi = psi.replace(p2[l], 1/sqrt(2)*(p1[l]+I*p2[l]))
    return psi

def BS(psi, p1, p2 ):
    expr0 = list(psi.expr_free_symbols) 
    phi = []
    psi1 = []
    for ii in expr0:
        if type(ii)==sp.tensor.indexed.Indexed:
             phi.append(ii)
    for phi0 in phi:
        if phi0.base == p1 or phi0.base == p2:
            psi1.append(phi0)
    if len(psi1) ==0:
        psi = psi      
    elif len(psi1)==1:
        psi = sp.expand(psi.xreplace({psi1[0]: BS_Fun(psi1[0], p1, p2)}))
    elif len(psi1)==2:    
        psi = sp.expand(psi.xreplace({psi1[0]: BS_Fun(psi1[0], p1, p2),psi1[1]: BS_Fun(psi1[1], p1, p2)}))
    return psi

def SPDC(psi,p1,p2,l1,l2):
    psi = psi + p1[l1]*p2[l2]
    return psi

def PBS_Fun(psi,a, b):
    if psi.base == a:
        psi = psi.replace(a[l],lambda l: a[l] if l==1 else b[l])
    elif psi.base == b:
        psi = psi.replace(b[l],lambda l: b[l] if l==1 else a[l])
    return psi
        
def PBS(psi, p1, p2):
    expr0 = list(psi.expr_free_symbols) 
    phi = []
    psi1 = []
    for ii in expr0:
        if type(ii)==sp.tensor.indexed.Indexed:
             phi.append(ii)
    for phi0 in phi:
        if phi0.base == p1 or phi0.base == p2:
            psi1.append(phi0)
    if len(psi1) ==0:
        psi = psi       
    elif len(psi1)==1:
        psi = sp.expand(psi.xreplace({psi1[0]: PBS_Fun(psi1[0], p1, p2)}))
    elif len(psi1)==2:    
        psi = sp.expand(psi.xreplace({psi1[0]: PBS_Fun(psi1[0], p1, p2),psi1[1]: PBS_Fun(psi1[1], p1, p2)}))
    return psi
 
def Phase_Shifter(psi, p, phi):
    psi = psi.replace(p[l], sp.exp(I*l*phi)*p[l])
    return(psi)


# In[ ]:


#define post selection
def post_select (psi, dimm, ns = []):
    expr = list(psi.expr_free_symbols) 
    base = []
    for ii in expr:
        if type(ii)==sp.tensor.indexed.Indexed:
             base.append(ii.base)
    path = list(set(base))
    path = [x for x in path if x not in ns]
    dim = [i for i in (range(len(path)))]
    dim = encoded_label(dim, dimm)
    phi =list(zip(path, dim))
    PHI = [phi[i][0][phi[i][1]] for i in range(len(phi))]
    expr1 = reduce(lambda x, y: x*y, PHI)
    dictadd = sp.collect(psi, [expr1], evaluate=False)
    term = list(dictadd.keys())
    value = list(dictadd.values())
    for tt in range(len(term)):
        if term[tt] == 1:
            value[tt] = 0
    select = list(zip(term,value))
    selection = [select[i][0]*select[i][1] for i in range(len(select))]
    final_state = sp.expand(sum(selection ))
    return(final_state)


# In[ ]:


# Graph to Entanglement by path identity
Graph = {(0, 1 , 0, 0 ): 1,
         (0, 1, 1, 1): 1,
         (1, 2 , 1 , 1 ): 1,
         (2, 3, 0 , 0): 1,
         (0, 3 , 1 , 0): 1,
         }   
def Graph_to_EbPI(Graph):
    global Paths
    global dim
    dictt = dict()
    GraphEdges = [grouper(2,i)[0] for i in list(Graph.keys())]
    GraphEdgesAlphabet = [encoded_label(path,get_num_label(Paths))for path in GraphEdges]
    Dimension  = [grouper(2,i)[1] for i in list(Graph.keys())]
    dd = len(np.unique(list(itertools.chain(*Dimension ))))
    SetupList = []
    for pp in range(len(Graph)):
        SetupList.append("SPDC(XXX,"+GraphEdgesAlphabet[pp][0]+","+GraphEdgesAlphabet[pp][1]                     +","+str(Dimension[pp][0])+","+str(Dimension[pp][1])+")")
    setup = SetupToStr(SetupList)
    dictt['Experiment'] = SetupList
    dictt['SetupLength'] = len(SetupList)
    dictt['OutputState'] = post_select(sp.expand((eval(setup.replace('XXX', str(0))))**dd), dim)
    return dictt

GraphtoEbPI = Graph_to_EbPI(Graph)
print(GraphtoEbPI )


# In[ ]:


#Graph to path-encoding (for on-chip) 
def Graph_to_PathEn(graph):
    global Paths
    dictt = {}
    GraphEdges = [grouper(2,i)[0] for i in list(Graph.keys())]
    GraphEdgesAlphabet = [encoded_label(path,get_num_label(Paths))for path in GraphEdges]
    Dimension  = [grouper(2,i)[1] for i in list(Graph.keys())]
    SetupList = []
    for pp in range(len(Graph)):
        SetupList.append("SPDC(XXX,"+GraphEdgesAlphabet[pp][0]+str(pp)+","+GraphEdgesAlphabet[pp][1]+str(pp)                 +","+str(Dimension[pp][0])+","+str(Dimension[pp][1])+")")
    AllPath= []
    AllDim = []
    for pp in range(len(Graph)):
        AllPath.append(str(GraphEdgesAlphabet[pp][0])+str(pp))
        AllPath.append(str(GraphEdgesAlphabet[pp][1])+str(pp))
        AllDim.append(str(Dimension[pp][0]))
        AllDim.append(str(Dimension[pp][1]))
    PossiblePath =(list(itertools.combinations(AllPath,2)))
    PossibleDim = (list(itertools.combinations(AllDim ,2)))    
    combine= list(zip(PossiblePath,PossibleDim))
    combination = [combine[i][0]+combine[i][1] for i in range(len(combine))]
    for pd in range(len(combination)):
        if combination[pd][0][0]== combination[pd][1][0] and combination[pd][2]==combination[pd][3]:
            SetupList.append("BS(XXX,"+combination[pd][0]+","+combination[pd][1]+")")
            SetupList.append("Absorber(XXX,"+combination[pd][1]+")") 
    setup = SetupToStr(SetupList)
    dictt['Experiment'] = SetupList
    dictt['SetupLength'] = len(SetupList)
    dictt['OutputState']= sp.expand(eval(setup.replace('XXX', str(0))))
    return dictt
GraphtoPathEn = Graph_to_PathEn(Graph)
print(GraphtoPathEn)


# In[ ]:


#Graph to polarisation-encoding (for bulk optics)
def Graph_to_PolEN(expr):
    dictt ={}
    SetupList = expr['Experiment']
    psi = expr['OutputState']
    ss = list(psi.expr_free_symbols)
    path = []
    dimension = []
    for ii in ss:
        if type(ii)==sp.tensor.indexed.Indexed:
            path.append(str(ii.base))
            dimension.append(str(ii.indices[0]))
    PossiblePath =(list(itertools.combinations(path,2)))
    PossibleDim = (list(itertools.combinations(dimension ,2)))  
    combine= list(zip(PossiblePath,PossibleDim))
    combination = [combine[i][0]+combine[i][1] for i in range(len(combine))]
    for pd in range(len(combination)):
        if combination[pd][0][0]== combination[pd][1][0] and combination[pd][2]!=combination[pd][3]:
            SetupList.append("PBS(XXX,"+combination[pd][0]+","+combination[pd][1]+")")
    setup = SetupToStr(SetupList)
    dictt['Experiment'] = SetupList
    dictt['SetupLength'] = len(SetupList)
    dictt['OutputState']= sp.expand(eval(setup.replace('XXX', str(0))))
    return dictt   
GraphtoPolEN = Graph_to_PolEN(GraphtoPathEn)
print(GraphtoPolEN)


# In[ ]:




