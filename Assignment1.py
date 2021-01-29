#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assignment 1
Due Friday, 29 January 2021
@author: Bruno Zecchi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
pd.set_option('display.max_columns',30)
data = pd.read_table("/Users/mewtripiller/Desktop/Everything/IEMS 308/Assignment 1/Medicare_Provider_Util_Payment_PUF_CY2018/Medicare_PUP.txt")
data = data.drop(0)


import sys
sys.stdout = open("c:\\goat.txt", "w")

#Take a quick look
print(data.head(10))
print(data.columns)
print(data.describe())


##  SUBSET SELECTION
datD = data[data["hcpcs_drug_indicator"]=="Y"]


## DATA EXPLORATION
x1 = datD['average_submitted_chrg_amt']
x2 = datD['average_Medicare_allowed_amt']
display(plt.scatter(x1,x2))
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('Average Medicare Allowed Amount')


## K-MEANS CLUSTERING
X =pd.concat([x1,x2],join='outer',axis=1)

#repeat for different number of clusters, record inertia
km = KMeans(n_clusters = 3, random_state=19,n_init=50)
km.fit(X)
new_labels = km.labels_
plt.scatter(x1,x2,c = new_labels)
plt.title('Number of clusters = 3')
plt.xlabel('Average Submitted Charge Amount')
plt.ylabel('Average Medicare Allowed Amount')
print(km.inertia_)


## ANALYSIS

#split into different clusters
cl1 = datD[new_labels==0]
cl2 = datD[new_labels==1]
cl3 = datD[new_labels==2]

#look for unique characteristics of each
print(cl1.describe(include='all'))
print(cl2.describe(include='all'))
print(cl2.describe(include='all'))






