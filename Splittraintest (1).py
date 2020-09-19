#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import re
import hanzidentifier
import io
import csv
from tqdm import tqdm, trange
df = pd.read_csv(r"Dataset.txt", encoding="UTF-8", sep = "\t",quoting=csv.QUOTE_NONE, error_bad_lines=False).fillna(method="ffill")
dat = df.values.tolist()
print(len(dat))


# In[11]:


#Choose g% testing data
e = input("Enter percentage of training data : ") 
e = int(e)
e = e/100
x = int(e*len(dat))
print(x)
testset1 = []
import random
i = 0
while i<x:
      random_item_from_list = random.choice(dat)
      testset1.append(random_item_from_list)
      dat.remove(random_item_from_list)
      i+=1
print("Length of trainset: " + str(len(dat)))
print("Length of testset: " + str(len(testset1)))


# In[12]:


i = 0
f2 = r"Testset.txt"
import io
with io.open(f2, "a", encoding="utf-8") as f:
        f.write("Query"  + "\t" + "Tags"  + "\n")
while i<len(testset1):
    with io.open(f2, "a", encoding="utf-8") as f:
        f.write(str(testset1[i][0]) + "\t"  + str(testset1[i][2])  + "\n")
    i+=1


# In[13]:


i = 0
f2 = r"Trainset.txt"
import io
with io.open(f2, "a", encoding="utf-8") as f:
        f.write("Query" + "\t" + "Tags"  + "\n")
while i<len(dat):
    with io.open(f2, "a", encoding="utf-8") as f:
        f.write(str(dat[i][0]) + "\t" + str(dat[i][2]) + "\n")
    i+=1

