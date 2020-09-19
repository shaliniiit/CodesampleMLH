#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import hanzidentifier
import io
import csv
from tqdm import tqdm, trange
df = pd.read_csv(r"RightTrainGerman.txt", encoding="UTF-8", sep = "\t",quoting=csv.QUOTE_NONE, error_bad_lines=False).fillna(method="ffill")
dat = df.values.tolist()
print(len(dat))
final = [[[],[]] for y in range(len(dat))]
i = 1
c = 0
while i<len(dat):
    #print(i)
    dict = {}
    a = eval(dat[i][0])
    b = eval(dat[i][1])
    for k in b:    
        if b[k].lower().strip() not in a:
            continue
        if b[k].lower().strip() in a:
            dict[k] = a[b[k].lower().strip()]
    for k in dict:
        final[i][0].append(k)
        final[i][1].append(dict[k]) 
    #print(final[i])
    i+=1
final =  list(filter(None, final))
print("Final length : " + str(len(final)))
import hanzidentifier
final = [[[],[]] for y in range(len(dat))]
i = 0
while i<len(dat):
    print(i)
    dict = {}
    a = eval(dat[i][0])
    b = eval(dat[i][1])
    for k in b:    
        if b[k] not in a:
            continue
        if b[k] in a:
            dict[k] = a[b[k]]
    #print(dict)
    for k in dict:
        if hanzidentifier.has_chinese(k) and len(k)>1:
            for j in k:
                if j!="" and j!=" ":
                    final[i][0].append(j)
                    if dict[k] == "B-entity" and j==k[0]:
                        final[i][1].append(dict[k])
                        continue
                    if dict[k] == "B-entity" and j!=k[0]:
                        final[i][1].append("I-entity")
                        continue
                    if dict[k] == "B-action" and j==k[0]:
                        final[i][1].append(dict[k])
                        continue
                    if dict[k] == "B-action" and j!=k[0]:
                        final[i][1].append("I-action")
                        continue
                    else:
                        final[i][1].append(dict[k])
            continue
        else:
            final[i][0].append(k)
            final[i][1].append(dict[k])  
    i+=1
f2 = r"RightGermanDataset.txt"
i = 0
import io
while i<len(final):
    r1 = ""
    r2 = ""
    j = 0
    while j<len(final[i][0]):
        if final[i][0][j] == "" or final[i][0][j] == " ":
            j+=1
            continue
        r1 = r1 + final[i][0][j] + " "
        r2 = r2 + final[i][1][j] + " "
        j+=1
    r1.strip()
    r1.lstrip()
    r2.strip()
    r2.lstrip()
    if r1=="" or r2=="":
        c+=1
        i+=1
        continue
    with io.open(f2, "a", encoding="utf-8") as f:
        f.write(r1 + "\t" + r2 + "\n")
    i+=1


# In[2]:


print(len(final))


# In[3]:


len(final)


# In[3]:


print(c)


# In[ ]:




