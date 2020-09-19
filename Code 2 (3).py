import pandas as pd
import numpy as np
import re
import hanzidentifier
import io
import csv
from tqdm import tqdm, trange
df = pd.read_csv(r"RightTrain.txt", encoding="UTF-8", sep = "\t",quoting=csv.QUOTE_NONE, error_bad_lines=False).fillna(method="ffill")
dat = df.values.tolist()
#print(len(dat))
eng = [[] for y in range(0)]
mape = [[] for y in range(0)]
query = [[] for y in range(0)]
tag = [[] for y in range(0)]
i = 0
c = 0
while i<len(dat):
    #print(i)
    dict = {}
    g = ""
    f = ""
    a = eval(dat[i][0])
    b = eval(dat[i][1])
    #print(a)
    #print(b)
    de = ""
    for k in b:    
        if b[k].lower().strip() not in a:
            continue
        if b[k].lower().strip() in a:
            dict[k] = a[b[k].lower().strip()]
            de = de + b[k] + " "
    #print(de)
    #print(dict)
    for k in dict : 
        f = f + k + " "
        g = g + dict[k] + " "
    query.append(f)
    tag.append(g) 
    eng.append(de)
    mape.append(dat[i][2])
    i+=1




f2 = r"RightDataset.txt"
import io
with io.open(f2, "a", encoding="utf-8") as f:
        f.write("Query" + "\t" + "English Translation" + "\t" + "Tags" + "\t" + "Mapping" + "\n")
i = 0
import io
i = 0
m = 0
n = 0
while i<len(query):
    if query[i]=="" or eng[i]=="" or tag[i]=="" or mape[i]=="":
        i+=1
        m+=1
        continue
    if query[i]==" " or eng[i]==" " or tag[i]==" " or mape[i]==" ":
        i+=1
        n+=1
        continue
    with io.open(f2, "a", encoding="utf-8") as f:
        f.write(str(query[i]) + "\t" + str(eng[i]) + "\t" + str(tag[i]) + "\t" + str(mape[i]) + "\n")
    i+=1


