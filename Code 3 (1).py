import pandas as pd
import numpy as np
import re
import hanzidentifier
import io
import csv
from tqdm import tqdm, trange
df = pd.read_csv(r"RightDataset.txt", encoding="UTF-8", sep = "\t",quoting=csv.QUOTE_NONE, error_bad_lines=False).fillna(method="ffill")
dat = df.values.tolist()
#print(len(dat))


#Choose g% testing data
g = input("Enter percentage of training data : ") 
g = int(g)
g = g/100
x = int(g*len(dat))
print(x)
testset1 = []
import random
i = 0
while i<x:
      random_item_from_list = random.choice(dat)
      testset1.append(random_item_from_list)
      dat.remove(random_item_from_list)
      i+=1
print("Length of trainset : " + str(len(dat)))
print("Length of testset: " + str(len(testset1)))


i = 0
f2 = r"Testset0.txt"
import io
with io.open(f2, "a", encoding="utf-8") as f:
        f.write("Query" + "\t" + "English Translation" + "\t" + "Tags" + "\t" + "Mapping" + "\n")
while i<len(testset1):
    with io.open(f2, "a", encoding="utf-8") as f:
        f.write(str(testset1[i][0]) + "\t" + str(testset1[i][1]) + "\t" + str(testset1[i][2]) + "\t" + str(testset1[i][3]) + "\n")
    i+=1


i = 0
f2 = r"Trainset0.txt"
import io
with io.open(f2, "a", encoding="utf-8") as f:
        f.write("Query" + "\t" + "English Translation" + "\t" + "Tags" + "\t" + "Mapping" + "\n")
while i<len(dat):
    with io.open(f2, "a", encoding="utf-8") as f:
        f.write(str(dat[i][0]) + "\t" + str(dat[i][1]) + "\t" + str(dat[i][2]) + "\t" + str(dat[i][3]) + "\n")
    i+=1


