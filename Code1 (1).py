import pandas as pd
import numpy as np
import re
import csv
import io
from tqdm import tqdm, trange
df = pd.read_csv(r"DataTrain1.txt",quoting=csv.QUOTE_NONE, encoding="UTF-8", sep = "\t", error_bad_lines=False).fillna(method="ffill")
dat = df.values.tolist()
#print(len(dat))
data = []
data = dat + []
i=0
while i<len(data):
  if len(data[i])<3:
    del data[i]
  i+=1
data =  list(filter(None, data))
#print(len(data))
i=0
import re
import io
dictquery = {}
dictmap = {}
dictact = {}
re1 = []
fname = "Translate.txt"
l = len(data)
repetition = 0
while i<len(data):
      #print(i)
      if data[i][1] in dictquery:
          re1.append(i)
          repetition+=1
      g = ""
      g = str(data[i][1])
      g = g.strip()
      g = g.lstrip()
      g = g.lower()
      j = 0
      g = re.sub(r'[^\w\s]','',g)
      #print(g)
      with io.open(fname, "a", encoding="utf-8") as f:
          f.write(g+"\n")
      dictact[g] = data[i][1]
      dictquery[g] = data[i][0]
      dictmap[g] = data[i][2]
      i+=1
print("length of data : " + str(len(data)))
print("length of dictionary : " + str(len(dictquery)))
print("No of repetitions : " + str(repetition))


# In[2]:


import pandas as pd
df1 = pd.read_csv("TaggedQueries.txt", encoding="UTF-8", sep = "\n").fillna(method="ffill")
dat1a = df1.values.tolist()
l = len(dat1a)
i = 0
dat1 = [[] for i in range(0)]
while i<len(dat1a):
    p  = str(dat1a[i][0])
    q = list(p.split("\t"))
    q = [k for k in q if k!=""]
    q = [k for k in q if k!="nan"]
    q = [k for k in q if k!="\t"]
    #print(q)
    dat1.append(q)
    i+=1
i = 0
while i<l:
  dat1[i] =  [x for x in dat1[i] if str(x) != 'nan']
  dat1[i] =  [x.strip() for x in dat1[i]]
  dat1[i] =  [x.lstrip() for x in dat1[i] if str(x) != 'nan']
  i+=1
i = 0
while i<l:
  dat1[i] =  [x for x in dat1[i] if str(x) != 'nan']
  i+=1
i = 0
while i<len(dat1) :
      dat1[i][0] = dat1[i][0].lower()
      i+=1
i = 0
import re
while i<len(dat1):
        dat1[i][0]=re.sub(r'[^\w\s]','',dat1[i][0])
        i+=1
i = 0
while i<len(dat1):
        dat1[i][0]=dat1[i][0].strip()
        dat1[i][0]=dat1[i][0].lstrip()
        dat1[i][0]=dat1[i][0].lower()
        i+=1
i=0
wrong = 0
right = 0
c=0
count1=0
h = []
proper = 0
notagging = 0
noenglish = 0
nofrenchquery = 0
nomapping = 0
f1 = "WrongTrain.txt"
f2 = "RightTrain.txt"
while i<len(dat1):
  tmp1 = str(dat1[i][0]).strip()#english tokens
  tmp1 = tmp1.lower()
  tmp1 = tmp1.lstrip()
  p1=[]
  p1=tmp1.split(" ")
  tmp1 = re.sub(r'[^\w\s]','',tmp1)
  p1 = [k for k in p1 if k!=""]
  p1 = [k for k in p1 if k!="nan"]
  tmp2 = str(dat1[i][1]).strip()#english labels
  p2=[]
  p2=tmp2.split(" ")
  p2 = [k for k in p2 if k!=""]
  p2 = [k for k in p2 if k!="nan"]
  if len(p1)!=len(p2):
    i+=1
    c+=1
    continue
  if len(p1)==len(p2):
          if tmp1 not in dictquery:
            h.append(tmp1)
            #print("Bye")
            nofrenchquery+=1
            i+=1
            continue
          if tmp1 not in dictmap:
            nomapping+=1
            i+=1
            continue
          dictentofr = {}
          dictfrtoen = {}
          mapi =str(dictmap[tmp1]).strip()#mapping
          t = str(dictquery[tmp1]).strip()#chinese query
          tmp = []
          tmp=mapi.split(" ")
          j=0
          final=[]
          while j<len(tmp):
              t1=[]
              t11=[]
              t12=[]
              t1 = tmp[j].split("-")
              t1 = [k for k in t1 if k!=""]
              t11 = t1[0].split(":")
              tl1 = [k for k in t1 if k!=""]
              t12 = t1[1].split(":")
              tl2 = [k for k in t1 if k!=""]
              tf = t11+t12
              final.append(tf)
              j+=1
          t = str(dictquery[tmp1]).strip()#chinese query
          j=0
          tmpf = dictact[tmp1]
          while j<len(final):
                      a = int(final[j][0])
                      b = int(final[j][1])
                      c = int(final[j][2])
                      d = int(final[j][3])
                      fr = t[a:(b+1)]
                      en = tmpf[c:(d+1)]
                      if fr in dictfrtoen:
                          j+=1
                          continue
                      dictentofr[en] = fr
                      dictfrtoen[fr] = en
                     
                      j+=1
          dictwordtoentity={}
          j=0
          while j<len(p1):
              a = p1[j]
              dictwordtoentity[a]=p2[j]
              
              j+=1
          mapped = 0
          for k in dictfrtoen:
                mapped = mapped + len(k)
          q = []
          q = t.split()
          q = [k for k in q if k!=""]
          q = [k for k in q if k!=" "]
          #print(q)
          actual = 0
          for k in q:
            actual = actual + len(k)
          if mapped!=actual:
                 wrong+=1
                 with io.open(f1, "a", encoding="utf-8") as f:
                      f.write(str(dictwordtoentity)+"\t") 
                      f.write(str(dictfrtoen)+"\t")
                      f.write(mapi + "\t")
                      f.write(t + "\t")
                      f.write("\n")
          else:
                right+=1
                with io.open(f2, "a", encoding="utf-8") as f:
                      f.write(str(dictwordtoentity)+"\t") 
                      f.write(str(dictfrtoen)+"\t")
                      f.write(mapi + "\t")
                      f.write("\n")
  i+=1
print("Number of wrong data : " + str(wrong))
print("Number of right data : " + str(right))
print("Length of grammatical errors : " + str(len(h))




