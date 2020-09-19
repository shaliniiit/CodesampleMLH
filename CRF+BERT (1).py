import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import csv
LEN = 50
df1 = pd.read_csv(r"CorrectedChineseTrainsetFinal.txt",quoting=csv.QUOTE_NONE, encoding="UTF-8", sep = "\t", error_bad_lines=False).fillna(method="ffill")
dat1 = df1.values.tolist()
df2 = pd.read_csv(r"CorrectedFrenchTrainsetFinal.txt",quoting=csv.QUOTE_NONE, encoding="UTF-8", sep = "\t", error_bad_lines=False).fillna(method="ffill")
dat2 = df2.values.tolist()
df3 = pd.read_csv(r"CorrectedGermanTrainsetFinal.txt",quoting=csv.QUOTE_NONE, encoding="UTF-8", sep = "\t", error_bad_lines=False).fillna(method="ffill")
dat3 = df3.values.tolist()
df4 = pd.read_csv(r"QuerySet_Tagged_TrainData.tsv",quoting=csv.QUOTE_NONE, encoding="UTF-8", sep = "\t", error_bad_lines=False).fillna(method="ffill")
dat4 = df4.values.tolist()
dat = dat1 + dat2 + dat3 + dat4
l = len(dat)
print("Length of dataset : " + str(l))
i = 0
data = [[] for i in range(l)]
while i<l:
  data[i] =  [x for x in dat[i] if str(x) != 'nan']
  i+=1
print("Length of dataset after removing nan : " + str(len(data)))
l= len(data)
count = 0
sentences = [[] for i in range(l)]
labels = [[] for i in range(l)]
i=0
while i<len(data):
  if len(data[i])<2:
    i+=1
    continue
  tmp1 = []
  tmp1 = data[i][0].split(" ")
  tl1  = len(tmp1)
  tmp2 = []
  tmp2 = data[i][1].split(" ")
  tl2  = len(tmp2)
  if tl1 != tl2 :
    i+=1
    continue
  if tl1>LEN:
    i+=1
    continue
  sentences[count] = tmp1
  labels[count] = tmp2
  count+=1
  i+=1
print("Length of sentences : " + str(len(sentences)))
l=len(sentences)
tag_values = []
i=0
while i<l:
  l1 = len(labels[i])
  j = 0
  while j<l1:
    if labels[i][j] not in tag_values:
      tag_values.append(labels[i][j])
    j+=1
  i+=1
n_tags = len(tag_values)
tag2idx = {t: i for i, t in enumerate(tag_values)}
idx2tag = {i: t for i, t in enumerate(tag_values)}
!pip install transformers==2.6.0
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
torch.__version__
MAX_LEN = LEN
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
#print(n_gpu)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels
sentencesf = list(filter(None, sentences))
labelsf = list(filter(None, labels))
#print(len(sentencesf))
i=0
while i<len(sentencesf):
      #print(i)
      sentencesf[i] = [ k for k in sentencesf[i] if k != ""]
      labelsf[i] = [ k for k in labelsf[i] if k != ""]
      i+=1
tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentencesf, labelsf)
]
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
print("length of input_ids : " + str(len(input_ids)))
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
import transformers
from transformers import BertForTokenClassification, AdamW
transformers.__version__
model = BertForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)
model.cuda()
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)
from transformers import get_linear_schedule_with_warmup
epochs = 2
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
!pip install seqeval
from seqeval.metrics import f1_score, accuracy_score
loss_values, validation_loss_values = [], []
for _ in trange(epochs, desc="Epoch"):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
 
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))
    loss_values.append(avg_train_loss)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)
    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


Pkl_Filename = "Pickle_CRF_Model.pkl"  
import pickle
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
saved_model = pickle.dumps(model)

Pkl_Filename = "Pickle_CRF_Model.pkl"  
import pickle
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
saved_model = pickle.dumps(model)

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
LEN =30
df2 = pd.read_csv("QuerySet_Tagged_TestData.tsv", encoding="UTF-8", sep = "\t").fillna(method="ffill")
dat2 = df2.values.tolist()
l = len(dat2)
print("length of testset : " + str(l))
i=0
wronglytokenisedquery = 0
wronglytokenisedtoken = 0
wronglysplitquery = 0
wronglysplittoken = 0
wronglytaggedquery = 0
wronglytaggedtoken = 0
rightlytaggedquery = 0
rightlytaggedtoken = 0
f1 = open(r"Queries with different number of actualtoken and predictedtoken after evaluating.txt", "a")
f2 = open(r"Queries with different number of actualtoken and actuallabel after splitting.txt", "a")
f3 = open(r"Error File CRF+BERT.txt", "a")
f4 = open(r"Proper File CRF+BERT.txt", "a")
f4.write("Query\tActual Tags\tPredicted Tags\n")
f3.write("Query\tActual Tags\tPredicted Tags\n")
res = ""
f1.write( "Query\tActual Tokens\tPredicted Tokens\tActual Labels\tPredicted Labels\n")
f2.write("Sentence\tActual Tokens by splitting\n")
#f.close()
c=len(dat2)
pred_tag = [[] for i in range(c)]
y_te_true_tag = [[] for i in range(c)]
while i<c:
  #print(i)
  res = ""
  test_sentence = dat2[i][0]
  #print(test_sentence)
  actualtokens=[]
  actualtokens=dat2[i][0].split(" ")
  actualtokens = [k for k in actualtokens if k!=""]
  actuallabels = []
  l2 = len(actualtokens)
  actuallabels = dat2[i][1].split(" ")
  actuallabels = [k for k in actuallabels if k!=""]
  if len(actualtokens)!=len(actuallabels):
    wronglysplitquery+=1
    wronglysplittoken+=l2
    f2 = open(r"Queries with different number of actualtoken and actuallabel after splitting.txt", "a")
    f2.write(test_sentence+"\t")
    r = ""
    k=0
    while k<len(actualtokens):
          r+=actualtokens[k]+"\t"
          k+=1
    f2.write(r)
    f2.write("\n")
    f2.close()
    i+=1
    continue
  tokenized_sentence = tokenizer.encode(test_sentence)
  input_ids = torch.tensor([tokenized_sentence]).cuda()
  with torch.no_grad():
    output = model(input_ids)
  label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
  tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, label_indices[0]):
      if token.startswith("##"):
          new_tokens[-1] = new_tokens[-1] + token[2:]
      else:
          new_labels.append(tag_values[label_idx])
          new_tokens.append(token)
  k=1
  predictedtokens=[]
  predictedlabels=[]
  while k<(len(new_tokens)-1):
    predictedtokens.append(new_tokens[k])#actual list of predicted tokens
    predictedlabels.append(new_labels[k])
    k+=1  
  if len(actualtokens)!=len(predictedtokens):
      wronglytokenisedquery+=1
      wronglytokenisedtoken+=l2
      res1 = ""
      res2 = ""
      k = 0
      while k<len(actualtokens):
        res1=res1+actualtokens[k]+"  "
        res2=res2+actuallabels[k]+"  "
        k+=1
      res3 = ""
      res4 = ""
      k = 0
      while k<len(predictedtokens):
        res3=res3+predictedtokens[k]+"  "
        res4=res4+predictedlabels[k]+"  "
        k+=1
      f1 = open(r"Queries with different number of actualtoken and predictedtoken after evaluating.txt", "a")
      f1.write(test_sentence+"\t"+res1+"\t"+res2+"\t"+res3+"\t"+res4+"\n")
      f1.close()
      #print(test_sentence+"\t"+res1+"\t"+res2+"\t"+res3+"\t"+res4+"\n")
      i+=1
      continue
  flag=0
  if len(actualtokens)==len(predictedtokens):
      pred_tag.append(predictedlabels)
      y_te_true_tag.append(actuallabels)
      k=0
      while k<len(actualtokens):
        if actuallabels[k]!=predictedlabels[k]:
          wronglytaggedquery+=1
          flag=1
          break
        k+=1
      if flag==1:
        while k<len(actualtokens):
          if actuallabels[k]!=predictedlabels[k]:
              wronglytaggedtoken+=1
          if actuallabels[k]==predictedlabels[k]:
              rightlytaggedtoken+=1
          k+=1
        re = test_sentence
        re1 = ""
        re2 = ""
        k=0
        while k<len(actualtokens):
              re1=re1+actuallabels[k]+"  "
              re2=re2+predictedlabels[k]+"  "
              k+=1
        f3 = open(r"Error File CRF+BERT.txt", "a")
        f3.write(re+"\t"+re1+"\t"+re2+"\n")
      if flag==0:
        rightlytaggedquery+=1
        rightlytaggedtoken+=l2
        re = test_sentence
        re1 = ""
        re2 = ""
        k=0
        while k<len(actualtokens):
              re1=re1+actuallabels[k]+"  "
              re2=re2+predictedlabels[k]+"  "
              k+=1
        f4 = open(r"Proper File CRF+BERT.txt", "a")
        f4.write(re+"\t"+re1+"\t"+re2+"\n")
  i+=1
f1.close()
f2.close()
f3.close()
f4.close()
#query basis
wrong  = wronglysplitquery + wronglytaggedquery + wronglytokenisedquery
right = rightlytaggedquery
i=0
actual = []
predicted = []
while i<right:
  actual.append(1)
  predicted.append(1)
  i+=1
i=0
while i<wrong:
  actual.append(1)
  predicted.append(0)
  i+=1
#query basis
from sklearn import metrics
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn import datasets, linear_model
#print("Query wise results")
print("Query Wise Accuracy : " +str (accuracy_score(actual,predicted)))
#print("Precision : " + str(precision_score(actual,predicted)))
#print("Recall score : " + str(recall_score(actual,predicted)))
#token basis
wrong  = wronglysplittoken + wronglytaggedtoken + wronglytokenisedtoken
right = rightlytaggedtoken
i=0
actual = []
predicted = []
while i<right:
  actual.append(1)
  predicted.append(1)
  i+=1
i=0
while i<wrong:
  actual.append(1)
  predicted.append(0)
  i+=1
#token basis
from sklearn import metrics
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
#print("Token wise results")
print("Token Wise Accuracy : " +str (accuracy_score(actual,predicted)))
#print("Precision : " + str(precision_score(actual,predicted)))
#print("Recall score : " + str(recall_score(actual,predicted)))
#actual only on tagging basis
wrong  = wronglytaggedtoken
right = rightlytaggedtoken
i=0
actual = []
predicted = []
while i<right:
  actual.append(1)
  predicted.append(1)
  i+=1
i=0
while i<wrong:
  actual.append(1)
  predicted.append(0)
  i+=1
#actual tagging basis
from sklearn import metrics
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
#from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
#print("Actual tagging wise results")
print("Actual Tagging Wise Accuracy : " +str (accuracy_score(actual,predicted)))
#print("Precision : " + str(precision_score(actual,predicted)))
#print("Recall score : " + str(recall_score(actual,predicted)))
#print("Wrongly split queries : " + str(wronglysplitquery))
#print("Wrongly tokenised queries : " + str(wronglytokenisedquery))
#print("Wrongly tagged queries : " + str(wronglytaggedquery))
#print("Rightly Tagged queries : " + str(rightlytaggedquery))
#print("Wrongly split tokens : " + str(wronglysplittoken))
#print("Wrongly tokenised tokens : " + str(wronglytokenisedtoken))
#print("Wrongly tagged tokens : " + str(wronglytaggedtoken))
#print("Rightly Tagged tokens : " + str(rightlytaggedtoken))
!pip install sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)
print("F1-score: {:.1%}".format(f1_score(y_te_true_tag,pred_tag)))
x = report.split()
precision = []
recall = []
f1score = []
support = []
i = 0
while i<len(x):
  if x[i]=="B-action":
    precision.append(x[i+1])
    recall.append(x[i+2])
    f1score.append(x[i+3])
    support.append(x[i+4])
  elif x[i]=="B-entity":
    precision.append(x[i+1])
    recall.append(x[i+2])
    f1score.append(x[i+3])
    support.append(x[i+4])
  elif x[i]=="I-action":
    precision.append(x[i+1])
    recall.append(x[i+2])
    f1score.append(x[i+3])
    support.append(x[i+4])
  elif x[i]=="I-entity":
    precision.append(x[i+1])
    recall.append(x[i+2])
    f1score.append(x[i+3])
    support.append(x[i+4])
  i+=1
i = 0
sum = 0
while i<len(support):
  sum+=int(support[i])
  i+=1
#print(sum)
y = int(0.2*sum)
#print(y)
p = 0
count = 0
i = 0
while i<len(precision):
  if int(support[i])>=y:
      p = p + float(precision[i])
      count+=1
  i+=1
print("Precision = " + str(p/count))
p = 0
count = 0
i = 0
while i<len(recall):
  if int(support[i])>=y:
      p = p + float(recall[i])
      count+=1
  i+=1
print("Recall = " + str(p/count))