import os
import pandas as pd
from evaluate import load
import datasets

path = '../samples/cnndm/Mistral-7B-Instruct-v0.1'
list = os.listdir(path)
list = sorted(list)
print(list[0:10])
count_zeroes = 0
f = 0
for i in range(0,1000):
    df = pd.read_csv(os.path.join(path, list[i]))
    c = (df['probability'] == 0.0).sum()
    count_zeroes+=c
    if c == 32:
        f+=1
print("Number of entries that are 0.0:", count_zeroes/32000)
print("Number of files that are 0.0:", f/1000)
#Number of entries that are 0.0: 0.8964375

def compute_evaluate(hyp, ref):
            evaluator = load('rouge')
            return evaluator.compute(predictions=[hyp], references=[[ref]])["rougeL"]

x = []
dataset = datasets.load_dataset("cnn_dailymail", "3.0.0",split= 'test')
src_lines = dataset['article']
target_lines = dataset["highlights"]
f = open('../model-based-mbr/mbr/eval.txt').readlines()
print(len(f))
for idx,line in enumerate(f):
       f[idx]= line.split('|||||')[0]

for i in range(0,1000):
      x.append(compute_evaluate(f[i],target_lines[i]))

mean_value = sum(x) / len(x)
with open('../model-based-mbr/data/DPO01BW.txt', 'w') as f:
        f.write(mean_value)
print(mean_value)

