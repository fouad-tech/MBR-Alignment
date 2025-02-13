import os
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from utility_func import *
from utils import load_dataset

def compute_evaluate(hyp, ref):
            evaluator = load('rouge')
            rougeL = evaluator.compute(predictions=[hyp], references=[[ref]])["rougeL"]
            rouge1 = evaluator.compute(predictions=[hyp], references=[[ref]])["rouge1"]
            rouge2 = evaluator.compute(predictions=[hyp], references=[[ref]])["rouge2"]
            evaluator = load('sacrebleu')
            bleu =evaluator.compute(predictions=[hyp], references=[ref])["score"]
            evaluator = load('bleurt', checkpoint="BLEURT-20")
            bleurt =evaluator.compute(predictions=[hyp], references=[ref])["scores"][0]

            return rougeL, rouge1, rouge2, bleu,bleurt

if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ResultsFile', help="ResultsFile path", default='../model-based-mbr/results/cnndm_Mistral-7B-Instruct-v0.1_032_0.02_00_1.00_bertscore_rouge_0-1000.csv')
    parser.add_argument('--hypothesis', help="is it mbr or model based mbr", default='ed_best')
    parser.add_argument('--dataset', help='the dataset name', default='cnndm')
    parser.add_argument('--outfile', help='the output file for the metrics', default='../model-based-mbr/resultsMetrics/MBR_Mistral_cnndm')
    parser.add_argument('--sampleDir', help='the sampling directory', default='../model-based-mbr/samples/cnndm/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--LowerRange', type=int, default=0)
    parser.add_argument('--UpperRange', type = int,default=1000)
    
    
    
    args = parser.parse_args()

    resultsFile = args.ResultsFile
    hypothesis = args.hypothesis
    dataset = args.dataset
    LowerRange = args.LowerRange
    UpperRange = args.UpperRange
    outfile = args.outfile
    sampleDir = args.sampleDir
    #####
    
    
    samplesArr = os.listdir(sampleDir)
    samplesArr = sorted(samplesArr)
    trg_lines = load_dataset(dataset, ref=True)
    df = pd.read_csv(resultsFile)
    df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
    hyp = df.iloc[:][hypothesis]
    
    rougeLScore = []
    rouge2Score = []
    bleuScore = []
    rouge1Score = []
    bleurtScore = []
    results = []
    for id in tqdm(range(LowerRange,UpperRange)):
       
       
        df = pd.read_csv(os.path.join(sampleDir,samplesArr[id]))
        df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        h = df.iloc[hyp[id]]['text']
        trg = trg_lines[id]

        rougeL, rouge1, rouge2, bleu,bleurt = compute_evaluate(h,trg)
        rougeLScore.append(rougeL)
        rouge1Score.append(rouge1)
        rouge2Score.append(rouge2)
        bleuScore.append(bleu)
        bleurtScore.append(bleurt)
        results.append((h,rougeL,rouge1,rouge2,bleu,bleurt))
            
    meanRougeL = sum(rougeLScore) / len(rougeLScore)
    meanRouge1 = sum(rouge1Score) / len(rouge1Score)
    meanBlue = sum(bleuScore)/len(bleuScore)
    meanRouge2 = sum(rouge2Score)/len(rouge2Score)
    meanBleurt = sum(bleurtScore)/len(bleurtScore)
       
    
    
    samplesFile = os.path.join(outfile,'Samples.csv')
    df = pd.DataFrame(results, columns=['text','rougeL','rouge1','rouge2','blue','bleurt'])
    df.to_csv(samplesFile, index=False)
    metricsFile= os.path.join(outfile,'Metrics.txt')
    with open(metricsFile, 'w') as f:
        f.write('meanRougeL: {}'.format(meanRougeL))
        f.write(' meanRouge1: {}'.format(meanRouge1))
        f.write(' meanRouge2: {}'.format(meanRouge2))
        f.write(' meanBlue: {}'.format(meanBlue))
        f.write(' meanBluert: {}'.format(meanBleurt))
