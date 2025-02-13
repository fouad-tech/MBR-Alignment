import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utility_func import *
from datasets import Dataset
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    load_dataset,
    load_matrix,
    load_samples_from_file,
    result_dir,
    matrix_dir,
    prompt_dir
)  # , approx_dir, diverse_dir
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix, compute_mbr


def compute_score(df, d_best, trg, compute_evaluate, src=None):
    d_hyp = df.iloc[d_best]["text"]
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score

def computeBert(trg,samples,compute_evaluate):
    scores = []
    for i in samples:
        scores.append(compute_evaluate(i,trg,None))
    scores = np.array(scores)
    return np.argsort(scores),np.argmax(scores),np.argmin(scores)

def compute_mbr(
    hyp=None,
    compute_similatiy=None,
    matrix=None,
    weights=None,
    src=None,
    incremental=False,
):
    assert (compute_similatiy is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similatiy, [src] * len(hyp))

    if weights is not None:
        mbr_scores = matrix @ np.transpose(weights)
    else:
        mbr_scores = np.sum(matrix, axis=1)

    if incremental:
        best_hyp = -1
        best_score = -np.inf
        bests = []
        for i in range(mbr_scores.shape[0]):
            if mbr_scores[i] > best_score:
                best_hyp = i
                best_score = mbr_scores[i]
            assert best_hyp >= 0
            bests.append(best_hyp)
        return bests  # List of hypothesis indices.
    else:
        hyp_list = np.argsort(mbr_scores)
        best_hyp = np.argmax(mbr_scores)
        worst_hyp = np.argmin(mbr_scores)

        assert len(hyp_list) >= 0
        return hyp_list,best_hyp,worst_hyp,mbr_scores

if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default='cnndm')
    parser.add_argument('--model_ref',default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--prompt', default='promptCNN.txt')
    parser.add_argument('--LowerRange',default=0,type = int)
    parser.add_argument('--UpperRange',default=1000,type=int)
    parser.add_argument('--eps',default=0.02,type=float)
    parser.add_argument('--topk',default=0, type = float)
    parser.add_argument('--topp',default=1.0,type = float)
    
    args = parser.parse_args()

    dataset = args.dataset
    prompt_path=args.prompt
    UpperRange = args.UpperRange
    LowerRange = args.LowerRange
    model_ref= args.model_ref

    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()
    print(prompt)


    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = 'bertscore'
    eval_func = 'bertscore'


    n_samples = 32



    # Algorithm config
    
    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)
    compute_evaluate_meteor, evaluator1 = load_evaluate('meteor', sim, similarity)
    compute_evaluate_rouge, evaluator2 = load_evaluate('rouge', sim, similarity)
    compute_evaluate_rouge1, evaluator3 = load_evaluate('rouge1', sim, similarity)

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    # client = boto3.client("s3")

    model_n = os.path.basename(model_ref)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)
    sample_dir = os.path.join('./samples',dataset,model_n)

    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp)

    assert len(filtered_files) > 0
    meteor = []
    bertscore =[]
    rouge1 = []
    rouge = []
    rows = []
    correlations = []
    print(len(filtered_files))
    for fileId in tqdm(range(LowerRange,UpperRange)):
        filename = filtered_files[fileId]
        sample_id = int(filename.split("_")[0])
        assert "{:04}".format(sample_id) in filename

        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]

        df = pd.read_csv(os.path.join(sample_dir, filename))

        assert len(df) >= n_samples
        df = df[:n_samples]

        df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        hyp = df.iloc[:]["text"]
        
        matrix = load_matrix(
                os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples
            )
        hyp_list,best_hyp,worst_hyp,mbr_scores = compute_mbr(matrix=matrix)
        #hyp_list = hyp_list[::-1]
        r_list,bert_max, _ = computeBert(trg,hyp,compute_evaluate)


        sorted_indices_method1 = r_list
        sorted_indices_method2 = hyp_list
        ranks_method1 = np.empty_like(sorted_indices_method1)
        ranks_method2 = np.empty_like(sorted_indices_method2)
        ranks_method1[sorted_indices_method1] = np.arange(len(r_list))
        ranks_method2[sorted_indices_method2] = np.arange(len(mbr_scores))
        correlation, _ = spearmanr(ranks_method1, ranks_method2)
        correlations.append(correlation)

        if dataset == 'squad_v2':
            d_hyp = df.iloc[bert_max]["text"]
            meteor.append(compute_evaluate_meteor(d_hyp,trg,None))
            bertscore.append(compute_evaluate(d_hyp,trg,None))
        elif dataset == 'cnndm':
            d_hyp = df.iloc[bert_max]["text"]
            meteor.append(compute_evaluate_meteor(d_hyp,trg,None))
            bertscore.append(compute_evaluate(d_hyp,trg,None))
            rouge.append(compute_evaluate_rouge(d_hyp,trg,None))
        else:
            d_hyp = df.iloc[bert_max]["text"]
            meteor.append(compute_evaluate_meteor(d_hyp,trg,None))
            bertscore.append(compute_evaluate(d_hyp,trg,None))
            rouge.append(compute_evaluate_rouge(d_hyp,trg,None))
            rouge1.append(compute_evaluate_rouge1(d_hyp,trg,None))




    print(f"Spearman's rank correlation coefficient for {dataset}: {sum(correlations)/len(correlations)}") 

    if dataset == 'squad_v2':
            
            print('meteor_score: ',sum(meteor)/len(meteor))
            print('bert_score: ', sum(bertscore)/len(bertscore))
    elif dataset == 'cnndm':
            print('meteor_score: ',sum(meteor)/len(meteor))
            print('bert_score: ', sum(bertscore)/len(bertscore))
            print('rouge_score: ',sum(rouge)/len(rouge))
    else:
            print('meteor_score: ',sum(meteor)/len(meteor))
            print('bert_score: ', sum(bertscore)/len(bertscore))
            print('rouge_score: ',sum(rouge)/len(rouge))
            print('rouge1_score: ',sum(rouge1)/len(rouge1))

        





   
