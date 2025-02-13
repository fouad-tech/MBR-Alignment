import os
import argparse
import json
import math
import numpy as np
import pandas as pd
import evaluate
from utility_func import *
from tqdm import tqdm
import argparse


def compute_score(df, d_best, trg, compute_evaluate, src=None):
    d_hyp = df.iloc[d_best]["text"]
    d_score = compute_evaluate(d_hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default='cnndm')
    args = parser.parse_args()
    dataset = args.dataset
    sim = 'bertscore'
    eval_func = 'bertscore'
    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)
    count = 0
    df = pd.read_csv(dataset)
    df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
    trg = df.iloc[:]["trg"]
    selected = df.iloc[:]['chosen']
    rejected = df.iloc[:]['rejected']
    for i in tqdm(range(0,len(trg))):
        selected_score = compute_evaluate(selected[i], trg[i], None)
        rejected_score = compute_evaluate(rejected[i], trg[i], None)
        if rejected_score>selected_score:
            count+=1

    print(count/len(trg))

     

      