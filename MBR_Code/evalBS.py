import os 
import torch
import pandas as pd
import numpy as np
from evaluate import load



import os
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from utility_func import *
from utils import (
    load_dataset,
    load_matrix,
    load_kwargs,
    load_samples_from_file,
    result_dir,
    matrix_dir,
    prompt_dir,
)  # , approx_dir, diverse_dir
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix, compute_mbr


def compute_score(hyp, trg, compute_evaluate, src=None):
    d_score = compute_evaluate(hyp, trg, src)
    return d_score


if __name__ == "__main__":
    """
    This script is the "main function" of the experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help="dataset name", default='../model-based-mbr/resultsMetrics/BeamMistralStrategyqaV/Samples.csv')
    parser.add_argument('--results_path',help='where to store the results', default='../model-based-mbr/resultsMetrics/BeamMistralStrategyqaV')
    parser.add_argument('--sim', default='bertscore', help='simalrity metrics')
    parser.add_argument('--eval', nargs='+', help='evaluation metrics')
    


    args = parser.parse_args()

    sim = args.sim
    eval_func = args.eval
    dataset_path = args.dataset_path
    results_path = args.results_path

   
    # Algorithm config
    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate = []
    for e in eval_func:
          c,_ = load_evaluate(e, sim, similarity)
          compute_evaluate.append(c)
    

    metricsPath = os.path.join(results_path,'{}.csv'.format(eval_func))
    resultsPath = os.path.join(results_path,'means.csv')

    texts=[]
    rows = []
   
    print('not recompute')
    df = pd.read_csv(dataset_path)
    df.fillna(
                "", inplace=True
            )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
    print(df.columns)
    for row in df.itertuples(index=False, name=None):
            texts.append([row[0],row[1]])
                   
    for item in tqdm(texts):

            hyp = item[1]
            target = item[0]
            row = []
            for e in compute_evaluate:
                score = compute_score(hyp, target, e, src=None)
                row.append(score)
       
            rows.append(row)

    
    columns = [
                '{}_score'.format(e) for e in eval_func
            ]
    df = pd.DataFrame(rows, columns=columns)
    print(rows)
    df.to_csv(metricsPath, index=False)
    means = df.mean()
    with open(resultsPath, 'w') as file:
        for column, mean in means.items():
            file.write(f'{column}: {mean}\n')