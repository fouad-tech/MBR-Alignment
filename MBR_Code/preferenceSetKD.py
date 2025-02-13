import os
import argparse
import json
import math
import numpy as np
import pandas as pd
import evaluate
from utility_func import *
from utils import (
    load_dataset,
    load_matrix,
    load_samples_from_file,
    result_dir,
    matrix_dir,
    prompt_dir,
)  # , approx_dir, diverse_dir
from parser import get_mbr_parser
from tqdm import tqdm


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
        return hyp_list,best_hyp,worst_hyp

def compute_evaluateN(hyp, ref):

      evaluator = evaluate.load('meteor')
      scores = [
                (evaluator.compute(predictions=[hyp], references=[r])["meteor"],r)
                for r in ref
            ]
      return max(scores)


def BW(hypsRanked,trg,src):
  top = hypsRanked[0]
  bottom = hypsRanked[-1]
  topPair = (src,trg,top,bottom)
  return topPair

def BMW(hypsRanked,trg,src):
  top = hypsRanked[0]
  middle = hypsRanked[math.ceil(len(hypsRanked)/2)]
  bottom = hypsRanked[-1]
  topPair = (src,trg,top,middle)
  bottomPair = (src,trg,middle,bottom)
  return topPair,bottomPair

def CPSSquad(hypsRanked,trg,src):
  samples = []
  m = compute_evaluateN(hypsRanked[0],trg)[1]
  for i in range(0,len(hypsRanked),2):
    if i+2 >= len(hypsRanked):
      break
    pair = ((src,m,hypsRanked[i],hypsRanked[i+2]))
    samples.append(pair)

  return samples

def CPS(hypsRanked,trg,src):
  samples = []
  for i in range(0,len(hypsRanked),2):
    if i+2 >= len(hypsRanked):
      break
    pair = ((src,trg,hypsRanked[i],hypsRanked[i+2]))
    samples.append(pair)
  
  return samples

def shuffle(df):
    shuffled_df = df.iloc[np.random.permutation(len(df))]

    split_index = int(len(shuffled_df) * 0.9)
    df_first_90 = shuffled_df.iloc[:split_index]
    df_last_10 = shuffled_df.iloc[split_index:]

    return df_first_90,df_last_10


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
    parser.add_argument('--model_name', help="model name", default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--kd_model_name', help="model name", default='meta-llama/Meta-Llama-3-70B-Instruct')
    parser.add_argument('--prompt', default='promptCNN.txt')
    parser.add_argument('--n_lines', default=1000, type = int)
    parser.add_argument('--n_samples', default=32, type = int)
    parser.add_argument('--kd_n_samples',default=16, type = int)
    parser.add_argument('--eps', default=0.02, type = float)
    parser.add_argument('--topk', default=0, type = int)
    parser.add_argument('--topp',default=1.0, type = float)
    parser.add_argument('--sim',default='bertscore')
    parser.add_argument('--eval',default='rouge')
    parser.add_argument('--UpperRange', type = int, default = 1000)
    parser.add_argument('--LowerRange',type = int, default = 0)

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name
    kd_model_name = args.kd_model_name

    sample_dir = os.path.join('./samples',dataset,os.path.basename(model_name))
    kd_sample_dir = os.path.join('./samples',dataset,os.path.basename(kd_model_name))

    print('sample_dir',sample_dir)
    print('####################')
    print('kd_sample_dir',kd_sample_dir)
    print('####################')
    prompt_path = args.prompt
    print(prompt_path)
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()

    n_lines = args.n_lines
    n_samples = args.n_samples
    kd_samples = args.kd_n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval
    UpperRange = args.UpperRange
    LowerRange = args.LowerRange

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    # client = boto3.client("s3")

    model_n = os.path.basename(model_name)
    model_kd = os.path.basename(kd_model_name)


    files = sorted(os.listdir(sample_dir))
    files_kd = sorted(os.listdir(kd_sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp)
    filtered_files_kd = load_samples_from_file(files_kd, epsilon, topk, topp)
    filtered_files_kd_dict = {}
    for f in filtered_files_kd:
        sample_id = int(f.split("_")[0])
        filtered_files_kd_dict[sample_id] = f

    assert len(filtered_files) > 0

    print("first 10 files=", filtered_files[:10])

    rowsCPS = []
    rowsBW = []
    rowsBMW = []

    for id in tqdm(range(LowerRange,UpperRange)):

        filename = filtered_files[id]
        sample_id = int(filename.split("_")[0])
        if sample_id not in filtered_files_kd_dict.keys():
            print('not in :' ,sample_id)
            continue
        else:
            filename_kd = filtered_files_kd_dict[sample_id]
        assert "{:04}".format(sample_id) in filename

        if sample_id >= n_lines:
            break

        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]

        df = pd.read_csv(os.path.join(sample_dir, filename))
        df_kd = pd.read_csv(os.path.join(kd_sample_dir, filename_kd))

        assert len(df) >= n_samples
        df = df[:n_samples]
        df_kd = df_kd[:kd_samples]

        df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        hyp = df.iloc[:]["text"]
        
        df_kd.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        hyp_kd = df_kd.iloc[:]["text"]

        
        matrix = load_matrix(
                os.path.join(matrix_dir, dataset, model_n), filename, sim, n_samples
            )
        matrix_kd = load_matrix(
                os.path.join(matrix_dir, dataset, model_kd), filename_kd, sim, kd_samples
            )
            
        ranking,best_hyp,worst_hyp = compute_mbr(matrix=matrix)
        ranking_kd,best_hyp_kd,worst_hyp_kd = compute_mbr(matrix=matrix_kd)
        hypsRanked = [df.iloc[i]["text"] for i in ranking]
        hypsRanked = hypsRanked[::-1]
        #middleText = hypsRanked[math.ceil(len(hypsRanked)/2)]
        
        
        if dataset == 'squad_v2T':
            m = compute_evaluateN(df_kd.iloc[best_hyp_kd]["text"],trg)
            m2 = compute_evaluateN(df.iloc[best_hyp]["text"],trg)
            trg = m[1]
            trg2 = m2[1]
            src_input = prompt +" " +src_input
            topmiddle = (src_input,trg,df_kd.iloc[best_hyp_kd]["text"],df.iloc[best_hyp]["text"])
            middleBottom = (src_input,trg2,df.iloc[best_hyp]["text"],df.iloc[worst_hyp]["text"])
            rowsBMW.append(topmiddle)
            rowsBMW.append(middleBottom)
            rowsBW.append(topmiddle)

      
        
        else:
            src_input = prompt.replace("[[QUESTION]]", src_input)
            topmiddle = (src_input,trg,df_kd.iloc[best_hyp_kd]["text"],df.iloc[best_hyp]["text"])
            middleBottom = (src_input,trg,df.iloc[best_hyp]["text"],df.iloc[worst_hyp]["text"])
            rowsBMW.append(topmiddle)
            rowsBMW.append(middleBottom)
            rowsBW.append(topmiddle)

            
            
            
        
        

    FilePath = '../model-based-mbr/PreferenceSetsSplits'
    columns = [
            "src",
            "trg",
            "chosen",
            "rejected",
        ]
    df = pd.DataFrame(rowsBW, columns=columns)
    df_first_90,df_last_10 = shuffle(df)
    bwFilePathTrain = os.path.join(FilePath,'bw_{}_{}_kd_train.csv'.format(dataset,n_samples))
    bwFilePathTest = os.path.join(FilePath,'bw_{}_{}_kd_test.csv'.format(dataset,n_samples))
    df_last_10.to_csv(bwFilePathTest, index=False)
    df_first_90.to_csv(bwFilePathTrain, index=False)
    print(bwFilePathTest)
        
    df = pd.DataFrame(rowsBMW, columns=columns)
    df_first_90,df_last_10 = shuffle(df)
    bwFilePathTrain = os.path.join(FilePath,'bmw_{}_{}_kd_train.csv'.format(dataset,n_samples))
    bwFilePathTest = os.path.join(FilePath,'bmw_{}_{}_kd_test.csv'.format(dataset,n_samples))
    df_last_10.to_csv(bwFilePathTest, index=False)
    df_first_90.to_csv(bwFilePathTrain, index=False)
    print(bwFilePathTest)
        

        

    

     
    