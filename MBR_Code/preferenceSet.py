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
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model

    sample_dir = args.sample_dir
    prompt_path = args.prompt
    print(prompt_path)
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()

    n_lines = args.n_lines
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval
    UpperRange = args.UpperRange
    LowerRange = args.LowerRange
    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix

    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    # client = boto3.client("s3")

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(matrix_dir, dataset, model_n), exist_ok=True)

    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp)

    assert len(filtered_files) > 0

    print("first 10 files=", filtered_files[:10])

    rowsCPS = []
    rowsBW = []
    rowsBMW = []

    for id in tqdm(range(LowerRange,UpperRange)):

        filename = filtered_files[id]
        sample_id = int(filename.split("_")[0])
        assert "{:04}".format(sample_id) in filename

        if sample_id >= n_lines:
            break

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
            
        ranking,best_hyp,worst_hyp = compute_mbr(matrix=matrix)
        hypsRanked = [df.iloc[i]["text"] for i in ranking]
        hypsRanked = hypsRanked[::-1]
        #middleText = hypsRanked[math.ceil(len(hypsRanked)/2)]
        
        
        if dataset == 'squad_v2T':
            '''m = compute_evaluateN(df.iloc[best_hyp]["text"],trg)
            m2 = compute_evaluateN(middleText,trg)
            trg = m[1]
            trg2 = m2[1]
            src_input = prompt +" " +src_input
            topmiddle = (src_input,trg,df.iloc[best_hyp]["text"],middleText)
            middleBottom = (src_input,trg2,middleText,df.iloc[worst_hyp]["text"])
            rowsBW.append(topmiddle)
            rowsBW.append(middleBottom)'''

            src_input = prompt +" " +src_input
            pairs = CPSSquad(hypsRanked,trg,src_input)
            for pair in pairs:
                rowsBW.append(pair)
        
        else:
            '''src_input = prompt.replace("[[QUESTION]]", src_input)
            topmiddle = (src_input,trg,df.iloc[best_hyp]["text"],middleText)
            middleBottom = (src_input,trg,middleText,df.iloc[worst_hyp]["text"])
            rowsBW.append(topmiddle)
            rowsBW.append(middleBottom)'''

            src_input = prompt.replace("[[QUESTION]]", src_input)
            pairs = CPS(hypsRanked,trg,src_input)
            for pair in pairs:
                rowsBW.append(pair)
            
            
        
        

    bwFilePath = '/rds/user/fk379/hpc-work/THESIS/model-based-mbr/PreferenceSetsSplits'
    columns = [
            "src",
            "trg",
            "chosen",
            "rejected",
        ]
    df = pd.DataFrame(rowsBW, columns=columns)
    df_first_90,df_last_10 = shuffle(df)
    bwFilePathTrain = os.path.join(bwFilePath,'cps_{}_{}_train.csv'.format(dataset,n_samples))
    bwFilePathTest = os.path.join(bwFilePath,'cps_{}_{}_test.csv'.format(dataset,n_samples))
    df_last_10.to_csv(bwFilePathTest, index=False)
    df_first_90.to_csv(bwFilePathTrain, index=False)
    print(bwFilePathTest)
        

        

    

     
    '''if dataset == "squad_v2T":
        
           src_input = prompt + " " + src_input
           s = compute_evaluateN(hypsRanked[0], trg)
           trg = s[1]
           topMiddle,middleBottom = BMW(hypsRanked,trg,src_input)
           topBottom = BW(hypsRanked,trg,src_input)
           cpsList = CPS(hypsRanked,trg,src_input)

           for elem in cpsList:
              rowsCPS.append(elem)
           
           rowsBW.append(topBottom)
           rowsBMW.append(topMiddle)
           rowsBMW.append(middleBottom)

        elif dataset == 'cnndm':

            src_input = prompt.replace("[[QUESTION]]", src_input)
            topMiddle,middleBottom = BMW(hypsRanked,trg,src_input)
            topBottom = BW(hypsRanked,trg,src_input)
            cpsList = CPS(hypsRanked,trg,src_input)
            for elem in cpsList:
              rowsCPS.append(elem)
           
            rowsBW.append(topBottom)
            rowsBMW.append(topMiddle)
            rowsBMW.append(middleBottom)


        else:

          topMiddle,middleBottom = BMW(hypsRanked,trg,src_input)
          topBottom = BW(hypsRanked,trg,src_input)
          cpsList = CPS(hypsRanked,trg,src_input)
          for elem in cpsList:
              rowsCPS.append(elem)
           
          rowsBW.append(topBottom)
          rowsBMW.append(topMiddle)
          rowsBMW.append(middleBottom)'''
        
  
    
    '''cpsFilePath = '/rds/user/fk379/hpc-work/THESIS/model-based-mbr/PreferenceSetsSplits'
    bwFilePath = '/rds/user/fk379/hpc-work/THESIS/model-based-mbr/PreferenceSetsSplits'
    bmwFilePath = '/rds/user/fk379/hpc-work/THESIS/model-based-mbr/PreferenceSetsSplits'

    columns = [
            "src",
            "trg",
            "chosen",
            "rejected",
        ]
    postfix = ""
 
    df = pd.DataFrame(rowsBW, columns=columns)
    df_first_90,df_last_10 = shuffle(df)
    bwFilePathTrain = os.path.join(bwFilePath,'bw_{}_train.csv'.format(dataset))
    bwFilePathTest = os.path.join(bwFilePath,'bw_{}_test.csv'.format(dataset))
    df_last_10.to_csv(bwFilePathTest, index=False)
    df_first_90.to_csv(bwFilePathTrain, index=False)

  

    df = pd.DataFrame(rowsBMW, columns=columns)
    df_first_90,df_last_10 = shuffle(df)
    bmwFilePathTrain = os.path.join(bmwFilePath,'bmw_{}_train.csv'.format(dataset))
    bmwFilePathTest = os.path.join(bmwFilePath,'bmw_{}_test.csv'.format(dataset))
    df_last_10.to_csv(bmwFilePathTest, index=False)
    df_first_90.to_csv(bmwFilePathTrain, index=False)

    df = pd.DataFrame(rowsCPS, columns=columns)
    df_first_90,df_last_10 = shuffle(df)
    cpsFilePathTrain = os.path.join(cpsFilePath,'cps_{}_train.csv'.format(dataset))
    cpsFilePathTest = os.path.join(cpsFilePath,'cps_{}_test.csv'.format(dataset))
    df_last_10.to_csv(cpsFilePathTest, index=False)
    df_first_90.to_csv(cpsFilePathTrain, index=False)'''
    

