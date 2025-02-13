import os
import argparse
import json
import llm_blender
import numpy as np
import pandas as pd
from tqdm import tqdm
from utility_func import *
from utils import (
    load_dataset,
    load_reward,
    load_samples_from_file,
    result_dir,
    reward_dir,
    prompt_dir,
)  # , approx_dir, diverse_dir
from parser import get_mbr_parser

from policy.mbr import compute_score_matrix, compute_mbr

blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") 


def getRewardList(text,prompt):
  data = {}
  for j in range(0,len(text)):
    candidates_texts = []
    inputs = []
    for i in range(0,len(text)):
      if i == j:
        continue
      
      candidates_texts.append([text[j],text[i]])
      inputs.append(prompt)
    ranks = blender.rank(inputs, candidates_texts, return_scores=True, batch_size=32)
    mean_first_column = np.mean(ranks[:, 0])
    data[str(j)] = mean_first_column
  
  return data

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

    n_lines = args.n_lines
    n_samples = args.n_samples

    epsilon = args.eps
    topk = args.topk
    topp = args.topp

    sim = args.sim
    eval_func = args.eval

    LowerRange = args.LowerRange
    UpperRange = args.UpperRange

    prompt_path = args.prompt
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()

    # Algorithm config
    algorithm = args.algorithm
    recompute_matrix = args.recompute_matrix
    
    compute_similarity, similarity = load_similarity(sim)
    compute_evaluate, evaluator = load_evaluate(eval_func, sim, similarity)

    src_lines = load_dataset(dataset)  # src is used only by comet and clip.
    trg_lines = load_dataset(dataset, ref=True)

    # client = boto3.client("s3")

    model_n = os.path.basename(model_name)

    os.makedirs(os.path.join(reward_dir, dataset, model_n), exist_ok=True)

    files = sorted(os.listdir(sample_dir))

    filtered_files = load_samples_from_file(files, epsilon, topk, topp)

    assert len(filtered_files) > 0
    #print('filtered_files size before ',len(filtered_files))
    #filtered_files= filtered_files[677:832]
    #print('filtered_files size after ',len(filtered_files))
    print("first 10 files=", filtered_files[:10])
    rows = []

    for fileId in tqdm(range(LowerRange,UpperRange)):
        filename = filtered_files[fileId]
        sample_id = int(filename.split("_")[0])
        assert "{:04}".format(sample_id) in filename

        if sample_id >= n_lines:
            break

        src_input = src_lines[sample_id]
        trg = trg_lines[sample_id]

        if dataset == 'squad_v2':
          input_source = prompt+' '+src_input
        else:
          input_source = prompt.replace("[[QUESTION]]", src_input)
        
       
         
        df = pd.read_csv(os.path.join(sample_dir, filename))

        assert len(df) >= n_samples
        df = df[:n_samples]

        df.fillna(
            "", inplace=True
        )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
        hyp = df.iloc[:]["text"]
        
        if not recompute_matrix:
            print('not recomputing')
            reward_dict = load_reward(
                os.path.join(reward_dir, dataset, model_n), filename+'_reward', sim, n_samples
            )
            matrix = 0
        else:
            matrix = None
        if matrix is None:
            reward_filename = filename + '_reward' + "_" + sim + "_" + str(n_samples)
            reward_path = os.path.join(reward_dir, dataset, model_n, reward_filename)
            reward_dict = getRewardList(hyp,input_source)
            rowsReward = list(reward_dict.items())
            columnsReward = ['index','reward']
            dfReward = pd.DataFrame(rowsReward, columns=columnsReward)
            dfReward.to_csv(reward_path,index=False)
   

        if algorithm in ["None"]:

            ed_best = max(reward_dict, key=reward_dict.get)
            ed_best=int(ed_best)
            
            ed_score = compute_score(df, ed_best, trg, compute_evaluate, src=src_input)
            ed_hyp = df.iloc[ed_best]["text"]
            
            row = [
                sample_id,
                trg,
                ed_hyp,
                ed_score,
                ed_best,
            ]
        else:
            assert False
        rows.append(row)

    if algorithm == "None":
        columns = [
            "sample_id",
            "trg",
            "ed_hyp",
            "ed_score",
            "ed_best",
        ]
        postfix = ""
    else:
        assert False

    df = pd.DataFrame(rows, columns=columns)

    filename = "reward_{}_{}_{:03}_{:.2f}_{:02d}_{:.2f}_{}_{}{}_{}-{}.csv".format(
        dataset, model_n, n_samples, epsilon, topk, topp, sim, eval_func, postfix,LowerRange,UpperRange
    )

    df_path = os.path.join(result_dir, filename)
    os.makedirs(result_dir, exist_ok=True)
    df.to_csv(df_path, index=False)


