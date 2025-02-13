from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
import datasets
import os 
import torch
import pandas as pd
import numpy as np
from evaluate import load


def get_texts(tokenizer, outputs, input_length):
    """
    This function is only compatible with langauge models. not for seq2seq
    """
    bsz = outputs.sequences.shape[0]
    output_texts = []
    for b in range(bsz):
        output_text = tokenizer.decode(
            outputs.sequences[b][input_length:], skip_special_tokens=True
        )
        output_texts.append(output_text)
    return output_texts


def compute_probability_lm(model, outputs):
    """
    This compute_prob function is compatible with langauge models.
    Doesn't work on seq2seq models.
    """

    # transition_scores = model.compute_transition_scores(
    #     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
    # )
    transition_scores = (
        model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        .cpu()
        .to(torch.float32)
    )

    seq_prob = torch.ones(transition_scores.shape[0]).to(torch.float32)
    for i in range(transition_scores.shape[1]):
        seq_prob *= np.exp(transition_scores[:, i])

    return seq_prob.numpy()

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
    parser.add_argument('--dataset', help="dataset name", default='squad_v2')
    parser.add_argument('--tag', help="dataset tag", default='2')
    parser.add_argument('--model_type', default='PRO')
    parser.add_argument('--model_ref',default='zephyr')
    parser.add_argument('--model_path',default='../PRO/checkpoints/index_1/stage_2/squad_v2T/epoch_0')
    parser.add_argument('--sim', default='bertscore', help='simalrity metrics')
    parser.add_argument('--eval', nargs='+', help='evaluation metrics')
    parser.add_argument('--prompt', default='SquadV2.txt')
    parser.add_argument('--LowerRange',default=0,type = int)
    parser.add_argument('--UpperRange',default=1000,type=int)
    parser.add_argument('--eps',default=0.02,type = int)
    parser.add_argument('--top_p',default=1.0,type=int)
    parser.add_argument('--top_k',default=0,type = int)
    parser.add_argument('--beam',type=int,default=1)
    parser.add_argument('--recompute',type=int,default=1)
    


    args = parser.parse_args()

    sim = args.sim
    eval_func = args.eval
    dataset = args.dataset
    tag = args.tag
    model_type=args.model_type
    prompt_path=args.prompt
    UpperRange = args.UpperRange
    LowerRange = args.LowerRange
    model_ref= args.model_ref
    eps = args.eps
    top_k = args.top_k
    top_p=args.top_p
    beam = args.beam
    recompute = args.recompute

    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()
    print(prompt)
    # Algorithm config
    compute_similarity, similarity = load_similarity(sim)
    compute_distance = load_distance(sim, compute_similarity)
    compute_evaluate = []
    for e in eval_func:
          c,_ = load_evaluate(e, sim, similarity)
          compute_evaluate.append(c)
    

    
    src_lines = load_dataset(dataset) 
    trg_lines = load_dataset(dataset, ref=True)

    #defining the model path
    #'./DPO/cnndm/bw/sig/BETA0.1'
    #model_path = os.path.join('./', model_type,dataset,tag,loss_type,"BETA{}".format(beta))
    #f = [int(i.split('-')[1]) for i in os.listdir(model_path) if '-' in i]
    #f = max(f)
    #model_path = os.path.join(model_path,'checkpoint-{}'.format(f))
    model_path = args.model_path
    print("DHDHDHDHDHDHDHDHHDHDDH")
    print(model_path)
    print("DHDHDHDHDHDHDHDHHDHDDH")
    model = AutoModelForCausalLM.from_pretrained(
                model_path, load_in_4bit=True, device_map="auto"
            )
    tokenizer = AutoTokenizer.from_pretrained(
                model_path, padding_size="left", use_fast=True
            )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    #creating the results dir
    #'./DPO_results/cnndm/bw/BETA0.1'
    resultsDir = os.path.join('./', model_type+'_results', dataset,tag)
    os.makedirs(resultsDir, exist_ok=True)

    metricsPath = os.path.join(resultsDir,'{}_{}.csv'.format(eval_func,UpperRange))
    samplePath = os.path.join(resultsDir,'samples_{}.csv'.format(UpperRange))

    rows = []
    texts = []
    model_kwargs = load_kwargs(dataset)
    if recompute:
        for sample_id in tqdm(range(LowerRange,UpperRange,4)):

            src_input = src_lines[sample_id:sample_id+4]
            trg = trg_lines[sample_id:sample_id+4]
            
            

            
            if "zephyr" in model_ref:
                       
                        # Zero shot prompting.
                        # TODO: Implement few shot prompting.
                        messages = [[
                            {
                                "role": "system",
                                "content": prompt,
                            },
                            {
                                "role": "user",
                                "content": dataset_line,
                            },
                        ]for dataset_line in src_input]

                        input_source = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
            else:
                        input_source = [prompt.replace("[[QUESTION]]", dataset_line) for dataset_line in src_input]
                        input_source = ["[INST] " + dataset_line + "[/INST]" for dataset_line in input_source]
                
            model_inputs = tokenizer(
                        input_source, return_tensors="pt", return_token_type_ids=False,padding = True,
                    ).to(model.device)
            print('mode device',model.device)
            input_length = model_inputs["input_ids"].shape[1]
                    
            set_seed(42)      
            if beam:
                        sample_output = model.generate(
                                **model_inputs,
                                **model_kwargs,
                                do_sample=False,
                                num_beams=5,
                                num_return_sequences=1,
                                num_beam_groups=1,
                                stopping_criteria=None,
                                return_dict_in_generate=True,
                                output_scores=True,
                                forced_bos_token_id=model.config.forced_bos_token_id,
                                
                            )
            else:
                        sample_output = model.generate(
                                **model_inputs,
                                **model_kwargs,
                                do_sample=True,
                                eps=eps,
                                top_k=top_k,              # Disables top-k sampling
                                top_p=top_p,            # Disables top-p (nucleus) sampling
                                num_return_sequences=1,
                                stopping_criteria=None,
                                return_dict_in_generate=True,
                                output_scores=True,
                                forced_bos_token_id=model.config.forced_bos_token_id,
                                
                            )

            output_text = get_texts(tokenizer, sample_output, input_length)
            t = len(output_text)  
            text=[]
            print('OUTPUT TEXT LENGTH',t)
            print('target TEXT LENGTH',len(trg))
            for j in range (0,t): 
                    trg[j]
                    output_text[j]
                    text.append((trg[j],output_text[j]))
                    texts.append((trg[j],output_text[j]))
            
            if os.path.exists(samplePath):
                    df = pd.DataFrame(text, columns=["target", "hypothesis"])
                    df.to_csv(samplePath, mode='a', header=False, index=False)
            else:
                df = pd.DataFrame(text, columns=["target", "hypothesis"])
                df.to_csv(samplePath, index=False)  
                  

    else:
            print('not recompute')
            df = pd.read_csv(samplePath)
            df.fillna(
                "", inplace=True
            )  # TODO: This is needed to remove empty strings. In reality empty strings can be ignored. probably it's better to drop.
            for row in df.itertuples(index=False, name=None):
                texts.append(list(row))
            print(len(texts))

              
              
    for item in texts:

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
