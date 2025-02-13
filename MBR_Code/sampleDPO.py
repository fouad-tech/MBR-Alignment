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
    sample_dir
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
    parser.add_argument('--dataset', help="dataset name", default='cnndm')
    parser.add_argument('--tag', help="dataset tag", default='bw')
    parser.add_argument('--model_ref',default='mistral')
    parser.add_argument('--model_path',default='../model-based-mbr/DPO/cnndm/bw/sig/BETA0.1/checkpoint-1800')
    parser.add_argument('--prompt', default='promptCNN.txt')
    parser.add_argument('--LowerRange',default=0,type = int)
    parser.add_argument('--UpperRange',default=1000,type=int)
    parser.add_argument('--eps',default=0.02,type = float)
    parser.add_argument('--top_p',default=1.0,type=int)
    parser.add_argument('--top_k',default=0,type = int)

    


    args = parser.parse_args()
    do_sample = True
    dataset = args.dataset
    tag = args.tag
    model_ref= args.model_ref
    model_path = args.model_path
    prompt_path=args.prompt
    UpperRange = args.UpperRange
    LowerRange = args.LowerRange    
    eps = args.eps
    topk = args.top_k
    topp=args.top_p

    n_batches = 2
    bsz = 16
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()
    print(prompt)
    # Algorithm config
    src_lines = load_dataset(dataset)

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
    model_kwargs = load_kwargs(dataset)

    os.makedirs(os.path.join(sample_dir, dataset, tag), exist_ok=True)
    for sample_id in tqdm(range(LowerRange,UpperRange)):
            if sample_id > len(src_lines):
                break

        
                
            if "zephyr" in model_ref:
                    # Zero shot prompting.
                    # TODO: Implement few shot prompting.
                    messages = [
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": src_lines[sample_id],
                        },
                    ]
                    input_source = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if sample_id == 0:
                        print("input_source=", input_source)
            
            else:
                    input_source = prompt.replace("[[QUESTION]]", src_lines[sample_id])
                    input_source = "[INST] " + input_source + "[/INST]"
                        

            model_inputs = tokenizer(
                    input_source, return_tensors="pt", return_token_type_ids=False
                ).to(model.device)
            input_length = model_inputs["input_ids"].shape[1]
            stopping_criteria = None
            set_seed(42)

            rows = []
            for i in range(n_batches):
                if do_sample:
                
                        sample_output = model.generate(
                        **model_inputs,
                        **model_kwargs,
                        do_sample=True,
                        epsilon_cutoff=eps,
                        top_k=topk,
                        top_p=topp,
                        num_beams=1,
                        num_return_sequences=bsz,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True,
                        forced_bos_token_id=model.config.forced_bos_token_id,
                    )
                else:
                    print('not sampling 1')
                    num_beam_groups = 1

                    sample_output = model.generate(
                        **model_inputs,
                        **model_kwargs,
                        do_sample=False,
                        epsilon_cutoff=eps,
                        top_k=topk,
                        top_p=topp,
                        num_beams=5,
                        num_return_sequences=1,
                        num_beam_groups=num_beam_groups,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True,
                        forced_bos_token_id=model.config.forced_bos_token_id,
                    )
                    
                
            
            
                    
                output_prob = compute_probability_lm(model, sample_output)
                output_text = get_texts(tokenizer, sample_output, input_length)
                        
                for j in range(bsz):
                    rows.append((output_text[j], output_prob[j]))
            
                
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(
                sample_id, eps, topk, topp
            )

            outfilepath = os.path.join(sample_dir, dataset, tag, filename)

            df = pd.DataFrame(rows, columns=["text", "probability"])
            df.to_csv(outfilepath, index=False)
 


