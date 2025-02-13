import os

import numpy as np
from transformers import set_seed
import torch
from tqdm import tqdm
import pandas as pd

from parser import get_mbr_parser
from utils import load_model, load_dataset, load_kwargs, StoppingCriteriaSub
from utils import sample_dir, prompt_dir

# import boto3


def compute_probability_s2s(sample_output):
    """
    This compute_prob function is compatible with seq2seq models.
    Doesn't work on language models.
    """
    bsz = sample_output.sequences.shape[0]
    probs = np.array([1.0] * bsz)
    # terms = [False] * bsz
    for i in range(len(sample_output.scores)):
        p = np.array([1.0] * bsz)
        for b in range(bsz):
            if hasattr(tokenizer, "pad_token_id"):
                if sample_output.sequences[b][i + 1] == tokenizer.pad_token_id:
                    continue
            log_probs = torch.nn.functional.log_softmax(
                sample_output.scores[i][b], dim=-1
            )
            p[b] = torch.exp(log_probs[sample_output.sequences[b][i + 1]])
        probs *= p
        # print('p=', p)
    return probs


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


def sample(
    dataset,
    tokenizer,
    model,
    trg_lines,
    torch_device,
    LowerRange,
    UpperRange
):
    lens = []
    maxLen = 0
    for sample_id in tqdm(range(LowerRange,UpperRange)):
            
            model_inputs = tokenizer(
                trg_lines[sample_id], return_tensors="pt", return_token_type_ids=False
            ).to(model.device)
            input_length = model_inputs["input_ids"].shape[1]
            lens.append(input_length)
            if input_length> maxLen:
                maxLen=input_length



    percentile_90 = np.percentile(lens, 90)
    percentile_50 = np.percentile(lens, 50)
    percentile_75 = np.percentile(lens, 75)
    print('percentile_75 ', percentile_75)
    print('percentile_50 ', percentile_50)
    print('percentile_90 ', percentile_90)            



   
    


if __name__ == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    
    dataset = 'strategyqaT'
    trg_lines = load_dataset(dataset,ref=True)
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    LowerRange = 0
    UpperRange = 1832
    tokenizer, model, model_name, stop_tokens = load_model(
        dataset, torch_device, model_name
    )

    sample(
        dataset,
        tokenizer,
        model,
        trg_lines,
        torch_device,
        LowerRange=LowerRange,
        UpperRange=UpperRange
    )
