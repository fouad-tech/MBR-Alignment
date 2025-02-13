
import os

import numpy as np
from transformers import set_seed
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from parser import get_mbr_parser
from utils import load_model, load_dataset, load_kwargs
from utils import sample_dir, prompt_dir
from evaluate import load
# import boto3


def getBestTrg(hyp, ref):
            evaluator = load('meteor')
            scores = [
                (evaluator.compute(predictions=[hyp], references=[r])["meteor"],r)
                for r in ref
            ]
            return max(scores)
            

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


def compute_evaluate(hyp, ref):
            evaluator = load('rouge')
            rougeL = evaluator.compute(predictions=[hyp], references=[[ref]])["rougeL"]
            rouge1 = evaluator.compute(predictions=[hyp], references=[[ref]])["rouge1"]
            rouge2 = evaluator.compute(predictions=[hyp], references=[[ref]])["rouge2"]
            evaluator = load('sacrebleu')
            bleu =evaluator.compute(predictions=[hyp], references=[ref])["score"]
            #evaluator = load('bleurt', checkpoint="BLEURT-20")
            #bleurt =evaluator.compute(predictions=[hyp], references=[ref])["scores"][0]

            return rougeL, rouge1, rouge2, bleu

def sample(
    dataset,
    tokenizer,
    model,
    src_lines,
    trg_lines,
    eps,
    topk,
    topp,
    prompt,
    LowerRange,
    UpperRange,
    outfilepath,
    beam,
    beam_size
):
    
    rougeLScore = []
    rouge2Score = []
    bleuScore = []
    rouge1Score = []
    meteor = []
    model_kwargs = load_kwargs(dataset)
    results = []
    bleurtScore = []
    print(model.device)
    for sample_id in tqdm(range(LowerRange,UpperRange,5)):
            if 'zephyr' in model_name:
                input_source = []
                batch_src_lines = src_lines[sample_id:sample_id+5]
                for batch_src_line in batch_src_lines:
                    messages = [
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": batch_src_line,
                        },
                    ]
                    input_source.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    ))


                model_inputs = tokenizer(
                input_source, return_tensors="pt", return_token_type_ids=False,padding=True).to(model.device)
                input_length = model_inputs["input_ids"].shape[1]
                stopping_criteria = None

            elif 'Meta' in model_name:
                input_source = []
                batch_src_lines = src_lines[sample_id:sample_id+4]        
                for batch_src_line in batch_src_lines:
                        messages = [
                            {
                                "role": "system",
                                "content": prompt,
                            },
                            {
                                "role": "user",
                                "content": batch_src_line,
                            },
                        ]
                        
                        input_source.append(tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True,
                                ))

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                model_inputs = tokenizer(
                input_source, return_tensors="pt", return_token_type_ids=False,padding=True).to(model.device)
                input_length = model_inputs["input_ids"].shape[1]
                stopping_criteria = None

            else:

                batch_src_lines = src_lines[sample_id:sample_id+5]
                input_source = ["[INST] " + prompt.replace("[[QUESTION]]", line) + "[/INST]" for line in batch_src_lines]

                model_inputs = tokenizer(
                    input_source, return_tensors="pt", return_token_type_ids=False,padding=True
                ).to(model.device)
                input_length = model_inputs["input_ids"].shape[1]
                stopping_criteria = None

            set_seed(42)       
            if beam:
                if "Meta" in model_name:
                        print('sampling from meta')
                        sample_output = model.generate(
                        **model_inputs,
                        **model_kwargs,
                        do_sample=False,
                        num_beams=beam_size,
                        num_return_sequences=1,
                        num_beam_groups=1,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True,
                        forced_bos_token_id=model.config.forced_bos_token_id,

                    )
                else:
                    print(beam_size)
                    sample_output = model.generate(
                        **model_inputs,
                        **model_kwargs,
                        do_sample=False,
                        num_beams=beam_size,
                        num_return_sequences=1,
                        num_beam_groups=1,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True,
                        forced_bos_token_id=model.config.forced_bos_token_id,
                        
                    )
            else:
                 sample_output = model.generate(
                        **model_inputs,
                        **model_kwargs,
                        do_sample=True,
                        eps=0.02,
                        top_k=0,              # Disables top-k sampling
                        top_p=1.0,            # Disables top-p (nucleus) sampling
                        num_return_sequences=1,
                        stopping_criteria=stopping_criteria,
                        return_dict_in_generate=True,
                        output_scores=True,
                        forced_bos_token_id=model.config.forced_bos_token_id,
                        
                    )
     
            #output_prob = compute_probability_lm(model, sample_output)
            output_text = get_texts(tokenizer, sample_output, input_length)
            t = len(output_text)            
            for j in range (0,t):
                if dataset == 'squad_v2' or dataset == 'common_gen':
                    bestTrg = getBestTrg(output_text[j],trg_lines[sample_id+j])
                    results.append((bestTrg[1],output_text[j]))
                    print(bestTrg[1])
                else:
                    results.append((trg_lines[sample_id+j],output_text[j]))
                
    samplePath = os.path.join(outfilepath,'Samples_{}.csv'.format(UpperRange))
    df = pd.DataFrame(results, columns=['target','text'])
    df.to_csv(samplePath, index=False)


if __name__ == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default='cnndm')
    parser.add_argument('--model', help="model name", default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--prompt',default='promptCNN.txt')
    parser.add_argument('--UpperRange',default=1000,type=int)
    parser.add_argument('--LowerRange',default=0,type=int)
    parser.add_argument('--beamSize',default=1,type=int)
    
    args = parser.parse_args()
  
    
        
    #dataset = 'squad_v2'
    #model_name = 'HuggingFaceH4/zephyr-7b-beta'
    #prompt_path = 'SquadV2.txt'
    #dataset = 'strategyqaV'
    #model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    #prompt_path = 'promptStrategyqamMistral.txt'
    #dataset = 'cnndm'
    #model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    #prompt_path = 'promptCNN.txt'

    prompt_path = args.prompt
    model_name = args.model
    dataset = args.dataset
    UpperRange = args.UpperRange
    eps = 0.02
    topk = 0
    topp = 1.0
    LowerRange = args.LowerRange
    beam=True
    beam_size = [args.beamSize]
    src_lines = load_dataset(dataset)
    trg_lines = load_dataset(dataset,ref=True)
    tokenizer, model, model_name, stop_tokens = load_model(
        dataset, torch_device, model_name
    )

    
    
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()
    for i in beam_size:
            if dataset == 'squad_v2':
                outfilepath = '../model-based-mbr/resultsMetrics/Beam{}Mistralsquadv2'.format(i)
            elif dataset == 'cnndm':
                outfilepath = '../model-based-mbr/resultsMetrics/Beam{}MistralCNN'.format(i)
            else:
                outfilepath = '../model-based-mbr/resultsMetrics/Beam{}Mistralstrategy'.format(i)

            sample(
            dataset,
            tokenizer,
            model,
            src_lines,
            trg_lines,
            eps,
            topk,
            topp,
            prompt,
            LowerRange,
            UpperRange,
            outfilepath,
            beam,
            i
            )

        
    