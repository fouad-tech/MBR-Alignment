from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import os
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoConfig ,AutoTokenizer, HfArgumentParser, TrainingArguments
from utils import *
from trl import DPOTrainer,DPOConfig
import math
import argparse
import datetime
token = ''
tokenW = ''
from huggingface_hub import login

# Replace 'your_api_token' with your actual API token
login(token=token)
login(token = tokenW)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default='cnndm')
    parser.add_argument('--tag', help="dataset name", default='bw')
    parser.add_argument('--model', help="model name", default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--quantize', type=int, default=-1)
    parser.add_argument('--samples', type=int, default=32)
 
    parser.add_argument('--beta', type=float, default=0.1, help='beta for DPO')
    parser.add_argument('--optim', type=str, default='rmsprop')
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--loss_type', type=str, default='sigmoid', help='loss type for DPO: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid')

    parser.add_argument('--bsz', type=int, default=4, help='batch size')
    parser.add_argument('--grad_acc', type = int, default=1)
    parser.add_argument('--max_output_length',type=int,default=155)
    parser.add_argument('--max_prompt_length',type=int,default=1601)#have 90% of the prompts length

    parser.add_argument('--BasePath', default='../model-based-mbr/DPO_KD', help=' directory to save the checkpoints and results')
    parser.add_argument('--DataDir', default='../model-based-mbr/PreferenceSetsSplits', help='path to data dir')
    parser.add_argument('--prompt', default='promptCNN.txt')
    dt_now = datetime.datetime.now()
    dtime = dt_now.strftime('%Y%m%d-%H%M%S')
    print('time=', dtime)
    
    args = parser.parse_args()
    prompt_path = args.prompt
    model_name = args.model
    dataset = args.dataset
    BasePath = args.BasePath
    tag = args.tag
    samples = args.samples
    DataDir= args.DataDir#os.path.join(args.DataDir,dataset,model_name.split('/')[-1],tag.upper())
    max_output_length = args.max_output_length
    max_prompt_length = args.max_prompt_length
    outputDir = os.path.join(BasePath,dataset,tag,args.loss_type[:3], "BETA{}".format(args.beta))
    os.makedirs(outputDir,exist_ok=True)
    name = "{}_{}_{}_b-{}_r-{}_lr-{}_lt-{}".format(model_name.split('/')[1],
                                                        args.tag,
                                                        args.dataset,
                                                        args.beta,
                                                        args.lora_r,
                                                        args.lr,
                                                        args.loss_type[:3])

    hub_id = "FouadAI" + '/' + name

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 load_in_4bit=(args.quantize == 4), 
                                                 load_in_8bit=(args.quantize == 8))
    model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()
  

    #input,accepted,rejected,target



    TrainFile = os.path.join(DataDir,'{}_{}_{}_kd_train.csv'.format(tag,dataset,samples)) 
    print(TrainFile)
    dfTrain = pd.read_csv(TrainFile)

    trainingData = {
         'prompt':[],
         'chosen':[],
         'rejected':[]
    }

    for index, row in dfTrain.iterrows():
    
        trainingData['prompt'].append(row['src'])
        trainingData['chosen'].append(row['chosen'])
        trainingData['rejected'].append(row['rejected'])

    print(len(trainingData['prompt']))
    print(len(trainingData['chosen']))
    print(len(trainingData['rejected']))
    train_size = len(trainingData['prompt'])
    
    ValidFile = os.path.join(DataDir,'{}_{}_{}_kd_test.csv'.format(tag,dataset,samples)) 
    dfValid = pd.read_csv(ValidFile)

    ValidationData = {
         'prompt':[],
         'chosen':[],
         'rejected':[]
    }
    

    for index, row in dfValid.iterrows():
         
        ValidationData['prompt'].append(row['src'])
        ValidationData['chosen'].append(row['chosen'])
        ValidationData['rejected'].append(row['rejected'])

    print(len(ValidationData['prompt']))
    print(len(ValidationData['chosen']))
    print(len(ValidationData['rejected']))
    
    eval_dataset=Dataset.from_dict(ValidationData)
    train_dataset=Dataset.from_dict(trainingData)
    
    print('total_steps',math.floor(train_size/(args.bsz*2)))
    # 4. initialize training arguments:
    totalSteps = math.floor(train_size/(args.bsz*2))

    
    training_args = DPOConfig(
        per_device_train_batch_size=args.bsz,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=math.ceil(totalSteps*0.1),
        eval_steps=math.ceil(totalSteps*0.3),
        output_dir=outputDir,
        save_strategy='steps',
        save_steps= totalSteps,
        optim=args.optim,
        warmup_steps=math.ceil(totalSteps*0.1),
        gradient_checkpointing=False,
        push_to_hub=False,
        hub_model_id=hub_id,
        hub_strategy='checkpoint',
        num_train_epochs=1
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_output_length+max_prompt_length,
        max_target_length=max_output_length,
        max_prompt_length=max_prompt_length,
        peft_config=None,
    )

    # 6. train
    dpo_trainer.train()
    #dpo_trainer.save_model(outputDir)
    #dpo_trainer.model.push_to_hub(hub_id, private=True,revision=dtime, commit_message='auto commit from dpo.py')
    
    #with open('./data/model_name.txt', 'w') as f:
    #    f.write(name)
