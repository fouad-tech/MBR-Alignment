from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import os
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoConfig ,AutoTokenizer, HfArgumentParser, TrainingArguments
from utils import *
from trl import KTOTrainer,KTOConfig
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
    parser.add_argument('--tag', help="dataset ratio", default='1:1')
    parser.add_argument('--model', help="model name", default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--nu', type=float, default=1, help='weight for undesirable')
    parser.add_argument('--nd', type=float, default=1.33, help='weight for desirable')
    parser.add_argument('--beta', type=float, default=0.1, help='beta for DPO')
    parser.add_argument('--optim', type=str, default='rmsprop')
    parser.add_argument('--lr', type=float, default =0.000001)
    parser.add_argument('--bsz', type=int, default=4, help='batch size')
    parser.add_argument('--grad_acc', type = int, default=1)
    parser.add_argument('--max_output_length',type=int,default=155)
    parser.add_argument('--max_prompt_length',type=int,default=1601)#have 90% of the prompts length

    parser.add_argument('--BasePath', default='../model-based-mbr/KTO', help=' directory to save the checkpoints and results')
    parser.add_argument('--DataDir', default='../model-based-mbr/PreferenceSetsSplitsKTO', help='path to data dir')
    
  
    dt_now = datetime.datetime.now()
    dtime = dt_now.strftime('%Y%m%d-%H%M%S')
    print('time=', dtime)
    
    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    BasePath = args.BasePath
    tag = args.tag
    DataDir= args.DataDir
    max_output_length = args.max_output_length
    max_prompt_length = args.max_prompt_length
    outputDir = os.path.join(BasePath,dataset,tag, "BETA{}".format(args.beta))
    os.makedirs(outputDir,exist_ok=True)
    

    model = AutoModelForCausalLM.from_pretrained(model_name,
                   torch_dtype=torch.bfloat16 )
    model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
  

    TrainFile = os.path.join(DataDir,'{}_{}_train.csv'.format(dataset,tag)) 
    print(TrainFile)
    dfTrain = pd.read_csv(TrainFile)



    trainingData = {
         'prompt':[],
         'completion':[],
         'label':[]
    }

    for index, row in dfTrain.iterrows():
    
        trainingData['prompt'].append(row['src'])
        trainingData['completion'].append(row['sample'])
        if row['type'] == 'F':
            trainingData['label'].append(False)
        else:
            trainingData['label'].append(True)

    print(len(trainingData['prompt']))

    train_size = len(trainingData['prompt'])
    
    ValidFile = os.path.join(DataDir,'{}_{}_test.csv'.format(dataset,tag)) 
    dfValid = pd.read_csv(ValidFile)

    ValidationData = {
         'prompt':[],
         'completion':[],
         'label':[]
    }
    

    for index, row in dfValid.iterrows():
        
        ValidationData['prompt'].append(row['src'])
        ValidationData['completion'].append(row['sample'])
        if row['type'] == 'F':
            ValidationData['label'].append(False)
        else:
            ValidationData['label'].append(True)

    print(len(ValidationData['prompt']))

    
    eval_dataset=Dataset.from_dict(ValidationData)
    train_dataset=Dataset.from_dict(trainingData)
    
    print('total_steps',math.floor(train_size/(args.bsz*2)))
    # 4. initialize training arguments:
    totalSteps = math.floor(train_size/(args.bsz*2))

    
    training_args =  KTOConfig(
        per_device_train_batch_size=args.bsz,
        desirable_weight=args.nd,
        undesirable_weight=args.nu,
        beta = args.beta,
        max_prompt_length=max_prompt_length,
        max_completion_length= max_output_length,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=math.ceil(totalSteps*0.1),
        eval_steps=math.ceil(totalSteps*0.1),
        output_dir=outputDir,
        save_strategy='steps',
        save_steps = math.ceil(totalSteps),
        optim=args.optim,
        warmup_steps=math.ceil(totalSteps*0.1),
        gradient_checkpointing=False,
        push_to_hub=False,
        hub_strategy='checkpoint',
        num_train_epochs=1,
        save_only_model = True
    )

   

    # 5. initialize the DPO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        #max_length=max_output_length+max_prompt_length,
        #max_target_length=max_output_length,
        #max_prompt_length=max_prompt_length,
        #peft_config=None,
    )

    # 6. train
    kto_trainer.train()
    #kto_trainer.save_model(outputDir)
    #dpo_trainer.model.push_to_hub(hub_id, private=True,revision=dtime, commit_message='auto commit from dpo.py')
    
    #with open('./data/model_name.txt', 'w') as f:
    #    f.write(name)
