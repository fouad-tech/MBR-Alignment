from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
import os
import torch
import datasets
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoConfig ,AutoTokenizer, HfArgumentParser, TrainingArguments
from utils import *
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM,SFTConfig,setup_chat_format
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
    parser.add_argument('--model', help="model name", default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--tag',default="bw")
    parser.add_argument('--bsz', type=int, default=4, help='batch size')
    parser.add_argument('--max_output_length',type=int,default=155)
    parser.add_argument('--max_prompt_length',type=int,default=1601)#have 90% of the prompts length
    parser.add_argument('--optim', type=str, default='rmsprop')
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--grad_acc', type = int, default=1)
    parser.add_argument('--BasePath', default='../model-based-mbr/SFT', help=' directory to save the checkpoints and results')
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
    samples = args.samples
    tag = args.tag
    optim = args.optim
    lr = args.lr
    DataDir= args.DataDir#os.path.join(args.DataDir,dataset,model_name.split('/')[-1],tag.upper())
    max_output_length = args.max_output_length
    max_prompt_length = args.max_prompt_length
    outputDir = os.path.join(BasePath,dataset)
    os.makedirs(outputDir,exist_ok=True)
    name = "{}_{}".format(model_name.split('/')[1], args.dataset)

    hub_id = "FouadAI" + '/' + name

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()
  

    TrainFile = os.path.join(DataDir,'{}_{}_{}_train.csv'.format(tag,dataset,samples)) 
    print(TrainFile)
    dfTrain = pd.read_csv(TrainFile)

    trainingData = {
         'text':[],
         'instruction':[],
         'output':[]
        }
    
    for index, row in dfTrain.iterrows():
        if  'mistralai' in model_name:
            text_row = "<s>[INST] {} [/INST] \\n {} </s>".format(row['src'],row["chosen"])
        else:
            text_row = "<|system|> \\n {} \\n <|user|> \\n {} \\n </s>".format(prompt,row["src"].replace(prompt,''))

        trainingData['text'].append(text_row)
        trainingData['instruction'].append(row['src'])
        trainingData['output'].append(row["chosen"])
           

   
    train_size = len(trainingData['text'])
    print(trainingData['text'][0])
    ValidFile = os.path.join(DataDir,'{}_{}_{}_test.csv'.format(tag,dataset,samples)) 
    dfValid = pd.read_csv(ValidFile)

    ValidationData = {
        'text':[],
        'instruction':[],
         'output':[]
    }
    
    
    for index, row in dfValid.iterrows():
        if  'mistralai' in model_name: 
            text_row = "<s>[INST] {} [/INST] \\n {} </s>".format(row['src'],row["chosen"])
        else:
            text_row = " {}  \\n {} </s>".format(row['src'],row["chosen"])
            ValidationData['text'].append(text_row)
            ValidationData['instruction'].append(row['src'])
            ValidationData['output'].append(row["chosen"])



    
    eval_dataset=Dataset.from_dict(ValidationData)
    train_dataset=Dataset.from_dict(trainingData)
    print(train_dataset)
    print('total_steps',math.floor(train_size/(args.bsz*2)))
    # 4. initialize training arguments:
    totalSteps = math.floor(train_size/(args.bsz*2))
    print(totalSteps)

    sft_config = SFTConfig(
        per_device_train_batch_size=args.bsz, #
        gradient_accumulation_steps=args.grad_acc, #
        learning_rate=args.lr, #
        evaluation_strategy="steps", #
        logging_first_step=True, #
        save_strategy='no', #
        logging_steps=math.ceil(totalSteps*0.1), #
        eval_steps=math.ceil(totalSteps*0.1), #
        output_dir=outputDir, #
        optim=args.optim, #
        warmup_steps=math.ceil(totalSteps*0.1),
        gradient_checkpointing=False,
        push_to_hub=False, #
        hub_model_id=hub_id,
        hub_strategy='checkpoint',
        num_train_epochs=1, #
   

)

    # 5. initialize the DPO trainer
    sft_trainer = SFTTrainer(
        model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_output_length+max_prompt_length,
        peft_config=None,
        dataset_text_field='text',
        packing=False
        
    )

    # 6. train
    sft_trainer.train()
    sft_trainer.save_model(outputDir)

    #dpo_trainer.model.push_to_hub(hub_id, private=True,revision=dtime, commit_message='auto commit from dpo.py')
    
    #with open('./data/model_name.txt', 'w') as f:
    #    f.write(name)

