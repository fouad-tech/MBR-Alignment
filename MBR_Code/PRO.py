from dataclasses import dataclass, field
from typing import Dict, Optional
import random
import pandas as pd
import os
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, Trainer,AutoConfig ,AutoTokenizer, HfArgumentParser, TrainingArguments
from utils import *
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
import math
import argparse
import datetime
import torch.nn.functional as F
token = ''
tokenW = ''
from huggingface_hub import login
from datasets import Dataset, load_dataset
from trl import SFTTrainer

# Replace 'your_api_token' with your actual API token
login(token=token)
login(token = tokenW)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default='cnndm')
    parser.add_argument('--tag', help="ranking number", default='2')
    parser.add_argument('--model', help="model name", default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--lr', type=float, default =0.000005)
    parser.add_argument('--bsz', type=int, default=1, help='batch size')
    parser.add_argument('--grad_acc', type = int, default=1)
    parser.add_argument('--max_output_length',type=int,default=155)
    parser.add_argument('--max_prompt_length',type=int,default=1601)#have 90% of the prompts length
    parser.add_argument('--n_gpu',type=int, default=2)
    parser.add_argument('--BasePath', default='../model-based-mbr/PRO', help=' directory to save the checkpoints and results')
    parser.add_argument('--DataDir', default='../model-based-mbr/PROdata', help='path to data dir')
    
  
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
    outputDir = os.path.join(BasePath,dataset,tag)
    n_gpu = args.n_gpu
    os.makedirs(outputDir,exist_ok=True)
    

    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map='auto')
    model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
  
    x = []
    for name, param in model.named_parameters():
        if param.device not in x:
            x.append(param.device)
    print(x)

    TrainPath = os.path.join(DataDir,dataset,tag,'train.json') 
    raw_datasets = load_dataset('json',  data_files = TrainPath, split="train")
    split_dataset = raw_datasets.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    #print(train_dataset)
    #print(eval_dataset)
    # Load the JSON file
    def is_torch_xpu_available():
      from torch import xpu
    def is_torch_available():
        import torch
    def is_tf_available():
        import tensorflow
    def set_seed(seed: int, deterministic: bool = False):
            """
            Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

            Args:
                seed (`int`):
                    The seed to set.
                deterministic (`bool`, *optional*, defaults to `False`):
                    Whether to use deterministic algorithms where available. Can slow down training.
            """
            random.seed(seed)
            np.random.seed(seed)
            if is_torch_available():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # ^^ safe to call this function even if cuda is not available
                if deterministic:
                    torch.use_deterministic_algorithms(True)
            #if is_torch_mlu_available():
            #    torch.mlu.manual_seed_all(seed)
            #if is_torch_npu_available():
            #    torch.npu.manual_seed_all(seed)
            if is_torch_xpu_available():
                torch.xpu.manual_seed_all(seed)
            if is_tf_available():
                import tensorflow as tf

                tf.random.set_seed(seed)
                if deterministic:
                    tf.config.experimental.enable_op_determinism()
                    
    def seed_worker(_):
        """
        Helper function to set worker seed during Dataloader initialization.
        """
        worker_seed = torch.initial_seed() % 2**32
        set_seed(worker_seed)
    
   
    
    class PROTrainer(Trainer):
        def __init__(self, *args, max_output_length=None,max_prompt_length=None,sft_weight=0.05, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_output_length = max_output_length
            self.max_prompt_length = max_prompt_length
            self.sft_weight = sft_weight
            
        
     
        def compute_loss(self, model, inputs, return_outputs=False):
            """
            batch = [batch, training_stage, seq_len]
            """
            batch = inputs
            batch_size = batch["labels"].shape[0]
            temp_training_stage = batch["labels"].shape[1]
            sub_batches = [{key: batch[key][:,time,:] for key in ["input_ids", "attention_mask"]} for time in range(temp_training_stage)]
            #for i in range(torch.cuda.device_count()):
            #    print(f"GPU {i}:")
            #    print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            #    print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
                
            score_list = []
            suffix_mask_list = []

            for batch_index, sub_batch in enumerate(sub_batches):
                local_outputs = model(**sub_batch, output_hidden_states=True, return_dict=True)
                local_logits = local_outputs.logits #[batch, seq_len, token_num]
                local_mask = sub_batch["attention_mask"] & (~batch["prefix_mask"][:, batch_index, :]) #[batch, seq_len]
                local_labels = batch["labels"][:, batch_index, :]

                # Shift
                shift_logits = local_logits[..., :-1, :].contiguous() #[batch, seq_len-1, token_num]
                shift_logits = F.log_softmax(shift_logits, dim=2) #[batch, seq_len-1, token_num]
                shift_masks = local_mask[..., :-1] #[batch, seq_len-1]
                shift_labels = local_labels[..., 1:].view(batch_size, -1, 1) #[batch, seq_len-1, 1]

                selected_logits = torch.gather(input=shift_logits, dim=2, index=shift_labels).view(batch_size, -1) #[batch, seq_len-1]
                selected_logits[shift_masks != 1] = 0.0 #[batch, seq_len-1]
                sentence_logits = torch.sum(selected_logits, dim=1) #[batch]
                sentence_logits = sentence_logits.view(batch_size, 1)
                score_list.append(sentence_logits)
                suffix_mask_list.append(torch.sum(shift_masks, dim=1).view(batch_size, 1))
            
            sum_scores = torch.cat(score_list, dim=1) #[batch, training_stage]
            suffix_mask = torch.cat(suffix_mask_list, dim=1) #[batch, training_stage]
            scores = sum_scores / suffix_mask #[batch, training_stage]
            total_loss = 0
            for time in range(temp_training_stage - 1):
                neg_reward = batch["rewards"][:, time+1:] # [batch, training_stage-time-1]
                pos_reward = batch["rewards"][:, time] # [batch]
                
                eps = 1e-10
                neg_temperatures = pos_reward.view(-1, 1) - neg_reward # [batch, training_stage-time-1]
                pos_temperature = torch.max(neg_temperatures, dim=1).values # [batch]
                loss = torch.log(eps + torch.exp(scores[:, time] * pos_temperature) + torch.sum(torch.exp(scores[:, time+1:] * neg_temperatures), dim=1)) - scores[:, time] * pos_temperature # [batch]
                loss = torch.mean(loss).to(local_outputs.hidden_states[0].dtype)
                
                
                total_loss += loss
            
            sft_index = batch["sft_index"].view(batch_size, 1)
            sft_scores = torch.gather(input = sum_scores, dim = 1, index = sft_index).view(batch_size) #[batch]
            sft_loss = torch.mean(-sft_scores).to(local_outputs.hidden_states[0].dtype)
            sft_loss = self.sft_weight * math.pow(temp_training_stage - 1, 2) * sft_loss
            total_loss += sft_loss
            
            print(total_loss)
            return (total_loss, local_outputs) if return_outputs else total_loss

        def batch_decode(self, model_output):
            # model_output = [batch, seq_len]
            return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        
        def train_data_collator(self,features):
            samples_num = len(features)
            print('samples_num',samples_num)
            training_stage = int(tag)
            origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

            self.tokenizer.truncation_side = "left"
            ps = []
            ss = []
            rs = []
            sft_index = []
            for feature_index, feature in enumerate(features):
                for p, s, r in zip(feature['prefix'][:training_stage], feature['suffix'][:training_stage], feature['reward'][:training_stage]):
                    ps.append(p)
                    ss.append(s)
                    rs.append(r)
                assert feature["sft_index"] < training_stage
                sft_index.append(feature["sft_index"])
            
            ps = self.batch_decode(
                self.tokenizer(
                    ps,
                    max_length = self.max_prompt_length,
                    truncation = True,
                    add_special_tokens = True,
                )['input_ids']
            )

            ps_input_ids = self.tokenizer(
                ps,
                add_special_tokens = True,
            )['input_ids']
            ps_lens = [len(p_input_ids)-1 for p_input_ids in ps_input_ids]
            
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "left"
            
            texts = []

            for p, s in zip(ps, ss):
                texts.append(p + " " + s)

            
            batch = self.tokenizer(
                texts,
                padding=True,
                max_length = self.max_output_length +self.max_prompt_length,
                truncation = True,
                add_special_tokens = True,
                return_tensors = "pt",
            )
            
            seq_len = batch["attention_mask"].shape[1]
            prefix_mask = []
            for p_len in ps_lens:
                assert seq_len > p_len
                prefix_mask.append(
                    [1 if i<p_len else 0 for i in range(seq_len)]
                )
            batch["prefix_mask"] = torch.tensor(prefix_mask)
            
            batch['labels'] = batch["input_ids"].clone().detach()
            for key in batch:
                batch[key] = batch[key].view(samples_num,training_stage,-1)
            
            batch['rewards'] = torch.tensor(rs).view(samples_num, -1)
            batch['sft_index'] = torch.tensor(sft_index) # [batch]
            # restore states
            self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state

            return batch

        def get_train_dataloader(self) -> DataLoader:
                """
                Returns the training [`~torch.utils.data.DataLoader`].

                Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
                training if necessary) otherwise.

                Subclass and override this method if you want to inject some custom behavior.
                """
                if self.train_dataset is None:
                    raise ValueError("Trainer: training requires a train_dataset.")

                train_dataset = self.train_dataset
                data_collator = self.train_data_collator
                self.data_collator = data_collator
                if  isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
                else:
                    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

                dataloader_params = {
                    "batch_size": self._train_batch_size,
                    "collate_fn": data_collator,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                    "persistent_workers": self.args.dataloader_persistent_workers,
                }

                if not isinstance(train_dataset, torch.utils.data.IterableDataset):

                    print('hellooooooooo')
                    dataloader_params["sampler"] = self._get_train_sampler()
                    dataloader_params["drop_last"] = self.args.dataloader_drop_last
                    dataloader_params["worker_init_fn"] = seed_worker
                    dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

                return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    print('total_steps',math.ceil(len(train_dataset)/(args.bsz)))
    # 4. initialize training arguments:
    totalSteps = math.ceil(len(train_dataset)/(args.bsz))

    
    training_args =  TrainingArguments(
        per_device_train_batch_size=args.bsz,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        eval_strategy="steps",
        logging_first_step=True,
        logging_steps=math.ceil(totalSteps*0.1),
        eval_steps=math.ceil(totalSteps*0.1),
        output_dir=outputDir,
        save_strategy='no',
        save_steps = math.ceil(totalSteps),
        optim=args.optim,
        warmup_steps=math.ceil(totalSteps*0.1),
        gradient_checkpointing=False,
        push_to_hub=False,
        hub_strategy='checkpoint',
        num_train_epochs=1,
        save_only_model = True,
        
)
    

   

    # 5. initialize the DPO trainer
    pro_trainer = PROTrainer(
        model,
        max_prompt_length=max_prompt_length,
        max_output_length= max_output_length,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #train_dataset = raw_datasets,
        tokenizer=tokenizer,
    )

    # 6. train
    pro_trainer.train()
    #pro_trainer.save_model(outputDir)
    #dpo_trainer.model.push_to_hub(hub_id, private=True,revision=dtime, commit_message='auto commit from dpo.py')
    
    #with open('./data/model_name.txt', 'w') as f:
    #    f.write(name)
