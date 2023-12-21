from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import EarlyStoppingCallback
import transformers
from transformers.utils import logging

import torch
import deepspeed
import argparse
from torch.utils.data import RandomSampler, DataLoader
from data_set import Seq2SeqDataSet, coll_fn
import os
from shutil import copy
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, \
    set_peft_model_state_dict
from accelerate import Accelerator
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score


@dataclass
class FinetuneArguments:
    train_path: str = field(default="data/alpaca")
    model_dir: str = field(default="output")
    lora_r: int = field(default=8)
    max_len: int = field(default=300)
    max_src_len: int = field(default=150, )
    prompt_text: str = field(default='请根据以下候选人的信息，推荐一个合适的岗位JD\n', )


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='/code/train.json', type=str, help='')
    parser.add_argument('--val_path', default='/code/valid.json', type=str, help='')
    parser.add_argument('--model_dir', default="/code/belle_merge/", type=str, help='')
    parser.add_argument('--num_train_epochs', default=100, type=int, help='')
    parser.add_argument('--train_batch_size', default=1, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_recEmb/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=500, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')
    parser.add_argument('--max_src_len', type=int, default=300, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--prompt_text', type=str,
                        default="请根据以下候选人的信息，推荐一个合适的岗位JD:\n",
                        help='')
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--sample", type=int, default=-1, help="Path to deepspeed config file.")
    parser.add_argument("--learning_rate", type=float, default= 3e-4, help="Path to deepspeed config file.")
    parser.add_argument("--seed", type=int, default= 0, help="Path to deepspeed config file.")
    return parser.parse_args()


class EmbModel(nn.Module):
    def __init__(self,config_path=None,pretrained_path=None, emb_size=None):
        super(EmbModel,self).__init__()
        self.config = LlamaConfig.from_pretrained(pretrained_path,output_hidden_states=True)
        # self.pre_model = AutoModelForCausalLM.from_pretrained(pretrained_path, local_files_only=True, trust_remote_code=True)
        self.pre_model = LlamaForCausalLM.from_pretrained(pretrained_path, config=self.config)
        self.prepare_inputs_for_generation = self.pre_model.prepare_inputs_for_generation
        self.fc = nn.Linear(emb_size,1)
        
    def forward(self,input_ids,attention_mask,inputs_embeds,labels,output_attentions,output_hidden_states,return_dict,num_seg): ##直接传入字典不需要其它的
        word_embeddings = self.pre_model.get_input_embeddings()
        inputs_embeds = word_embeddings(input_ids)
        inputs_weight = F.tanh(self.fc(inputs_embeds))
        # print(np.shape(inputs_embeds))
        len_seg = len(inputs_embeds[0])//num_seg
        inputs_weight1 = inputs_weight[:,:num_seg * len_seg, :].view(-1, num_seg, len_seg)
        # print(np.shape(inputs_weight1))
        inputs_weight1 = torch.mean(inputs_weight1, dim=2).unsqueeze(-1)
        # print(np.shape(inputs_weight1))
        inputs_weight1 = inputs_weight1.repeat(1, 1, len_seg)
        # print(np.shape(inputs_weight1))
        inputs_weight1 = inputs_weight1.view(-1,num_seg * len_seg , 1)
        inputs_weight2 = inputs_weight[:,num_seg * len_seg:, :]
        # print(np.shape(inputs_weight2))
        inputs_weight = torch.cat((inputs_weight1, inputs_weight2), 1)
        # print(inputs_weight)
        inputs_embeds = inputs_embeds + 0.1 * inputs_weight * inputs_embeds
        ret_value = self.pre_model(inputs_embeds = inputs_embeds, labels=labels)
        loss = ret_value.loss
        return ret_value

def main():
    args = set_args()

    # tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    cutoff_len = args.max_len
    
    train_on_inputs =  True  # if False, masks out inputs in loss
    group_by_length =  False  # faster, but produces an odd training loss curve
    resume_from_checkpoint = None
    
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        result["input_ids"][-3] = 29973

        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        print(full_prompt)
        print('------------after tokenize---------')
        
        if not train_on_inputs:
            print("-------------------not train on inputs---------------")
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        print(tokenized_full_prompt)
        return tokenized_full_prompt
    
    # model = LlamaForCausalLM.from_pretrained(args.model_dir)
    # model.gradient_checkpointing_enable()
    # print(model)
    model = EmbModel(pretrained_path = args.model_dir,emb_size=1024*4)

    lora_target_modules = [
        "q_proj",
        "v_proj",
    ]
    
    config = LoraConfig(r=args.lora_r,
                        lora_alpha=32,
                        target_modules=lora_target_modules,
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    model = get_peft_model(model, config)
    model = model.half()#.cuda()
    model.fc.weight.requires_grad = True
    # model.cuda()
    # model = model.cuda()
    
    train_data_path = args.train_path
    val_data_path = args.val_path
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    if train_data_path.endswith(".json"):  # todo: support jsonl
        print('111111')
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        print('222222')
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)
        
    print('------------------ori val---------------')
    print(train_data_path)
    print(val_data_path)
    print(train_data)
    print(val_data)

    # print_trainable_parameters(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)
            
    seed = args.seed
    sample = args.sample
    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    val_data["train"] = val_data["train"].shuffle(seed=seed)
    # print(train_data["train"])
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    print('------------------val---------------')
    print(train_data)
    print(val_data)
    
    def compute_metrics(eval_preds):
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        print(pre[1])
        print(pre[0])
        return {'auc': auc}
    
    # logging.set_verbosity_info()
    # logger = logging.get_logger("transformers")
    def preprocess_logits_for_metrics(logits, labels):
        labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        # print(labels_index)
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        # print(gold)
        labels_index[: , 1] = labels_index[: , 1] - 1
        # print(labels_index)
        print(type(logits))
        logits, _ = logits
        logits = logits.softmax(dim=-1)
        # print(logits)
        # print(np.shape(logits))
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
        return logits[:, 1][2::3], gold[2::3]
    
    if sample > -1:
        if sample <= 128 :
            eval_step = 1
        else:
            eval_step = sample / 128 * 5


    # Define training args
    # output_dir = args.repository_id if args.repository_id else args.model_id.split("/")[-1]
    output_dir = args.output_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        warmup_steps=20,
        adam_beta1=0.9,
        adam_beta2=0.95,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=50,
        # per_device_eval_batch_size=args.per_device_eval_batch_size,
        # predict_with_generate=True,
        # generation_max_length=args.generation_max_length,
        # generation_num_beams=args.generation_num_beams,
        fp16=True,  # T5 overflows with fp16
        # bf16=args.bf16,  # Use BF16 if available
        # max_steps=20,
        num_train_epochs=args.num_train_epochs,
        deepspeed=args.deepspeed,
        # gradient_checkpointing=args.gradient_checkpointing,
        # logging & evaluation strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1,
        # evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=5,
        metric_for_best_model="eval_auc",
        load_best_model_at_end=True,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        # data_collator=coll_fn,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
    )
    model.config.use_cache = False


    # Start training
    trainer.train()

    # Save our tokenizer and create model card
    # save_dir = os.path.join(args.output_dir, "global_step-last")
    # model_engine.save_pretrained(save_dir)

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

if __name__ == "__main__":
    main()

'''
deepspeed --num_gpus=1 XX.py --deepspeed configs/ds_zero2.json
'''
