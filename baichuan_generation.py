import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
import argparse
import json, os
import random
from modeling_baichuan import BaiChuanForCausalLM
from tokenization_baichuan import BaiChuanTokenizer
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import random
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--template_id',default=-1,type=int)
parser.add_argument('--batch_size',default=4,type=int)
parser.add_argument('--data_file',default=None, type=str,help="file that contains instructions (one instruction per line).")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--with_prompt',action='store_true')
parser.add_argument('--interactive',action='store_true')
args = parser.parse_args()


generation_config = dict(
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=5,
    max_new_tokens=100
    )

DEFAULT_PAD_TOKEN = "[PAD]"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.


    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))


    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data


        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)


        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

PROMPT_DICT = [(
        "Instruction: X\n\n"
        "Output: {instruction}\n\n"
        "What kind of instruction could this be the answer to?\n\n"
        "X:"
    ),
    (
        "You are a chatbot. A user sent you an informal message and your reply is as follows.\n\n"
        "Message: X\n\n"
        "Reply: {instruction}\n\n"
        "What is the informal message X?\n\n"
        "X:"
    ),
    (
        "You are a search engine. A person queried something in detail and the most relevant document about the query is as follows.\n\n"
        "Query: X\n\n"
        "Document: {instruction}\n\n"
        "What is the detailed query X?\n\n"
        "X:"
    )]


# PROMPT_DICT = {
#     "prompt_input": (
#         "下面是描述任务的指令，与提供进一步上下文的输入配对。 "
#         "编写适当地完成请求的回复。\n\n"
#         "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回复:"
#     ),
#     "prompt_no_input": (
#         "下面是描述任务的指令。 "
#         "编写适当地完成请求的回复。\n\n"
#         "### 指令:\n{instruction}\n\n### 回复:"
#     ),
# }


sample_data = ["为什么要减少污染，保护环境？"]


def generate_prompt(list_data_dict, template_id=-1):
    if template_id!=-1:
        return [PROMPT_DICT[template_id].format_map(example) for example in list_data_dict]
    else:
        return [random.choice(PROMPT_DICT).format_map(example) for example in list_data_dict]




if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = BaiChuanTokenizer.from_pretrained(args.tokenizer_path,padding_side='left')
    model = BaiChuanForCausalLM.from_pretrained(
        args.base_model, 
        )

        # load_in_8bit=True,
        # torch_dtype=load_type,
        # low_cpu_mem_usage=True,

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )        



    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    # tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    # if model_vocab_size!=tokenzier_vocab_size:
    #     assert tokenzier_vocab_size > model_vocab_size
    #     print("Resize model embeddings to fit tokenizer")
    #     base_model.resize_token_embeddings(tokenzier_vocab_size)





    if device==torch.device('cpu'):
        model.float()
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file,'r') as f:
            examples = json.load(f)
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
        examples[:5]

    # model.to(device)
    model.eval()



    with torch.no_grad():
        if args.interactive:
            while True:
                raw_input_text = input("Input:")
                if len(raw_input_text.strip())==0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(instruction=raw_input_text,template_id=args.template_id)
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids = inputs["input_ids"].to(device), 
                    attention_mask = inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    **generation_config
                )
                s = generation_output[0]
                output = tokenizer.decode(s,skip_special_tokens=True)
                if args.with_prompt:
                    response = output.split("### Response:")[1].strip()
                else:
                    response = output
                print("Response: ",response)
                print("\n")
        else:
            results = []
            # for index, example in enumerate(examples):


            step = len(examples)//args.batch_size
            if step*args.batch_size < len(examples):
                step += 1
            for i in trange(step):
                
                try:
                    example = examples[i*args.batch_size:(i+1)*args.batch_size]
                    
                    if args.with_prompt is True:
                        input_text = generate_prompt(list_data_dict=example,template_id=args.template_id)
                    else:
                        input_text = example


                    inputs = tokenizer(input_text,return_tensors="pt", padding=True)  #add_special_tokens=False ?


                    generation_output = model.generate(
                        input_ids = inputs["input_ids"].to(device), 
                        attention_mask = inputs['attention_mask'].to(device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        **generation_config
                    )
        


                    batch_output = tokenizer.batch_decode(generation_output,skip_special_tokens=True)




                    for j in range(len(example)):



                        candidate_instruction = []
                        output = batch_output[j*5:(j+1)*5]
                        for item in output:
                            candidate_instruction.append(item.split("\n\nX:")[1].strip())


                        results.append({"input":example[j]['instruction'],"output":candidate_instruction,"reference":example[j]['output']})



                except:
                    try:
                        for example in examples[i*args.batch_size:(i+1)*args.batch_size]:
                            example = [example]
                        
                            if args.with_prompt is True:
                                input_text = generate_prompt(list_data_dict=example,template_id=args.template_id)
                            else:
                                input_text = example


                            inputs = tokenizer(input_text,return_tensors="pt", padding=True)  #add_special_tokens=False ?


                            generation_output = model.generate(
                                input_ids = inputs["input_ids"].to(device), 
                                attention_mask = inputs['attention_mask'].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                **generation_config
                            )
                


                            batch_output = tokenizer.batch_decode(generation_output,skip_special_tokens=True)




                            for j in range(len(example)):



                                candidate_instruction = []
                                output = batch_output[j*5:(j+1)*5]
                                for item in output:
                                    candidate_instruction.append(item.split("\n\nX:")[1].strip())


                                results.append({"input":example[j]['instruction'],"output":candidate_instruction,"reference":example[j]['output']})
                    except:
                        continue
                        




            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname,exist_ok=True)
            with open(args.predictions_file,'w') as f:
                json.dump(results,f)
            with open(dirname+'/generation_config.json','w') as f:
                json.dump(generation_config,f)