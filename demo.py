import os
import sys

import fire
import gradio as gr
import torch
import transformers

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch._dynamo
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
import gc
import traceback
from queue import Queue
from threading import Thread

import torch
import transformers
from torch.nn import CrossEntropyLoss

torch._dynamo.config.suppress_errors = True

IGNORE_INDEX = -100

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True

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

def generate_prompt(example, template_id=-1):
    if template_id!=-1:
        return PROMPT_DICT[template_id].format_map({'instruction':example})
    else:
        return random.choice(PROMPT_DICT).format_map({'instruction':example})

# reverse_dolly_superni_firefly

def main(
    base_model: str = "./reverse_dolly_superni_firefly",
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):

    DEFAULT_PAD_TOKEN = "[PAD]"



    choices_map = [
        "诗歌 - 举头望明月，低头思故乡",
        "数学 - 三角形面积为15平方厘米",
        "地理 - 世界上最小的洲，包含澳大利亚等",
        # ... 其他选项 ...
    ]

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')


    tokenizer = BaiChuanTokenizer.from_pretrained('baichuan-inc/Baichuan-7B',padding_side='left')
    model = BaiChuanForCausalLM.from_pretrained(
        base_model, 
        )

    model_vocab_size = model.get_input_embeddings().weight.size(0)

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )        


    model.eval()


    def rank_candidate(instruction,output):
        
        


        template = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:")

        sources = [template.format_map({'instruction':example}) for example in instruction]

        targets = [f"{output}{tokenizer.eos_token}"] * len(instruction)

        data_dict = preprocess(sources, targets, tokenizer)

        input_ids = data_dict['input_ids']
        labels = data_dict['labels']

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ).to(device)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).to(device)


        outputs = model(input_ids)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        losses = []
        for j in range(shift_logits.size(0)):
            loss_fct = CrossEntropyLoss()
            new_shift_logits = shift_logits[j].view(-1, model_vocab_size)
            new_shift_labels = shift_labels[j].view(-1)

            losses.append(loss_fct(new_shift_logits, new_shift_labels).item())


        pair = list(zip(instruction, losses))
        sorted_pairs = sorted(pair, key=lambda x: x[1])
        rank_list = [pair[0] for pair in sorted_pairs]


        return rank_list

    def evaluate(

        given_text,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):

        template_id = -1
        input_text = generate_prompt(example=given_text,template_id=template_id)
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_beams,
            do_sample=True,
            **kwargs,
        )


        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        try:
            with torch.no_grad():

                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=max_new_tokens,
                )
            
            output = tokenizer.batch_decode(generation_output.sequences,skip_special_tokens=True)
            instruction = [item.split("\n\nX:")[1].strip() for item in output]

            output_lines = rank_candidate(instruction,input_text)
        except:
            return '','输入长度过长，请减少文本长度'

        # 构建HTML输出，第一行为红色
        html_output = "<br>".join(
            [f"Candidate Instruction {i+1}:<br>" + f"<b style='color: red;'>{line}</b>" + "<br>" if i == 0 else f"Candidate Instruction {i+1}:<br>" + line + "<br>" for i, line in enumerate(output_lines)]
        )

        double_quotation = '\"'
        # 构建配对输出
        filtered_instrcution = output_lines[0].replace('\n','\\n')

        paired_output = f"{{{double_quotation}instruction{double_quotation}:{double_quotation}{filtered_instrcution}{double_quotation},{double_quotation}output{double_quotation}:{double_quotation}{given_text}{double_quotation}}}"

        return html_output, paired_output  # 返回HTML和配对输出


    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Selected output to generate corresponding instruction",
                placeholder="Input text",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.3, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.9, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=10, step=1, value=5, label="Return Sequences Number"
            ),
            gr.components.Slider(
                minimum=128, maximum=2000, step=1, value=256, label="Max tokens"
            ),
        ],
    outputs=[
        gr.outputs.HTML(label="输出结果"),  # HTML输出
        gr.outputs.Textbox(label="Paired instruction and output in json format"),  # 配对输出
    ],
        allow_flagging=False, 
        title="OpenING: Open INstruction Generation指令微调数据生成工具",
    ).queue().launch(share=share_gradio)

        # description="",  # noqa: E501
            # gr.inputs.Checkbox(label="提交"),
if __name__ == "__main__":
    fire.Fire(main)