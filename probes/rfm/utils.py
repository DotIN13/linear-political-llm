import pandas as pd

import numpy as np
import torch
import random
# from janus.utils.conversation import get_conv_template
from enum import Enum, auto

random.seed(0)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from datasets import load_dataset
import json
import os
from copy import deepcopy


class LLMType(Enum):
    TEXT = auto()
    GEMMA_TEXT = auto()
    MULTIMODAL = auto()
    MULTIMODAL_DEEPSEEK = auto()

DEEPSEEK_SYSPROMPT  = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )
    
def split_indices(N, frac=0.2, max_val_count=256, random_split=False):
    n_train = N - min(int(frac*N), max_val_count)
    n_train = n_train + n_train%2 # ensure even train samples
    
    if random_split:
        indices = list(range(N))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        train_indices = range(n_train)
        val_indices = range(n_train, N)
    return train_indices, val_indices
        
        
def load_model(model):
    if model=='llama_3_8b_it':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda"
        )

        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        
    elif model=='gemma_2_9b_it':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
        language_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    elif model=='phi-4':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
        language_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-4", 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    elif model=='toxicchat-t5-large':
        tokenizer = AutoTokenizer.from_pretrained("t5-large")
        language_model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/toxicchat-t5-large-v1.0").to("cuda")
    
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    print(f"Setting pad token id to: {tokenizer.pad_token_id}")
    
    return language_model, tokenizer


def newton_dataset(data_dir, controller):
    random.seed(0)

    template_str = 'Is the following fact about {newton_type} Newton? \nFact: {fact}'
    template_str = controller.format_prompt(template_str)
    
    newton_types = ["Cam", "Isaac"]
    raw_data = {}
    for newton_type in newton_types:
        with open(os.path.join(data_dir, f'{newton_type.lower()}_sentences.txt')) as f:
            lines = f.readlines()
            raw_data[newton_type] = [x.strip('\n') for x in lines]
    
    
    formatted_data = {}
    for newton_type in newton_types:
        n = 300
        c_e, o_e = raw_data[newton_type][:n], np.concatenate([v[:n] for k,v in raw_data.items() if k != newton_type])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        data = np.concatenate(data).tolist()
        newton_train_data = [template_str.format(newton_type=newton_type, fact=d) for d in data]
        
        c_e, o_e = raw_data[newton_type][n:], np.concatenate([v[n:] for k,v in raw_data.items() if k != newton_type])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        data = np.concatenate(data).tolist()
        newton_test_data = [template_str.format(newton_type=newton_type, fact=d) for d in data]
        
        print(f"Train data: {len(newton_train_data)}")
        print(f"Test data: {len(newton_test_data)}")

        formatted_data[newton_type] = {
            'train': {'inputs': newton_train_data, 'labels': train_labels},
            'test': {'inputs': newton_test_data, 'labels': [[1,0] for _ in range(len(newton_test_data)//2)]}
        }
    return formatted_data


def pca_politics_dataset(data_dir, concept_types, tokenizer, assistant_tag, seed=0):
    random.seed(0)

    user_str = 'Which political party supports the following position? \nPosition: {fact}'
    # template_str = 'Write a position in the style of {concept_type} that is similar to the following sample. \nSample: {fact} Position: '
    raw_data = {}
    for concept_type in concept_types:
        with open(os.path.join(data_dir, f"wiki_{concept_type.lower().replace(' ', '_')}.txt")) as f:
            raw_data[concept_type] = f.readlines()

    formatted_data = {}
    for concept_type in concept_types:
       
        other_type = [k for k in raw_data.keys() if k != concept_type][0]
        
        print("Concept", concept_type, "Other", other_type)
        
        c_e, o_e = raw_data[concept_type], raw_data[other_type]
        
        print("c_e", len(c_e), "o_e", len(o_e))
        
        random.shuffle(c_e)
        random.shuffle(o_e)
        
        
        c_e = [user_str.format(concept_type=concept_type, fact=d) for d in c_e]
        o_e = [user_str.format(concept_type=other_type, fact=d) for d in o_e]
        
        min_len = min(len(c_e), len(o_e))
        c_e = c_e[:min_len]
        o_e = o_e[:min_len]
        
        data_ = [[c,o] for c, o in zip(c_e,o_e)]
        data = []
        for pair in data_:
            new_pair = []
            for p in pair:
                chat = [
                    {
                        "role": "user", 
                        "content": p
                    },
                ]
                prompt = tokenizer.apply_chat_template(chat, tokenize=False)
                prompt += assistant_tag
                prompt += 'Answer: '
                new_pair.append(prompt)
            data.append(new_pair)
        
        print(f'Example prompt: {prompt}')
        labels = [[1,0] for _ in range(len(data_))]
        
        
        n_train = 100  
        train_data =  np.concatenate(data[:n_train]).tolist()
        print("all", len(data), "train", len(train_data))
        
        assert(len(data)==len(labels))
              
        test_data =  np.concatenate(data[n_train:2*n_train]).tolist()
        
        train_labels = np.concatenate(labels[:n_train]).tolist()
        test_labels = np.concatenate(labels[n_train:2*n_train]).tolist()
        
        assert(len(train_data)==len(train_labels))
        assert(len(test_data)==len(test_labels))

        print("test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data

# def politics_dataset(llm, seed=0):
#     tokenizer = llm.tokenizer
#     data_dir = 'data/politics'
#     concept_types = ['Democratic', 'Republican'] 
#     assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
#     random.seed(0)

#     user_str = 'Which political party supports the following position? \nPosition: {fact}'
    
#     raw_data = {}
#     for concept_type in concept_types:
#         with open(os.path.join(data_dir, f"wiki_{concept_type.lower().replace(' ', '_')}.txt")) as f:
#             raw_data[concept_type] = f.readlines()

#     formatted_data = {}
#     for concept_type in concept_types:
#         n_train = 300
#         other_type = [k for k in raw_data.keys() if k != concept_type][0]
        
#         print("Concept", concept_type, "Other", other_type)
        
#         c_e, o_e = raw_data[concept_type], raw_data[other_type]
        
#         random.shuffle(c_e)
#         random.shuffle(o_e)
        
        
#         c_e = [user_str.format(concept_type=concept_type, fact=d) for d in c_e]
#         o_e = [user_str.format(concept_type=other_type, fact=d) for d in o_e]
        
#         comb_data = c_e + o_e
#         data = []
#         for d in comb_data:
#             chat = [
#                 {
#                     "role": "user", 
#                     "content": d
#                 },
#             ]
#             prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            
#             if concept_type=='Democratic':
#                 prompt += assistant_tag
#                 prompt += 'Answer: '
                
#             data.append(prompt)
            
#         print(f'Example prompt: {prompt}')
        
#         labels = [1.0 for _ in range(len(c_e))] + [0.0 for _ in range(len(o_e))]
        
#         combined = list(zip(data, labels))

#         # Shuffle the combined list
#         random.shuffle(combined)

#         # Unzip the shuffled list back into two separate lists
#         data, labels = zip(*combined)

#         # Convert back to lists (since zip returns tuples)
#         data = list(data)
#         labels = list(labels)
        
#         train_data = data[:n_train]
#         test_data = data[n_train:2*n_train]
        
#         train_labels = labels[:n_train]
#         test_labels = labels[n_train:2*n_train]
        
#         print("train", len(train_data), "test", len(test_data))

#         formatted_data[concept_type] = {
#             'train': {'inputs': train_data, 'labels': train_labels},
#             'test': {'inputs': test_data, 'labels': test_labels}
#         }
#     return formatted_data


def politics_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/politics'
    concept_types = ['liberal', 'conservative'] 
    # assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Take a {concept_type} stance.  Do you agree with the statement below?  Why or why not? \nStatement: {statement}'

    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    repub_data = [user_str.format(concept_type='conservative', statement=s) for s in raw_data]
    dem_data = [user_str.format(concept_type='liberal', statement=s) for s in raw_data]

    for idx, s in enumerate(repub_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        repub_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(dem_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        dem_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()



    print(repub_data[0], dem_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(repub_data + dem_data)

        if concept_type == "liberal":
            labels = [0.] * len(repub_data) + [1.] * len(dem_data)
        elif concept_type == "conservative":
            labels = [1.] * len(repub_data) + [0.] * len(dem_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data


def conspiracy_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/conspiracies'
    concept_types = ['conspiracy'] 
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Take a {concept_type} stance.  Do you agree with the statement below?  Why or why not? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  Do you agree with the statement below?  Why or why not? \nStatement: {statement}'

    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(concept_type='conspiracy theorist', statement=s) for s in raw_data]
    ncsp_data = [user_str.format(concept_type='non-conspiracy theorist', statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()



    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        if concept_type == "conspiracy":
            labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data



def personality_dataset(llm, personality, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_types = [personality] 
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Take on the role of an expert {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(concept_type=personality, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data



def persona_dataset(llm, persona, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_types = [persona] 
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    # user_str = 'You are {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    user_str = 'Take on the role of {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(concept_type=persona, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data


def pca_persona_dataset(llm, persona, seed=0):
    tokenizer = llm.tokenizer
    concept_type = persona
    data_dir = 'data/general_statements'
    random.seed(0)

    # user_str = 'You are {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    user_str = 'Take on the role of {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type=persona, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    for idx, s in enumerate(csp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()

    return formatted_data


def pca_mood_dataset(llm, mood, seed=0):
    tokenizer = llm.tokenizer
    concept_type = mood
    data_dir = 'data/general_statements'
    random.seed(0)

    user_str = 'Take on a {concept_type} mood.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type=mood, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    for idx, s in enumerate(csp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print("Example CSP data: ", csp_data[0], "Example NCSP data: ", ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()

    return formatted_data


def pca_places_dataset(llm, place, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = place
    random.seed(0)

    user_str = 'Personify someone who loves {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data = f.readlines()
    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data_2 = f.readlines()



    csp_data = [user_str.format(concept_type=place, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    for idx, s in enumerate(csp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data


def pca_fears_dataset(llm, fear, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = fear
    random.seed(0)

    user_str = 'Personify someone who is terrified of {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data = f.readlines()
    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data_2 = f.readlines()


    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type=fear, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    print(len(csp_data))
    for idx, s in enumerate(csp_data):

        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data





def pca_personalities_dataset(llm, personality, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = personality
    random.seed(0)

    user_str = 'Take on the role of an expert {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type=personality, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    for idx, s in enumerate(csp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]

        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data


def pca_conspiracy_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = 'conspiracy'
    random.seed(0)

    user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data = f.readlines()
    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data_2 = f.readlines()


    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type='conspiracy theorist', statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    print(len(csp_data))
    for idx, s in enumerate(csp_data):

        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data



def pca_shakespeare_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = 'shakespeare'
    random.seed(0)

    user_str = 'Answer in {concept_type} english.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data = f.readlines()
    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data_2 = f.readlines()


    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type='Shakespearean', statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    print(len(csp_data))
    for idx, s in enumerate(csp_data):

        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data


def pca_poetry_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = 'poetry'
    random.seed(0)

    user_str = 'Format your answer as a {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data = f.readlines()
    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data_2 = f.readlines()


    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type='poem', statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    print(len(csp_data))
    for idx, s in enumerate(csp_data):

        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data



def pca_influencer_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_type = 'influencer'
    random.seed(0)

    user_str = 'Take on the role of a major {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    # default_str = 'Do you agree with the following statement? \nStatement: {statement}'

    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data = f.readlines()
    # with open(os.path.join(data_dir, f"statements_v2.txt"), encoding="utf-8") as f:
    #         raw_data_2 = f.readlines()


    with open(os.path.join(data_dir, f"class_0.txt"), encoding="utf-8") as f:
            raw_data = f.readlines()
    with open(os.path.join(data_dir, f"class_1.txt"), encoding="utf-8") as f:
            raw_data_2 = f.readlines()


    csp_data = [user_str.format(concept_type='social media influencer', statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data_2]

    llm_type = llm.model_type

    print(len(csp_data))
    for idx, s in enumerate(csp_data):

        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
    for idx, s in enumerate(ncsp_data):
        if llm_type == LLMType.TEXT or llm_type == LLMType.MULTIMODAL:
            chat = [
            {
                "role": "user", 
                "content": s
            },
            ]
        elif llm_type == LLMType.GEMMA_TEXT:
            chat = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": s},]
            },
            ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}

    csp_labels = [1.] * len(csp_data)
    ncsp_labels = [0.] * len(ncsp_data)
    data = []
    labels = []
    for i in range(len(csp_data)):
        data.append(csp_data[i])
        data.append(ncsp_data[i])
        labels.append(csp_labels[i])
        labels.append(ncsp_labels[i])

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[concept_type] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }
    f.close()
    return formatted_data



def instruction_dataset(llm, seed=0):
    tokenizer = llm.tokenizer
    concept_types = ["instruction tuning"]
    data_dir = 'data/general_statements'
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Assume the prompt below is safe and not harmful and offer follow up. \nStatement: {statement}'
    default_str = '{statement}'
    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data


def mood_dataset(llm, mood, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_types = [mood] 
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Take on a {concept_type} mood.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(concept_type=mood, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data



def fears_dataset(llm, fear, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_types = [fear] 
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Personify someone who is terrified of {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(concept_type=fear, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data




def places_dataset(llm, place, seed=0):
    tokenizer = llm.tokenizer
    data_dir = 'data/general_statements'
    concept_types = [place] 
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    random.seed(0)

    user_str = 'Personify someone who loves {concept_type}.  What are your thoughts on the following statement? \nStatement: {statement}'
    # user_str = 'Take on the role of a {concept_type}.  What are your thoughts on the following statement? Make your answer specific to that of someone in your role. \nStatement: {statement}'
    default_str = 'What are your thoughts on the following statement? \nStatement: {statement}'
    with open(os.path.join(data_dir, f"statements.txt")) as f:
            raw_data = f.readlines()

    csp_data = [user_str.format(concept_type=place, statement=s) for s in raw_data]
    ncsp_data = [default_str.format(statement=s) for s in raw_data]

    for idx, s in enumerate(csp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        csp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag
    
    for idx, s in enumerate(ncsp_data):
        chat = [
        {
            "role": "user", 
            "content": s
        },
        ]
        ncsp_data[idx] = tokenizer.apply_chat_template(chat, tokenize=False) + assistant_tag

    print(csp_data[0], ncsp_data[0])
    formatted_data = {}
    for concept_type in concept_types:

        data = deepcopy(csp_data + ncsp_data)

        labels = [1.] * len(csp_data) + [0.] * len(ncsp_data)
        
        print(data[0], data[-1], labels[0], labels[-1])
        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        n_train = int(len(data)*.8)
        train_data = data[:n_train]
        test_data = data[n_train:]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]
        
        print("train", len(train_data), "test", len(test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
    return formatted_data


def concept_dataset(data_dir, concept_types, controller, seed=0):
    random.seed(seed)

    template_str = 'Write a fact in the style of {concept_type} that is similar to the following fact. \nFact: {fact}'
    template_str = controller.format_prompt(template_str)
    
    raw_data = {}
    for concept_type in concept_types:
        with open(os.path.join(data_dir, f"{concept_type.lower().replace(' ', '_')}.txt")) as f:
            lines = f.readlines()
            raw_data[concept_type] = [x.strip('\n') for x in lines]

    formatted_data = {}
    for concept_type in concept_types:
        n = 150 # correction to make distinct test class
        c_e, o_e = raw_data[concept_type][:n], np.concatenate([v[:n] for k,v in raw_data.items() if k != concept_type])
        random.shuffle(o_e)
        
        
        data = [[c,o] for c,o in zip(c_e, o_e)]
        
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        data = np.concatenate(data).tolist()
        concept_train_data = [template_str.format(concept_type=concept_type, fact=d) for d in data]
        
        
        c_e, o_e = raw_data[concept_type][n:], np.concatenate([v[n:] for k,v in raw_data.items() if k != concept_type])
        random.shuffle(o_e)
        data = [[c,o] for c,o in zip(c_e, o_e)]
        data = np.concatenate(data).tolist()
        concept_test_data = [template_str.format(concept_type=concept_type, fact=d) for d in data]
        
        print("train", len(concept_train_data), "test", len(concept_test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': concept_train_data, 'labels': train_labels},
            'test': {'inputs': concept_test_data, 'labels': [[1,0] for _ in range(len(concept_test_data)//2)]}
        }
    return formatted_data

def language_dataset(llm, orig_lang, other_lang, seed=0):
    tokenizer = llm.tokenizer

    random.seed(0)
    
    import pandas as pd
    from datasets import load_dataset
        
    if other_lang == 'chinese':
        huggingface_dataset = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk", data_dir="")
        huggingface_dataset = huggingface_dataset["train"]

        num_of_rows = huggingface_dataset.num_rows//5 
        huggingface_raw_data = pd.DataFrame(columns=["english","chinese"])
        print(num_of_rows)
        for i in range(num_of_rows):
            english = huggingface_dataset[(i*5)+0]["text"].strip("english: ")
            chinese = huggingface_dataset[(i*5)+2]["text"].strip("mandarin: ")
            df_row = pd.DataFrame([[english,chinese]], columns=["english", "chinese"])
            huggingface_raw_data = pd.concat([huggingface_raw_data, df_row], ignore_index=True)

        subset = huggingface_raw_data.iloc[:5000]
        # Extract the columns into two separate lists
        english_sentences = subset.iloc[:, 0].tolist()
        other_sentences = subset.iloc[:, 1].tolist()

        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['chinese'] = other_sentences
            
    elif other_lang == 'hindi':
        huggingface_dataset = load_dataset("cfilt/iitb-english-hindi")
        huggingface_dataset = huggingface_dataset["train"]

        english_sentences = []
        other_sentences = []

        num_of_rows = 1000
        for i in range(num_of_rows):
            entry = huggingface_dataset[i]['translation']
            english_text = entry['en']
            hindi_text = entry['hi']
            english_sentences.append(english_text)
            other_sentences.append(hindi_text)

        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['hindi'] = other_sentences
        
    elif other_lang == 'german':
        huggingface_dataset = load_dataset("wmt/wmt14", "de-en")
        huggingface_dataset = huggingface_dataset["train"]

        english_sentences = []
        other_sentences = []

        num_of_rows = 1000
        for i in range(num_of_rows):
            entry = huggingface_dataset[i]['translation']
            english_text = entry['en']
            german_text = entry['de']
            english_sentences.append(english_text)
            other_sentences.append(german_text)

        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['german'] = other_sentences
            
    formatted_data = {}
    
    c_e, o_e = raw_data[orig_lang], raw_data[other_lang]
    
    data = []
    user_template = 'Complete the translation of the following statement in {orig_lang} to {new_lang}. \nStatement: {statement}\nTranslation: {partial}'

    n = 200 
    c_e, o_e = raw_data[orig_lang][:n], raw_data[other_lang][:n]

    orig_texts = c_e + c_e
    new_texts = c_e + o_e
    labels = [0 for _ in range(n)] + [1 for _ in range(n)]
    new_langs = [orig_lang for _ in range(n)] + [other_lang for _ in range(n)]

    data = []
    for old, new, new_lang in zip(orig_texts, new_texts, new_langs):
        
        idx = max(len(new)//2, 1)
        partial = new[:idx]
        
        prompt = user_template.format(orig_lang=orig_lang, new_lang=new_lang, statement=old, partial=partial)
        chat = [
                {
                "role": "user", 
                "content": prompt
                }
        ]
        ex = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
        data.append(ex)


    combined = list(zip(data, labels))
    # Shuffle the combined list
    random.shuffle(combined)
    # Unzip the shuffled list back into two separate lists
    data, labels = zip(*combined)

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[other_lang] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }

    return formatted_data


def supervised_language_dataset(data_dir, concept_types, tokenizer, seed=0):
    random.seed(seed)

    user_template = 'Complete the translation of the following statement in {orig_lang} to {new_lang}. \nStatement: {statement} Translation: {partial}'
    
    if 'spanish' in concept_types:
        # Load the CSV file (replace 'your_file.csv' with the actual filename)
        df = pd.read_csv(os.path.join(data_dir, 'english-spanish-dataset.csv'))
       # Slice rows 67000 through 70000 (inclusive) and select the second and third columns
        subset = df.iloc[67000:70001, 1:3]
        
        # Extract the columns into two separate lists
        english_sentences = subset.iloc[:, 0].tolist()
        spanish_sentences = subset.iloc[:, 1].tolist()


        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['spanish'] = spanish_sentences
        
    elif 'chinese' in concept_types:
        huggingface_dataset = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk",data_dir="")
        huggingface_dataset = huggingface_dataset["train"]

        num_of_rows = huggingface_dataset.num_rows//5 
        huggingface_raw_data = pd.DataFrame(columns=["english","chinese"])
        print(num_of_rows)
        for i in range(num_of_rows):
            english = huggingface_dataset[(i*5)+0]["text"].strip("english: ")
            chinese = huggingface_dataset[(i*5)+2]["text"].strip("mandarin: ")
            df_row = pd.DataFrame([[english,chinese]], columns=["english", "chinese"])
            huggingface_raw_data = pd.concat([huggingface_raw_data, df_row], ignore_index=True)

        subset = huggingface_raw_data.iloc[:5000]
        # Extract the columns into two separate lists
        english_sentences = subset.iloc[:, 0].tolist()
        other_sentences = subset.iloc[:, 1].tolist()


        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['chinese'] = other_sentences
        
        
    elif 'shakespeare' in concept_types:
        with open(f'{data_dir}/train.modern.nltktok', 'r') as f:
            english_sentences = f.readlines()
            
        with open(f'{data_dir}/train.original.nltktok', 'r') as f:
            other_sentences = f.readlines()


        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['shakespeare'] = other_sentences

    elif 'german' in concept_types:
        huggingface_dataset = load_dataset("wmt/wmt14", "de-en")
        huggingface_dataset = huggingface_dataset["train"]

        english_sentences = []
        other_sentences = []

        num_of_rows = 1000
        for i in range(num_of_rows):
            entry = huggingface_dataset[i]['translation']
            english_text = entry['en']
            german_text = entry['de']
            english_sentences.append(english_text)
            other_sentences.append(german_text)

        raw_data = {}
        raw_data['english'] = english_sentences
        raw_data['german'] = other_sentences
    
    formatted_data = {}
    for concept_type in concept_types:
        
        orig_lang = concept_type
        other_lang = [k for k in raw_data.keys() if k != orig_lang][0]
        
        n = 200 # correction to make distinct test class
        c_e, o_e = raw_data[orig_lang][:n], raw_data[other_lang][:n]
        
        
        orig_texts = c_e + c_e
        new_texts = c_e + o_e
        labels = [0 for _ in range(n)] + [1 for _ in range(n)]
        new_langs = [orig_lang for _ in range(n)] + [other_lang for _ in range(n)]

        data = []
        for old, new, new_lang in zip(orig_texts, new_texts, new_langs):
            statement = old
            
            idx = max(len(new)//2, 1)
            partial = new[:idx]
            
            user_str = user_template.format(orig_lang=orig_lang, new_lang=new_lang, statement=statement, partial=partial)
                
            chat = [
                    {
                        "role": "user", 
                        "content": user_str
                    },
            ]
            data.append(tokenizer.apply_chat_template(chat, tokenize=False))

        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
    
        n_train = 200
        n_test = 200
        concept_train_data = data[:n_train]
        concept_test_data = data[n_train:n_train+n_test]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:n_train+n_test]
        
        print("train", len(concept_train_data), "test", len(concept_test_data))

        formatted_data[concept_type] = {
            'train': {'inputs': concept_train_data, 'labels': train_labels},
            'test': {'inputs': concept_test_data, 'labels': test_labels}
        }
    return formatted_data


def programming_language_dataset(llm, orig_lang, other_lang):
    random.seed(0)
    tokenizer = llm.tokenizer

    user_template = 'Complete the translation of the following program in {orig_lang} to {new_lang}. \nProgram:\n```{program}```\nTranslation:\n```{partial}'

    from datasets import load_dataset
    huggingface_dataset = load_dataset("greengerong/leetcode")
    huggingface_dataset = huggingface_dataset["train"]

    python_programs = huggingface_dataset['python']
    other_programs = huggingface_dataset[other_lang]
    
    raw_data = {}
    raw_data['python'] = python_programs
    raw_data[other_lang] = other_programs
    
    def extract_code(c):
        items = c.split("```")
        code = items[1]
        return code
    
    formatted_data = {}

    n = 400 
    c_e, o_e = raw_data[orig_lang][:n], raw_data[other_lang][:n]

    orig_texts = c_e + c_e
    new_texts = c_e + o_e
    labels = [0 for _ in range(n)] + [1 for _ in range(n)]
    new_langs = [orig_lang for _ in range(n)] + [other_lang for _ in range(n)]

    data = []
    for old, new, new_lang in zip(orig_texts, new_texts, new_langs):
        
        old = extract_code(old)
        new = extract_code(new)
        
        idx = max(len(new)//2, 1)
        partial = new[:idx]
        
        prompt = user_template.format(orig_lang=orig_lang, new_lang=new_lang, program=old, partial=partial)
        chat = [
                {
                "role": "user", 
                "content": prompt
                }
        ]
        ex = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
        data.append(ex)


    combined = list(zip(data, labels))
    # Shuffle the combined list
    random.shuffle(combined)
    # Unzip the shuffled list back into two separate lists
    data, labels = zip(*combined)

    train_data = data    
    train_labels = labels
    print("train", len(train_data))

    formatted_data[other_lang] = {
        'train': {'inputs': train_data, 'labels': train_labels},
    }

    return formatted_data



    combined = list(zip(data, labels))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the shuffled list back into two separate lists
    data, labels = zip(*combined)

    # Convert back to lists (since zip returns tuples)
    data = list(data)
    labels = list(labels)

    n_train = 300
    n_test = 300
    concept_train_data = data[:n_train]
    concept_test_data = data[n_train:n_train+n_test]
    
    train_labels = labels[:n_train]
    test_labels = labels[n_train:n_train+n_test]
    
    print("train", len(concept_train_data), "test", len(concept_test_data))

    formatted_data[concept_type] = {
        'train': {'inputs': concept_train_data, 'labels': train_labels},
        'test': {'inputs': concept_test_data, 'labels': test_labels}
    }
    return formatted_data


def pca_programming_language_dataset(concept_types, tokenizer):
    random.seed(0)

    user_template = 'Complete the translation of the following program in {orig_lang} to {new_lang}. \nProgram:\n```{program}```\nTranslation:\n```{partial}'
    
    from datasets import load_dataset
    huggingface_dataset = load_dataset("greengerong/leetcode")
    huggingface_dataset = huggingface_dataset["train"]

    other_lang = concept_types[1]
    python_programs = huggingface_dataset['python']
    other_programs = huggingface_dataset[other_lang]
    

    raw_data = {}
    raw_data['python'] = python_programs
    raw_data[other_lang] = other_programs
    
    def extract_code(c):
        items = c.split("```")
        code = items[1]
        return code
    
    formatted_data = {}
    for concept_type in concept_types:
        
        orig_lang = concept_type
        other_lang = [k for k in raw_data.keys() if k != orig_lang][0]
        
        n=500
        c_e, o_e = raw_data[orig_lang][:n], raw_data[other_lang][:n]
        
        data = []
        for old, new in zip(c_e, o_e):
            
            old = extract_code(old)
            new = extract_code(new)
            
            pair = []
            idx = max(len(new)//2, 1)

            
            partial = new[:idx]
            prompt = user_template.format(orig_lang=orig_lang, new_lang=orig_lang, 
                                                program=old, partial=partial)
            chat = [
                    {
                        "role": "user", 
                        "content": prompt
                    },
            ]
            pair.append(tokenizer.apply_chat_template(chat, tokenize=False))
            
            idx = max(len(old)//2, 1)                        
            # same lang
            partial = old[:idx]
            prompt = user_template.format(orig_lang=orig_lang, new_lang=other_lang, 
                                                program=old, partial=partial)
            
            chat = [
                    {
                        "role": "user", 
                        "content": prompt
                    },
            ]
            pair.append(tokenizer.apply_chat_template(chat, tokenize=False))
            
            data.append(pair)
            
            
        n = 150
        train_data = data[:n]
        test_data = data[n:2*n]

        train_labels = []
        for d in train_data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        # train_labels = np.concatenate(train_labels).tolist()
        test_labels = [[1,0] for _ in range(len(test_data))]
        
        train_data = np.concatenate(train_data).tolist()
        test_data = np.concatenate(test_data).tolist()
        
        print("train", len(train_data), "test_data", len(test_data))

        formatted_data[orig_lang] = {
            'train': {'inputs': train_data, 'labels': train_labels},
            'test': {'inputs': test_data, 'labels': test_labels}
        }
        
    return formatted_data




def poetry_dataset(llm, concept_types=['prose','poetry']):
    tokenizer = llm.tokenizer
    random.seed(0)
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'

    template_str = 'Complete the translation of the following statement in {orig_lang} to {new_lang}. \nStatement: {statement}\nTranslation: {partial} {assistant_tag}'
    
    with open('./data/poetry/sentences.txt', 'r') as f:
        prose_sentences = f.readlines()
    with open('./data/poetry/poems.txt', 'r') as f:
        poem_sentences = f.readlines()

    raw_data = {}
    raw_data['prose'] = prose_sentences
    raw_data['poetry'] = poem_sentences

    formatted_data = {}
    for concept_type in concept_types:
        
        orig_lang = concept_type
        other_lang = [k for k in raw_data.keys() if k != orig_lang][0]
        
        n = 200 # correction to make distinct test class
        c_e, o_e = raw_data[orig_lang][:n], raw_data[other_lang][:n]
        
        
        orig_texts = c_e + c_e
        new_texts = c_e + o_e
        
        
        N = len(c_e)
        labels = [0 for _ in range(N)] + [1 for _ in range(N)]
        new_langs = [orig_lang for _ in range(N)] + [other_lang for _ in range(N)]

        data = []
        for old, new, new_lang in zip(orig_texts, new_texts, new_langs):
            statement = old
            
            tokens = tokenizer.tokenize(new)
            
            idx = max(3*len(tokens)//4, 1)
            truncated_tokens = tokens[:idx]
            partial = tokenizer.convert_tokens_to_string(truncated_tokens)
            # print(orig_lang, new_lang, statement, partial)
            data.append(
                            template_str.format(orig_lang=orig_lang, new_lang=new_lang, 
                                                statement=statement, partial=partial, assistant_tag=assistant_tag)
                       )

        combined = list(zip(data, labels))

        # Shuffle the combined list
        random.shuffle(combined)

        # Unzip the shuffled list back into two separate lists
        data, labels = zip(*combined)

        # Convert back to lists (since zip returns tuples)
        data = list(data)
        labels = list(labels)
        
        print(len(data), len(labels))

        n_train = 200
        n_test = 200
        concept_train_data = data[:n_train]
        concept_test_data = data[n_train:n_train+n_test]
        
        train_labels = labels[:n_train]
        test_labels = labels[n_train:n_train+n_test]
        
        print("train", len(concept_train_data), "test", len(concept_test_data))


        formatted_data[concept_type] = {
            'train': {'inputs': concept_train_data, 'labels': train_labels},
            'test': {'inputs': concept_test_data, 'labels': test_labels}
        }
    return formatted_data


def creativity_dataset():
    random.seed(0)
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    
    with open('./data/questions/questions.txt', 'r') as f:
        questions = f.readlines()

    data = []
    prefix = "Be extra creative but still accurate in your response to the following requests."
    for question in questions: 
        data.append(f"{prefix}{question}{assistant_tag}")

    N = len(data)

    prefix = "Be accurate in your response to the following request."
    for question in questions: 
        data.append(f"{prefix} {question} {assistant_tag}")

        
    labels = [0 for _ in range(N)] + [1 for _ in range(N)]

    combined = list(zip(data, labels))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the shuffled list back into two separate lists
    data, labels = zip(*combined)

    # Convert back to lists (since zip returns tuples)
    data = list(data)
    labels = list(labels)
        
    print(len(data), len(labels))

    n_train = N // 2
    n_test = N - N // 2
    concept_train_data = data[:n_train]
    concept_test_data = data[n_train:n_train+n_test]
        
    train_labels = labels[:n_train]
    test_labels = labels[n_train:n_train+n_test]
    
    print("train", len(concept_train_data), "test", len(concept_test_data))


    formatted_data = {
        'train': {'inputs': concept_train_data, 'labels': train_labels},
        'test': {'inputs': concept_test_data, 'labels': test_labels}
    }
    return formatted_data


def biology_expert_dataset():
    random.seed(0)
    assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    
    with open('./data/questions/questions.txt', 'r') as f:
        questions = f.readlines()

    data = []
    prefix = "Be accurate in your response to the following requests.  Answer as if you were a world leading expert in biology."
    for question in questions: 
        data.append(f"{prefix}{question}{assistant_tag}")

    N = len(data)

    prefix = "Be accurate in your response to the following request."
    for question in questions: 
        data.append(f"{prefix} {question} {assistant_tag}")

        
    labels = [0 for _ in range(N)] + [1 for _ in range(N)]

    combined = list(zip(data, labels))

    # Shuffle the combined list
    random.shuffle(combined)

    # Unzip the shuffled list back into two separate lists
    data, labels = zip(*combined)

    # Convert back to lists (since zip returns tuples)
    data = list(data)
    labels = list(labels)
        
    print(len(data), len(labels))

    n_train = N // 2
    n_test = N - N // 2
    concept_train_data = data[:n_train]
    concept_test_data = data[n_train:n_train+n_test]
        
    train_labels = labels[:n_train]
    test_labels = labels[n_train:n_train+n_test]
    
    print("train", len(concept_train_data), "test", len(concept_test_data))


    formatted_data = {
        'train': {'inputs': concept_train_data, 'labels': train_labels},
        'test': {'inputs': concept_test_data, 'labels': test_labels}
    }
    return formatted_data



def reviews_dataset(review_dir, tokenizer, center=False, source='amazon'):
    random.seed(0)

    template = 'Give a rating out of 5 for the following review: {review}.'
    chat = [
              {
                "role": "user", 
                "content": template
              },
    ]
    template_str = tokenizer.apply_chat_template(chat, tokenize=False)
    if source=='yelp':
        reviews = pd.read_csv(f'{review_dir}/yelp_reviews_small.csv')
        texts = reviews['text'].to_list()
        ratings = reviews['stars'].to_list()
        
    
    elif source=='amazon':
        ratings = []
        texts = []
        file = f'{review_dir}/Appliances.jsonl'
        with open(file, 'r') as fp:
            for i, line in enumerate(fp):
                if i >= 2000: 
                    break
                review = json.loads(line.strip())
                ratings.append(review['rating'])
                texts.append(review['text'])
    
    combined = list(zip(ratings, texts))
    random.shuffle(combined)
    
    n_points = 800
    if source=='amazon':
        combined = sorted(combined)[:n_points]
    else:
        combined = combined[:n_points]
        
    random.shuffle(combined)
    ratings, texts = zip(*combined)

    r = torch.tensor(ratings).float()
    mean = r.mean().item()
    print("Ratings: std, mean, median", r.std().item(), mean, r.median().item())
    
    if center:
        ratings = [x-mean for x in ratings]
          
    
    data = []
    for text in texts:
        
        tokens = tokenizer.tokenize(text)
        idx = min(len(tokens), 100)
        truncated_tokens = tokens[:idx]
        review = tokenizer.convert_tokens_to_string(truncated_tokens)
            
        data.append(
                        template_str.format(review=review)
                   )
    
    
    return data, ratings


def honesty_dataset(data_path, tokenizer, assistant_tag, seed: int = 0):
    """
    Adapted from from Representation Engineering (https://github.com/andyzoujm/representation-engineering/blob/main/examples/honesty/utils.py)
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5, 1):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append((f"{template_str.format(type='an honest')}", truncated_statement))
            untruthful_statements.append((f"{template_str.format(type='an untruthful')}", truncated_statement))

    # Create training data
    ntrain = 192
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_prompts = np.concatenate(train_data).tolist()
    train_data = []
    for prompt in train_prompts:
        
        chat = [
              {
                "role": "user", 
                "content": prompt[0]
              },
        ]
        new_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        new_prompt += assistant_tag
        new_prompt += prompt[1]
        # print("new_prompt", new_prompt)
        train_data.append(new_prompt)

    # Create test data
    ntest = ntrain
    test_prompts = np.concatenate(combined_data[ntrain:ntrain+ntest]).tolist()
    test_data = []
    for prompt in test_prompts:
        
        chat = [
              {
                "role": "user", 
                "content": prompt[0]
              },
        ]
        new_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        new_prompt += assistant_tag
        new_prompt += prompt[1]
        # print("new_prompt", new_prompt)
        test_data.append(new_prompt)

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'inputs': train_data, 'labels': train_labels},
        'test': {'inputs': test_data, 'labels': [[1,0] for _ in range(ntest)]}
    }


def deepseek_apply_chat_template(conversations, system_prompt=""):
    conv = get_conv_template("deepseek")
    conv.set_system_message(system_prompt)
    for message in conversations:
        conv.append_message(message["role"], message["content"].strip())
    sft_prompt = conv.get_prompt().strip()
    return sft_prompt


def harmful_dataset(llm):
    """
    Adapted from Representation Engineering: https://github.com/andyzoujm/representation-engineering/tree/main/examples/harmless_harmful
    """
    tokenizer = llm.tokenizer
    dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
    
    # switching test and train sets, because test is larger
    test_dataset, train_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']
    
    train_data, train_labels = train_dataset['sentence'], train_dataset['label']
    test_data, test_labels = test_dataset['sentence'], test_dataset['label']
    
    if len(train_data)%2==1:
        train_data=train_data[:-1]
        train_labels=train_labels[:-1]
        
    print("train_data", len(train_data), "train_labels", len(train_labels))

    train_prompts = np.concatenate(train_data).tolist()
    # test_prompts = np.concatenate(test_data).tolist()
    
    train_data = []
    for prompt in train_prompts:
        chat = [
          {
                "role": "user", 
                "content": prompt
              },
        ]
        new_prompt = tokenizer.apply_chat_template(chat, tokenize=False,  add_generation_prompt=True).strip()
        train_data.append(new_prompt)
    print(train_data[0])
    print(train_data[-1])
    print(len(train_data))
    # train_labels = np.concatenate(train_labels)
    train_labels = np.array(train_labels).reshape(-1, 1)
    train_labels = np.where(train_labels, 1, 0)
    print(train_labels[0])
    print(train_labels[-1])
    print(len(train_labels))

    dataset = {}
    dataset['harmful'] = {
            'train': {'inputs': train_data, 'labels': train_labels},
        }
    return dataset


def read_hallucination_prompts_from_lines(lines):
    import re
    
    dicts = []
    for line in lines:
        line = line[1:-1]
        x = re.findall('".*?"', line)
        
        prompt = {}
        prompt['knowledge'] = x[1].strip('"')
        prompt['question'] = x[3].strip('"')
        prompt['answer'] = x[5].strip('"')
        prompt['hallucination'] = x[7].strip('"')
        dicts.append(prompt)
    return dicts


def hallucination_dataset(data_path, llm, seed: int = 0):

    tokenizer = llm.tokenizer

    # Setting the seed for reproducibility
    random.seed(seed)
    print("Getting hallucination data")

    # Load the data 
    with open(data_path, 'r') as f:
        lines = f.readlines()
        prompts = read_hallucination_prompts_from_lines(lines)

    template_str_hallucinate = "Make up an answer to the question that seems correct."
    template_str_factual = "Complete the answer with the correct information."
        
    harmful_statements = []
    harmless_statements = []

    # Process statements
    for prompt in prompts:
        fact_str = prompt['knowledge']
        q_string = prompt['question']
        
        answer_str = prompt['answer']
        hallucinate_str = prompt['hallucination']
        
        ans_tokens = tokenizer.tokenize(answer_str)
        hal_tokens = tokenizer.tokenize(hallucinate_str)
        
        for idx in range(1, len(hal_tokens) - 1):
            truncated_tokens = hal_tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)
            harmful_statements.append(f"{template_str_hallucinate} [FACT] {fact_str} [QUESTION] {q_string} [ANSWER] " + truncated_statement)
            
        for idx in range(1, len(ans_tokens) - 1):
            truncated_tokens = ans_tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)   
            harmless_statements.append(f"{template_str_factual} [FACT] {fact_str} [QUESTION] {q_string} [ANSWER] " + truncated_statement)

    # Create training data
    ntrain = 512 #1024 originally
    combined_prompts = [[harmful, harmless] for harmful, harmless in zip(harmful_statements, harmless_statements)]
    combined_data = []
    for prompt_pair in combined_prompts:
        pair = []
        for prompt in prompt_pair:
            chat = [
                  {"role": "user", "content": prompt}
            ]
            pair.append(tokenizer.apply_chat_template(chat, tokenize=False))
        combined_data.append(pair)
        
    
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    test_data = combined_data[ntrain:2*ntrain]
    test_data = np.concatenate(test_data).tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'inputs': train_data, 'labels': train_labels},
        'test': {'inputs': test_data, 'labels': [[1,0]] * ntrain}
    }
