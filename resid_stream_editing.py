# Janky code to do different setup when run in a Colab notebook vs VSCode
DEBUG_MODE = False
IN_COLAB = False
print("Running as a Jupyter notebook - intended for development only!")
# from IPython import get_ipython

# ipython = get_ipython()
# # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

if IN_COLAB or not DEBUG_MODE:
    # Thanks to annoying rendering issues, Plotly graphics will either show up in colab OR Vscode depending on the renderer - this is bad for developing demos! Thus creating a debug mode.
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "png"

# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from jaxtyping import Float, Int
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import pysvelte

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs).show(renderer)

def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

line(np.arange(5))

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# example_prompt = "He didn't understand, so I told"
# example_answer = " him"
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# example_prompt = "He took something that wasn't"
# example_answer = " his"
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# example_prompt = "She took something that wasn't"
# example_answer = " hers"
# tokens = model.to_tokens(example_prompt + example_answer, prepend_bos=True)
# print(tokens)
# logits = utils.remove_batch_dim(model(tokens))
# print(logits.softmax(dim=-1).shape)
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# example_prompt = "She told me something shocking about"
# example_answer = " herself"
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# example_prompt = "She laughed because the jacket he was wearing was way too big for"
# example_answer = " him"
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# example_prompt = "She told him that if he told anyone, she would have to kill"
# example_answer = " him"
# utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

example_prompt = "When Tamarale and Tamastate went to the store, Tamarale gave a drink to"

prompts = [
    "He didn't understand, so I told",
    "She didn't understand, so I told",
    "He took something that wasn't",
    "She took something that wasn't",
    "He told me something shocking about",
    "She told me something shocking about",
]
print([model.to_str_tokens(prompt) for prompt in prompts])

# List of answers, in the format (correct, incorrect)
answers = [
    (" he", " she"),
    (" she", " he"),
    (" him", " her"),
    (" her", " him"),
    (" his", " hers"),
    (" hers", " his"),
    (" himself", " herself"),
    (" herself", " himself"),
]
# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)

answer_tokens = [
    tuple([model.to_single_token(i) for i in answer])
    for answer in answers
]

answer_tokens = torch.tensor(answer_tokens).cuda()
print(prompts)
print(answers)
print(answer_tokens)

print(model.to_single_token(' hers'))

tokens = model.to_tokens(prompts, prepend_bos=True)
print(tokens)
new_tokens = []
# move all the '<|endoftext|>' tokens to the beginning of the token_list
for token_list in tokens:
    new_tokens.append([])
    for token in token_list:
        if token == 50256:
            new_tokens[-1].append(token)
    for token in token_list:
        if token != 50256:
            new_tokens[-1].append(token)

# UNCOMMENT THE BELOW LINE TO IMPLEMENT MY HACKY FIX
tokens = torch.IntTensor(new_tokens)

print(tokens)
    

# Move the tokens to the GPU
tokens = tokens.cuda()
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)

print(original_logits.shape)
# original_logits = model(tokens)

def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()

print("Per prompt logit difference:", logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True))
original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
print("Average logit difference:", logits_to_ave_logit_diff(original_logits, answer_tokens).item())
