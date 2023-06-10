import torch as t
import pandas as pd
import algebraic_value_editing
import einops
import prettytable

from typing import List, Dict, Union, Callable
from functools import partial
from transformer_lens.HookedTransformer import HookedTransformer
from algebraic_value_editing import hook_utils
from algebraic_value_editing.completion_utils import print_n_comparisons
from algebraic_value_editing.prompt_utils import ActivationAddition, get_x_vector

model_name: str = "gpt2-xl"
# model_name: str = "gpt2-small"
device: str = "cuda" if t.cuda.is_available() else "cpu"
model: HookedTransformer = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)
_ = model.to(device)
_ = t.set_grad_enabled(False)
num_layers: int = model.cfg.n_layers
print(f"{model_name} has {num_layers} layers.")

default_kwargs = {
    "temperature": 1.0,
    "top_p": 0.3,
    "freq_penalty": 1.0,
    "num_comparisons": 10,
    "tokens_to_generate": 50,
    "seed": 0, # For reproducibility
}

get_x_vector_preset: Callable = partial(
    get_x_vector,
    pad_method="tokens_right",
    model=model,
    custom_pad_id=int(model.to_single_token(" ")),
)

summand: List[ActivationAddition] = [
    *get_x_vector_preset(
        prompt1="Hello",
        prompt2="H-E-L-L-O",
        coeff=3,
        act_name=10,
    )
]
HATE_PROMPT = (
    "I hate you because"
)

SPELLING_PROMPT = (
    '"Hello" is spelled H-E-L-L-O. "World" is spelled W-O-R-L-D. Tree is spelled'
    # 'Tree is spelled'
)

print_n_comparisons(model=model,
    prompt=SPELLING_PROMPT,
    activation_additions=summand,
    **default_kwargs,
)