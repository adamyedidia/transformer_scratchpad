from example_sentences import EXAMPLE_SENTENCES

IN_COLAB = False
M1_MAC = True

import random
import pickle
import einops
from fancy_einsum import einsum
from transformer_lens import HookedTransformer
from transformer_lens import loading_from_pretrained as loading
from transformer_lens.utils import gelu_new
from dataclasses import dataclass
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
import math
import IPython
import tiktoken
import matplotlib.pyplot as plt
import tqdm.auto as tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

LIST_OF_ALL_TOKENS = []
enc = tiktoken.get_encoding('r50k_base')
i = 0
for i in range(50257):
    LIST_OF_ALL_TOKENS.append(enc.decode([i]))


def alphabetic_rank(word):
    base = ord('a')
    rank = 0
    current_letter_value = 1
    for char in word.lower():
        if char.isalpha():
            rank += current_letter_value * (ord(char) - base + 1)
        else:
            raise ValueError(f"Non-alphabetic character '{char}' found in word")

        current_letter_value *= 1 / 26
    return rank


# import IPython
# IPython.embed()

def visualize_tensor(tensor, title, dim=0, cmap="viridis", max_cols=5, scaling_factor=4):
    tensor = tensor.detach().cpu().numpy()
    num_slices = tensor.shape[dim]
    num_rows = int(np.ceil(num_slices / max_cols))
    num_cols = min(num_slices, max_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(scaling_factor * max_cols, scaling_factor * num_rows),
                             constrained_layout=True)

    for i in range(num_slices):
        row, col = divmod(i, max_cols)

        if num_rows > 1:
            ax = axes[row, col]
        else:
            if num_cols == 1:
                ax = axes
            else:
                ax = axes[col]

        ax.imshow(np.take(tensor, i, axis=dim), cmap=cmap)
        ax.set_title(f"{title} (slice {i})")
        ax.axis("off")

    for i in range(num_slices, num_rows * max_cols):
        row, col = divmod(i, max_cols)
        if num_rows > 1:
            ax = axes[row, col]
        else:
            if num_cols == 1:
                ax = axes
            else:
                ax = axes[col]
        ax.axis("off")

    # plt.show()


# model_name = "gpt2-large"
model_name = "gpt2-small"
# model_name = "pythia-70m-v0"

reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                                   center_writing_weights=False)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)


def cuda(x):
    return x.to('cpu') if M1_MAC else x.cuda()


tokens = cuda(tokens)
logits, cache = reference_gpt2.run_with_cache(tokens)

log_probs = logits.log_softmax(dim=-1)
probs = logits.log_softmax(dim=-1)

next_token = logits[0, -1].argmax(dim=-1)

next_tokens = torch.cat(
    [tokens, torch.tensor(next_token, device='cpu' if M1_MAC else 'cuda', dtype=torch.int64)[None, None]], dim=-1)
new_logits = reference_gpt2(next_tokens)

for activation_name, activation in cache.cache_dict.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(activation_name, activation.shape)

for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(name, param.shape)


# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
# print(reference_gpt2.cfg)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# Returns the configuration parameters of the model as a basic Config dataclass
def get_basic_config(model_name: str, **kwargs) -> Config:
    return Config(
        **{k: v for k, v in loading.get_pretrained_model_config(model_name,
                                                                **kwargs).to_dict().items() if k in [
               'd_model',
               'layer_norm_eps',
               'd_vocab',
               'init_range',
               'n_ctx',
               'd_head',
               'd_mlp',
               'n_heads',
               'n_layers',
           ]})


cfg = get_basic_config(model_name)

print(loading.get_pretrained_model_config(model_name))


def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    random_input = cuda(torch.randn(shape))
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    random_input = cuda(torch.randint(100, 1000, shape))
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    print()
    return output


def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict=cache.cache_dict):
    cfg = Config(debug=True)
    layer = cuda(cls(cfg))
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    # Allow inputs of strings or tensors
    if isinstance(input_name, str):
        reference_input = cache_dict[input_name]
    else:
        reference_input = input_name
    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape:", reference_output.shape)

    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct")
    return output


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1",
                               "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized


# _ = rand_float_test(LayerNorm, [2, 4, 768])
# _ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post")

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :]  # [batch, position, d_model]
        # visualize_tensor(self.W_E, 'WE')
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed


# rand_int_test(Embed, [2, 4])
# load_gpt2_test(Embed, reference_gpt2.embed, tokens)

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :]  # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed


# rand_int_test(PosEmbed, [2, 4])
# load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cpu" if M1_MAC else "cuda"))

    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head",
                   normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
                   normalized_resid_pre, self.W_K) + self.b_K

        attn_scores = einsum(
            "batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1)  # [batch, n_head, query_pos, key_pos]

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
                   normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head",
                   pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z,
                          self.W_O) + self.b_O
        return attn_out

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device),
                          diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# rand_float_test(Attention, [2, 4, 768])
# load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"])

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid,
                     self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out


# rand_float_test(MLP, [2, 4, 768])
# load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)
        resid_mid = resid_pre + attn_out

        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post


# rand_float_test(TransformerBlock, [2, 4, 768])
# load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final,
                        self.W_U) + self.b_U
        return logits


SaveTokensAtPeriodInfo = namedtuple('SaveTokensAtPeriodInfo',['filename', 'end_indexes', 'middle_indexes'])

# rand_float_test(Unembed, [2, 4, 768])
# load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens, display=False, save_with_prefix=None, load=False, load_with_mod_vector=None,
                intervene_in_resid_at_layer=None, resid_intervention_filename=None, save_tokens_at_index=None):
        # tokens [batch, position]

        if load:
            residual = pickle.load(open('resid.p', 'rb'))

            plt.plot(residual.detach().numpy().flatten())
            # plt.show()

            if load_with_mod_vector is not None:
                residual = residual + load_with_mod_vector
        else:
            embed = self.embed(tokens)
            # visualize_tensor(self.embed.W_E, 'we')
            # print(embed.shape)
            # visualize_tensor(embed, "Embedding")
            pos_embed = self.pos_embed(tokens)
            # print(pos_embed.shape)
            # visualize_tensor(pos_embed, "Positional Embedding")
            residual = embed + pos_embed
            if intervene_in_resid_at_layer == 'start' and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                residual = (residual + torch.from_numpy(residual_intervention)).float()
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_start.p', 'wb'))
                pickle.dump(embed, open(f'resid_{save_with_prefix}_embed.p', 'wb'))
                pickle.dump(pos_embed, open(f'resid_{save_with_prefix}_pos_embed.p', 'wb'))







        # print(residual.shape)
        for i, block in enumerate(self.blocks):
            residual = block(residual)
            if i == intervene_in_resid_at_layer and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                print('intervening!')
                residual = (residual + torch.from_numpy(residual_intervention)).float()
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_{i}.p', 'wb'))
            if save_tokens_at_index and i > 8:
                filename = save_tokens_at_index.filename
                end_indexes = save_tokens_at_index.end_indexes
                end_tokens_list = pickle.load(open(f'periods_end_{filename}_{i}.p', 'rb'))
                for index in end_indexes:
                    end_tokens_list.append(residual[0][index])
                pickle.dump(end_tokens_list, open(f'periods_end_{filename}_{i}.p', 'wb'))

                middle_indexes = save_tokens_at_index.middle_indexes
                middle_tokens_list = pickle.load(open(f'periods_middle_{filename}_{i}.p', 'rb'))
                for index in middle_indexes:
                    middle_tokens_list.append(residual[0][index])
                pickle.dump(middle_tokens_list, open(f'periods_middle_{filename}_{i}.p', 'wb'))
            # print(residual)

        normalized_resid_final = self.ln_final(residual)

        # visualize_tensor(residual)

        # pickle.dump(residual, open('resid.p', 'wb'))

        if display:
            visualize_tensor(residual, "Residual")
        #
        # print(normalized_resid_final)
        logits = self.unembed(normalized_resid_final)
        # print(logits)
        # logits have shape [batch, position, logits]
        return logits


# rand_int_test(DemoTransformer, [2, 4])
# load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

demo_gpt2 = DemoTransformer(get_basic_config(model_name=model_name))
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
print(cuda(demo_gpt2))


def print_top_n_last_token_from_logits(my_logits, n, compare_on_these_token_indices):
    # Get the logits for the last predicted token
    last_logits = my_logits[-1, -1]
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()

    # Get the indices of the top n probabilities
    topk_indices = np.argpartition(probabilities, -n)[-n:]
    # Get the top n probabilities
    topk_probabilities = probabilities[topk_indices]
    # Get the top n tokens
    topk_tokens = [reference_gpt2.tokenizer.decode(i) for i in topk_indices]

    prob_token_list = list(zip(topk_probabilities, topk_tokens))
    prob_token_list.sort()
    # Print the top n tokens and their probabilities
    for probability, token in prob_token_list:
        print(f"Token: {token}, Probability: {probability}")
    if compare_on_these_token_indices:
        return [probabilities[index] for index in compare_on_these_token_indices]
    else:
        return None


def test_if_token_in_top_n_tokens(my_logits, goal_token, n):
    # Get the logits for the last predicted token
    last_logits = my_logits[-1, -1]
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()

    # Get the indices of the top n probabilities
    topk_indices = np.argpartition(probabilities, -n)[-n:]
    # Get the top n probabilities
    topk_probabilities = probabilities[topk_indices]
    # Get the top n tokens
    topk_tokens = [reference_gpt2.tokenizer.decode(i) for i in topk_indices]

    prob_token_list = list(zip(topk_probabilities, topk_tokens))
    prob_token_list.sort()
    # Print the top n tokens and their probabilities
    for probability, token in prob_token_list:
        print(f"Token: {token}, Probability: {probability}")

    if goal_token in topk_tokens:
        return True
    return False


def run_gpt2_small_on_string(input_string, prefix):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    # is enc.encode('?.!') [30, 13, 0]
    end_of_sentence_tokens = [30, 13, 0]
    end_of_sentence_indicies = [ind for ind, ele in enumerate(test_tokens_in[0]) if ele.item() in end_of_sentence_tokens]
    demo_logits_def = demo_gpt2(test_tokens_in, save_with_prefix=prefix)
    return end_of_sentence_indicies


def check_if_gpt2_gets_right_answer(input_string, correct_next_token):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    demo_logits_def = demo_gpt2(test_tokens_in)
    return test_if_token_in_top_n_tokens(demo_logits_def, correct_next_token, 1)


def get_repeating_token_pattern(cycle_length, total_length):
    random_token_cycle_as_array = [random.choice(LIST_OF_ALL_TOKENS) for _ in range(cycle_length)]
    random_cycle = ''.join(random_token_cycle_as_array)
    num_copies = math.floor(total_length / cycle_length)
    complete_cycles = ''.join([random_cycle for _ in range(num_copies)])
    leftovers = total_length - num_copies*cycle_length
    incomplete_cycle = ''
    next_element = random_token_cycle_as_array[0]
    if leftovers > 0:
        last_elements = random_token_cycle_as_array[:leftovers]
        incomplete_cycle = ''.join(last_elements)
        next_element = random_token_cycle_as_array[leftovers]
    return {
        'next_element': next_element,
        'cycle_array': random_token_cycle_as_array,
        'string': ''.join([complete_cycles, incomplete_cycle])
    }


def get_repeating_token_pattern_and_run_gpt2_small(cycle_length, total_length):
    helper_dict = get_repeating_token_pattern(cycle_length, total_length)
    goal_token = helper_dict['next_element']
    print(f'Goal Token:'+goal_token)
    success = check_if_gpt2_gets_right_answer(helper_dict['string'], goal_token)
    print(success)


def get_angle_between_vectors(a, b):
    dot_product = np.dot(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:
        return None
    return np.arccos(np.clip(dot_product/norms, -1.0, 1.0))


def get_angle_between_one_d_tensors(a, b):
    a_vec = a.detach().numpy()
    b_vec = b.detach().numpy()
    return get_angle_between_vectors(a_vec, b_vec)


def get_n_random_sentences(n):
    sentences = pickle.load(open('sentences.p', 'rb'))['sentences']
    random_sentence_list = [random.choice(sentences)['sentence'] for _ in range(n)]
    string = " ".join(random_sentence_list)
    return string


def run_gpt_on_n_random_sentences(n, prefix):
    string = get_n_random_sentences(n)
    indicies = run_gpt2_small_on_string(string, prefix)
    return indicies, string


def run_gpt_on_string(string, prefix):
    indicies = run_gpt2_small_on_string(string, prefix)
    return indicies, string


def get_residual_streams_with_prefix(prefix):
    resids = {
        i: pickle.load(open(f'resid_{prefix}_{i}.p', 'rb'))
        for i in range(12)
    }
    resids['start'] = pickle.load(open(f'resid_{prefix}_start.p', 'rb'))
    return resids


def get_angles_from_a_given_index(residual_stream, index):
    print(residual_stream.shape)
    angle_index = [
        (get_angle_between_one_d_tensors(residual_stream[0][index], residual_stream[0][i]),i )
        for i in range(len(residual_stream[0]))
    ]
    angle_index.sort()
    return angle_index


def given_index_get_sorted_angles(residual_streams, index):
    return [
        get_angles_from_a_given_index(residual_stream, index)
        for residual_stream in residual_streams.values()
    ]


def given_index_what_bucket(bucket_ends, index):
    for i in range(len(bucket_ends)):
        if index <= bucket_ends[i]:
            return i
    return len(bucket_ends)


def given_string_end_locations_get_average_vec(residual_streams, sentence_ends):
    number_of_tokens = sentence_ends[-1]+1
    angle_dict = {}
    num_sentences = len(sentence_ends)
    sums = np.zeros((num_sentences, num_sentences))
    counts = np.zeros((num_sentences, num_sentences))
    for key, resid_stream in residual_streams.items():
        angles = np.empty((number_of_tokens, number_of_tokens))
        for i in range(number_of_tokens):
            for j in range(number_of_tokens):
                angles[i][j] = get_angle_between_one_d_tensors(resid_stream[0][i], resid_stream[0][j])

                if i != j:
                    bucket_i = given_index_what_bucket(sentence_ends, i)
                    bucket_j = given_index_what_bucket(sentence_ends, j)
                    sums[bucket_i][bucket_j] += angles[i][j]
                    counts[bucket_i][bucket_j] += 1
        angle_dict[key] = angles
    return angle_dict, sums, counts


# a bullshit function to do what I want
def do_the_thing(n):
    indicies, string = run_gpt_on_n_random_sentences(n)
    residual_dict = get_residual_streams_with_prefix('andrea_messing_around')
    print([residual_dict[11][0][i] - residual_dict[0][0][i] for i in indicies])


def get_random_period_sentence_and_indexes():
    sentence = random.choice(EXAMPLE_SENTENCES)
    test_tokens_in = cuda(reference_gpt2.to_tokens(sentence))
    middle_indicies = [
        ind for ind, ele in enumerate(test_tokens_in[0])
        if ele.item() == 13 and ind != len(test_tokens_in[0])-1
    ]
    return sentence, test_tokens_in, [len(test_tokens_in[0])-1,], middle_indicies


def compare_internal_vs_external_periods(filename, num_trials):
    for i in range(8, 12):
        pickle.dump([], open(f'periods_end_{filename}_{i}.p', 'wb'))
        pickle.dump([], open(f'periods_middle_{filename}_{i}.p', 'wb'))

    for _ in tqdm.tqdm(range(num_trials)):
        sentence, test_tokens_in, end_indexes, middle_indexes = get_random_period_sentence_and_indexes()
        #'filename', 'end_indexes' , 'middle_indexes')
        info = SaveTokensAtPeriodInfo(filename=filename, end_indexes=end_indexes, middle_indexes=middle_indexes)
        demo_gpt2(test_tokens_in, save_tokens_at_index=info)

    ends = pickle.load(open(f'periods_end_{filename}_{11}.p', 'rb'))
    middles = pickle.load(open(f'periods_middle_{filename}_{11}.p', 'rb'))
    return ends, middles


def learn_you_an_svm(filename, num_trials):
    print(len(EXAMPLE_SENTENCES))
    ends, middles = compare_internal_vs_external_periods(filename, num_trials)
    # Convert lists of tensors into 2D arrays
    set1 = np.vstack([t.detach().numpy() for t in ends])
    set2 = np.vstack([t.detach().numpy() for t in middles])

    # Combine the sets into one array for training data
    X = np.vstack((set1, set2))

    # Create labels for the sets. We'll use 0 for set1 and 1 for set2.
    y = np.array([0] * len(set1) + [1] * len(set2))
    print(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a SVM Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return clf




IPython.embed()
