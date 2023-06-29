IN_COLAB = False
M1_MAC = True

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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tiktoken

if M1_MAC:
    import IPython


TITLE_TOKEN_INDEXES = [1583,  1770, 9074, 6997, 6187, 5246, 27034, 10128]

enc = tiktoken.get_encoding('r50k_base')

model_name = "gpt2-small"


def cuda(x):
    return x.to('cpu') if M1_MAC else x.cuda()


reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                                   center_writing_weights=False)


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


class OblationInstruction:
    def __init__(self, layer, head_number):
        self.layer = layer
        self.head_number = head_number


class Attention(nn.Module):
    def __init__(self, cfg, index, ):
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
        self.index = index

    def forward(self, normalized_resid_pre, oblation_instruction=None):
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

        # if we are instructing oblation then oblate at that layer and head number
        if oblation_instruction:
            o_i = oblation_instruction
            layer = o_i.layer
            head_number = o_i.head_number
            if self.index == layer:
                for row in range(pattern[0][head_number].shape[0]):
                    for column in range(pattern[0][head_number].shape[1]):
                        pattern[0][head_number][row][column] = 0.0

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


class TransformerBlock(nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg, i)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        self.index = i

    def forward(self, resid_pre, o_i=None):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre,  oblation_instruction=o_i)
        resid_mid = resid_pre + attn_out

        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post


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

class DemoTransformer(nn.Module):
    def __init__(self, cfg,):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens, save_with_prefix=None, load=False, load_with_mod_vector=None,
                intervene_in_resid_at_layer=None, resid_intervention_filename=None, save_tokens_at_index=None,
                split_tokens_by_lists=None, split_tokens_by_lists_filename=None, reflect_vector_info=None,
                o_i=None, index_lists_with_output_lists=None, store_index_diffs=None,
                replace_layer_i_with_title_intervention=None, pca_intervention_layer_and_index_list=None):
        # tokens [batch, position]

        if load:
            residual = pickle.load(open('resid.p', 'rb'))

            plt.plot(residual.detach().numpy().flatten())
            # plt.show()

            if load_with_mod_vector is not None:
                residual = residual + load_with_mod_vector
        else:
            embed = self.embed(tokens)
            pos_embed = self.pos_embed(tokens)
            residual = embed + pos_embed
            start_residual = embed + pos_embed
            if intervene_in_resid_at_layer == 'start' and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                residual = (residual + torch.from_numpy(residual_intervention)).float()
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_start.p', 'wb'))
                pickle.dump(embed, open(f'resid_{save_with_prefix}_embed.p', 'wb'))
                pickle.dump(pos_embed, open(f'resid_{save_with_prefix}_pos_embed.p', 'wb'))
            if index_lists_with_output_lists:
                for index_list, dict_list_by_layer in index_lists_with_output_lists:
                    for index in index_list:
                        if store_index_diffs:
                            dict_list_by_layer['start'].append(residual[0][index] - start_residual[0][index])
                        else:
                            dict_list_by_layer['start'].append(residual[0][index])

        for i, block in enumerate(self.blocks):
            residual = block(residual, o_i)
            if pca_intervention_layer_and_index_list:
                layer = pca_intervention_layer_and_index_list['layer']
                if i == layer:
                    index_list = pca_intervention_layer_and_index_list['index_list']
                    if len(index_list) > 0:
                        size_of_residual = torch.norm(residual).item()
                        multiplier = pca_intervention_layer_and_index_list['multiplier']
                        # if we want an intervention that is multiplier of the size of residual
                        print(size_of_residual)
                        per_index_multiplier = multiplier * size_of_residual / len(index_list)
                        component_number = pca_intervention_layer_and_index_list['component_number']
                        is_absolute = pca_intervention_layer_and_index_list['is_absolute']
                        pca_vector = torch.from_numpy(
                            (per_index_multiplier * get_pca_vector(
                                layer, component_number, is_absolute=is_absolute
                            ))
                        )
                        for index in index_list:
                            residual[0][index] = (residual[0][index] + pca_vector).float()

                        print(f'Size of residual at start: {size_of_residual}')
                        size_of_intervention = torch.norm(pca_vector) * len(index_list)
                        print(f'Size of intervention: {size_of_intervention}')
                        print(f'Size of residual after intervention: {torch.norm(residual)}')
                        percent = 100 * size_of_intervention / size_of_residual
                        print(f'Percent {percent} %')
            if replace_layer_i_with_title_intervention and i == replace_layer_i_with_title_intervention:
                my_intervention_dict = pickle.load(open('title_intervention_dict.p', 'rb'))
                # find indexes which need to be overwritten. We are looking for a .
                # proceded by a title_token_index
                for title_token_index in TITLE_TOKEN_INDEXES:
                    for j in range(len(tokens[0])-1):
                        if (tokens[0][j].item() == title_token_index) and (tokens[0][j+1] == 13):
                            period_token = j+1
                            residual[0][period_token] = start_residual[0][period_token] + my_intervention_dict[title_token_index][i]

            if reflect_vector_info and i == reflect_vector_info['layer']:
                cls = reflect_vector_info['cls']
                token_index = reflect_vector_info['token_index']
                token_as_np_array = residual[0][token_index].detach().numpy()
                resid_at_start = residual.clone()
                new_resid_for_token = reflect_vector(token_as_np_array, cls)
                for j in range(len(residual[0][token_index])):
                    residual[0][token_index][j] = new_resid_for_token[0][j]

                print(f'Size of residual at start: {torch.norm(resid_at_start)}')
                print(f'Size of intervention: {torch.norm(residual- resid_at_start)}')
                size_of_intervention = torch.norm(residual - resid_at_start)
                print(f'Size of residual after intervention: {torch.norm(residual)}')
                percent = 100* size_of_intervention/torch.norm(resid_at_start)
                print(f'Percent {percent} %')
                array_for_size_percent = reflect_vector_info['array_for_size_percent']
                array_for_size_percent.append((i, percent))
            if i == intervene_in_resid_at_layer and resid_intervention_filename:
                residual_intervention = pickle.load(open(resid_intervention_filename, 'rb'))
                print('intervening!')
                print(sum(sum(sum(residual))))
                residual = (residual + torch.from_numpy(residual_intervention)).float()
                print(sum(sum(sum(residual))))
            if save_with_prefix:
                pickle.dump(residual, open(f'resid_{save_with_prefix}_{i}.p', 'wb'))
            if save_tokens_at_index:
                filename = save_tokens_at_index.filename
                end_indexes = save_tokens_at_index.end_indexes
                end_tokens_list = pickle.load(open(f'ends_{filename}_{i}.p', 'rb'))
                for index in end_indexes:
                    end_tokens_list.append(residual[0][index])
                    # print(f'len end ={len(end_tokens_list)}')
                pickle.dump(end_tokens_list, open(f'ends_{filename}_{i}.p', 'wb'))

                middle_indexes = save_tokens_at_index.middle_indexes
                middle_tokens_list = pickle.load(open(f'middles_{filename}_{i}.p', 'rb'))
                for index in middle_indexes:
                    middle_tokens_list.append(residual[0][index])
                    # print(f'len middle ={len(middle_tokens_list)}')
                pickle.dump(middle_tokens_list, open(f'middles_{filename}_{i}.p', 'wb'))
            # print(residual)
            if split_tokens_by_lists and split_tokens_by_lists_filename:
                my_filename = split_tokens_by_lists_filename
                for key, index_list in split_tokens_by_lists.items():
                    current_token_list = pickle.load(open(f'{my_filename}_{key}_{i}.p', 'rb'))
                    for index in index_list:
                        current_token_list.append(residual[0][index])
                    pickle.dump(current_token_list, open(f'{my_filename}_{key}_{i}.p', 'wb'))
            if index_lists_with_output_lists:

                for index_list, dict_list_by_layer in index_lists_with_output_lists:
                    for index in index_list:
                        if store_index_diffs:
                            dict_list_by_layer[i].append(residual[0][index] - start_residual[0][index])
                        else:
                            dict_list_by_layer[i].append(residual[0][index])

        normalized_resid_final = self.ln_final(residual)

        logits = self.unembed(normalized_resid_final)
        return logits


demo_gpt2 = DemoTransformer(get_basic_config(model_name=model_name))
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)


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
        print(f"Token: \"{token}\", Probability: {probability}")
    if compare_on_these_token_indices:
        return [probabilities[index] for index in compare_on_these_token_indices]
    else:
        return None

def run_gpt2_small_on_string(input_string, prefix, o_i=None, print_logits=False):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    # is enc.encode('?.!') [30, 13, 0]
    end_of_sentence_tokens = [30, 13, 0]
    end_of_sentence_indicies = [ind for ind, ele in enumerate(test_tokens_in[0]) if ele.item() in end_of_sentence_tokens]
    demo_logits_def = demo_gpt2(test_tokens_in, save_with_prefix=prefix, o_i=o_i)
    if print_logits:
        print_top_n_last_token_from_logits(demo_logits_def, 5, None)
    return end_of_sentence_indicies



# Run a test to see how the classifier works
def run_a_test_against_clf_layer_i(string, file_name, layer, cls_file_name, prediction_interpreter=None):
    clf = pickle.load(open(f'cls_{cls_file_name}_{layer}.p', 'rb'))
    run_gpt2_small_on_string(string, file_name)
    x = pickle.load(open(f'resid_{file_name}_{layer}.p', 'rb'))
    token_vecs = [t.detach().numpy() for t in x[0]]
    set_1 = np.vstack(token_vecs)
    predictions = clf.predict(set_1)
    if prediction_interpreter:
        predictions = [prediction_interpreter[prediction] for prediction in predictions]
    tokens_1 = [enc.decode([j]) for j in cuda(reference_gpt2.to_tokens(string))[0]]
    return [(a, b) for a, b in zip(tokens_1, predictions)]


def demo_of_subject_end_of_sentence_other(sentence, layer):
    test_string = "test_andrea"
    file_name = "split_token_by_lists_test_subject"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "Subject",
            1: "Other",
            2:"End of Sentence"
        }
    )


def oblate_and_run_a_test_against_clf_layer_i(string, file_name, i, cls_file_name,
                                              prediction_interpreter=None,
                                              o_i=None,
                                              ):
    clf = pickle.load(open(f'cls_{cls_file_name}_{i}.p', 'rb'))
    run_gpt2_small_on_string(string, file_name,o_i, True)
    x = pickle.load(open(f'resid_{file_name}_{i}.p', 'rb'))
    token_vecs = [t.detach().numpy() for t in x[0]]
    set_1 = np.vstack(token_vecs)
    predictions = clf.predict(set_1)
    if prediction_interpreter:
        predictions = [prediction_interpreter[prediction] for prediction in predictions]
    tokens_1 = [enc.decode([j]) for j in cuda(reference_gpt2.to_tokens(string))[0]]
    return [(a, b) for a, b in zip(tokens_1, predictions)]


def oblate_and_run(sentence):
    test_string = "test_andrea"
    file_name = "middle_vs_not"


    for layer in range(12):
        for head_number in range(12):
            o_i = OblationInstruction(layer, head_number)
            origional = run_a_test_against_clf_layer_i(
                sentence, test_string, 11, file_name,
                prediction_interpreter={
                    0: "End of Sentence",
                    1: "Other",
                }
            )

            new = oblate_and_run_a_test_against_clf_layer_i(
                sentence, test_string, 11, file_name,
                prediction_interpreter={
                    0: "End of Sentence",
                    1: "Other",
                },
                o_i=o_i,
            )

            if origional != new:
                print(f'===== DIFF at Layer={layer} and head_number={head_number}')
                print(list(zip(origional,new)))
                print(f'<<<<<<<<<<<<Differences >>>>>>>>>>>>>>>>>>>')
                for orig_tok, new_tok in zip(origional,new):
                    if orig_tok != new_tok:
                        print(f'orig={orig_tok} new={new_tok}')

                print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


def demo_of_end_of_sentence_vs_other(sentence, layer):
    test_string = "test_andrea"
    file_name = "middle_vs_not"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "End of Sentence",
            1: "Other",
        }
    )


def demo_of_verbing_verbs(sentence, layer):
    test_string = "test_andrea"
    file_name = "andrea_verbing_verbs"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "Subject",
            1: "Verb",
            2: "Other",
            3: "End of Sentence"
        }
    )


def get_undo_layer_i(file_name, i):
    cls = pickle.load(open(f'cls_{file_name}_{i}.p', 'rb'))
    w = cls.coef_[0]
    w_norm = w / np.linalg.norm(w)
    return w_norm


def make_intervention_on_one_token(change_index, length, file_name, layer_number, multiplier):
    w_norm = multiplier * get_undo_layer_i(file_name, layer_number)
    array = np.zeros((1, length, 768))
    print(f"Intervention Shape={array.shape}")
    array[0, change_index, :] = w_norm
    # print(f"Intervention Array={array}")
    pickle.dump(array, open(f'w_norm{file_name}_index_{change_index}_layer_{layer_number}.p', 'wb'))


def get_pca_vector(layer, component_number, is_absolute=False):
    if is_absolute:
        print("Aboslute")
        pca_vector = pickle.load(open(f'pca_files/10_token_absolute_pca_layer_{layer}_component_{component_number}.p', 'rb'))
        return pca_vector/ np.linalg.norm(pca_vector)

    pca_vector = pickle.load(open(f'pca_files/10_token_pca_layer_{layer}_component_{component_number}.p', 'rb'))
    return pca_vector / np.linalg.norm(pca_vector)


def reflect_vector(X, clf):
    X = X.reshape(1, -1)  # reshape the data
    # Calculate the distance of the point to the hyperplane
    distance = np.abs(clf.decision_function(X)) / np.linalg.norm(clf.coef_)

    # Calculate the direction to move the point
    direction = np.sign(clf.decision_function(X))

    # Reflect the point across the hyperplane
    X_reflected = X - 2 * distance * direction * (clf.coef_ / np.linalg.norm(clf.coef_))

    # Predict the classes of X and X_reflected
    y_pred = clf.predict(X)
    y_pred_reflected = clf.predict(X_reflected)

    print(f'Predicted class for X: {y_pred[0]}')
    print(f'Predicted class for X_reflected: {y_pred_reflected[0]}')

    return X_reflected


def run_gpt_demo_with_intervention(input_string, layer, filename, percents=[]):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    period_index = len(test_tokens_in[0])-1

    print(f"++++ period_index={period_index} +++++  len(tokens)={len(test_tokens_in[0])}")

    print(f"======= REFLECTING Layer {layer}=========")
    my_logits = demo_gpt2(
        test_tokens_in,
        reflect_vector_info={
            'layer': layer,
            'cls': pickle.load(open(f'cls_{filename}_{layer}.p', 'rb')),
            'token_index': period_index,
            'array_for_size_percent': percents,
        },
    )
    print_top_n_last_token_from_logits(my_logits, 5, None)

    return my_logits


def total_variation_distance(vec_p, vec_q):
    return 0.5 * np.sum(np.abs(vec_p - vec_q))


def get_probabilities_from_logits(my_logits):
    # Get the logits for the last predicted token
    last_logits = my_logits[-1, -1]
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
    return probabilities


def title_compare_run_gpt2(input_string, layer):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    default_logits_def = demo_gpt2(test_tokens_in,)
    intervention_logits_def = demo_gpt2(test_tokens_in, replace_layer_i_with_title_intervention=layer)
    default_probs = get_probabilities_from_logits(default_logits_def)
    intervention_probs = get_probabilities_from_logits(intervention_logits_def)

    print('========= DEFAULT =============')
    print_top_n_last_token_from_logits(default_logits_def, 5, None)

    print(f'========= INTERVENTION LAYER {layer} =============')
    print_top_n_last_token_from_logits(intervention_logits_def, 5, None)

    print('========== TVD ==============')
    t_v_d = total_variation_distance(default_probs, intervention_probs)
    print(f'TVD = {t_v_d}')

    return t_v_d


def get_indexes(lst, subset):
    return [i for i, x in enumerate(lst) if x.item() in subset]


def run_pca_intervention_on_listed_tokens(layer, component_number, multiplier, input_string, list_of_tokens, is_absolute=False):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    list_of_token_indexes = [reference_gpt2.tokenizer.encode(t)[0] for t in list_of_tokens]
    print(list_of_token_indexes)
    index_list = get_indexes(test_tokens_in[0], list_of_token_indexes)
    print(index_list)
    pca_run_intervention(layer, component_number, multiplier, index_list, test_tokens_in, is_absolute=is_absolute)


def run_pca_intervention_on_last_token(layer, component_number, multiplier, input_string, is_absolute=False):
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    index_list = [len(test_tokens_in[0])-1,]
    pca_run_intervention(layer, component_number, multiplier, index_list, test_tokens_in, is_absolute=is_absolute)


def pca_run_intervention(layer, component_number, multiplier, index_list, tokens, is_absolute=False):
    intervention_dict = {
        'layer': layer,
        'index_list': index_list,
        'multiplier': multiplier,
        'component_number': component_number,
        'is_absolute': is_absolute,

    }
    default_logits = demo_gpt2(tokens,)
    intervention_logits = demo_gpt2(tokens, pca_intervention_layer_and_index_list=intervention_dict)

    print('========= DEFAULT =============')
    print_top_n_last_token_from_logits(default_logits, 5, None)

    print(f'========= PCA INTERVENTION Layer={layer} Component={component_number} Multiplier={multiplier} Token Indexes={index_list}=============')
    print_top_n_last_token_from_logits(intervention_logits, 5, None)


def titles_run_many_sentences_and_plot(list_of_sentences):
    # Store results in a dictionary: {sentence: [tvd1, tvd2, ..., tvd12]}
    results = {}
    tvd_at_layer_4 = []

    # Iterate over each sentence
    for sentence in list_of_sentences:
        print(f'[[[[[[[[[[[[[[[[[[[[[[[ {sentence} ]]]]]]]]]]]]]]]]]]]]]]]')
        tvd_values = []
        # For each layer
        for layer in range(12):
            # Calculate total variation distance
            tvd = title_compare_run_gpt2(sentence, layer)
            tvd_values.append(tvd)
        results[sentence] = tvd_values
        # Store the TVD at layer 4 for each sentence
        tvd_at_layer_4.append((sentence, tvd_values[4]))

    # Set up plot
    plt.figure(figsize=(10, 5))

    # Create a color cycle for lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(list_of_sentences)))

    # Plot each sentence's results
    for i, (sentence, tvd_values) in enumerate(results.items()):
        plt.plot(range(12), tvd_values, label=sentence, color=colors[i])

        # Set labels, title, and legend
    plt.xlabel('Layer')
    plt.ylabel('Total Variation Distance')
    plt.title('Total Variation Distance by Layer for Each Sentence')

    # Customizing x-axis ticks
    plt.xticks(range(12))

    # Adjusting legend position
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Show plot with legend outside of plot area
    plt.tight_layout()
    plt.show()

    # Print the TVD at layer 4 for each sentence
    for sentence, tvd in tvd_at_layer_4:
        print(f'TVD at layer 4 for sentence "{sentence}": {tvd}')

def demo_of_title_vs_end(sentence, layer):
    test_string = "test_andrea"
    file_name = "title_vs_end"
    return run_a_test_against_clf_layer_i(
        sentence, test_string, layer, file_name,
        prediction_interpreter={
            0: "End period",
            1: "Title period",
        }
    )


NEW_SENTENCE_TOKENS = [' The', ' They', '  Mr', ' Dr', ' Mrs', ' His', ' Ms', ' Miss', ' He', ' She']
STUPID_NAMES = [' Adams']


def run_gpt_demo_title_intervention_all_layers_and_graph(input_string):
    filename = "title_vs_end"
    new_sentence_index = [reference_gpt2.tokenizer.encode(i)[0] for i in NEW_SENTENCE_TOKENS]
    name_index = [reference_gpt2.tokenizer.encode(i)[0] for i in STUPID_NAMES]

    new_sentence_probs_all = []
    name_probs_all = []
    percents = []

    print("======= DEFAULT =========")
    test_tokens_in = cuda(reference_gpt2.to_tokens(input_string))
    default_logits = demo_gpt2(test_tokens_in,)
    print_top_n_last_token_from_logits(default_logits, 5, None)
    # Get the logits for the last predicted token
    last_logits = default_logits[-1, -1]
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
    new_sentence_probs = probabilities[new_sentence_index]
    name_probs = probabilities[name_index]

    default_new_sentence_prob = new_sentence_probs.mean()
    default_name_prob = name_probs.mean()

    for layer in range(12):
        my_logits = run_gpt_demo_with_intervention(input_string, layer, filename, percents=percents)
        # Get the logits for the last predicted token
        last_logits = my_logits[-1, -1]
        # Apply softmax to convert the logits to probabilities
        probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
        new_sentence_probs = probabilities[new_sentence_index]
        name_probs = probabilities[name_index]

        new_sentence_probs_all.append(new_sentence_probs.mean())
        name_probs_all.append(name_probs.mean())

    print(f"Percents  = {percents}")
    # Extract layers and percents
    layers = [layer for layer, _ in percents]
    percent_values = [percent.item() for _, percent in percents]  # use .item() to convert tensors to numbers

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(layers, percent_values, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Intervention Size')
    plt.title('Intervention Size % of Residual Size by Layer')
    plt.grid(True)
    plt.show()

    # Plot the average probabilities for each set of tokens at each layer
    plt.figure(figsize=(10, 5))
    plt.plot(range(12), new_sentence_probs_all, label='New Sentence Tokens', color='blue')
    plt.plot(range(12), name_probs_all, label=f'{STUPID_NAMES}', color='red')
    plt.axhline(default_new_sentence_prob, color='blue', linestyle='dotted', label='Default New Sentence Prob.')
    plt.axhline(default_name_prob, color='red', linestyle='dotted', label='Default Name Prob.')
    plt.xlabel('Layer')
    plt.ylabel('Average Probability')
    plt.title(f'Average Probability by Layer for "{input_string}"')
    plt.legend()
    plt.show()


    ## Prob of all things
    # Extract layers and percents
    layers = [layer for layer, _ in percents]
    percent_values = [percent.item() for _, percent in percents]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot the intervention sizes
    ax1.plot(layers, percent_values, 'o-', color='tab:purple', label='Intervention Size', linewidth=2.5,
             linestyle='dashed')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Intervention Size', color='tab:purple')
    ax1.tick_params(axis='y', labelcolor='tab:purple')

    ax2 = ax1.twinx()  # Create a second y-axis

    # Plot the probabilities on the second y-axis
    ax2.plot(layers, new_sentence_probs_all, 's-', label='New Sentence Tokens', color='tab:orange')
    ax2.plot(layers, name_probs_all, 'd-', label=f'{STUPID_NAMES}', color='tab:green')
    ax2.axhline(default_new_sentence_prob, color='tab:orange', linestyle='dotted', label='Default New Sentence Prob.')
    ax2.axhline(default_name_prob, color='tab:green', linestyle='dotted', label='Default Name Prob.')
    ax2.set_ylabel('Average Probability', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Add explanations to the legend
    intervention_line = mlines.Line2D([], [], color='tab:purple', marker='o', linestyle='dashed',
                                      label='Intervention Size')
    left_label = mlines.Line2D([], [], color='white', label='Left: Intervention Size')
    right_label = mlines.Line2D([], [], color='white', label='Right: Average Probability')
    ax2.legend(handles=[intervention_line, ax2.get_legend_handles_labels()[0][0], ax2.get_legend_handles_labels()[0][1],
                        ax2.get_legend_handles_labels()[0][2], ax2.get_legend_handles_labels()[0][3], left_label,
                        right_label])

    fig.subplots_adjust(top=0.9)  # Adjust the top space
    fig.suptitle(f'Intervention Sizes and Probabilities by Layer for "{input_string}"',
                 y=0.98)  # Adjust the position of title
    plt.grid(True)
    plt.show()

if M1_MAC:
    IPython.embed()
