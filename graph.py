# %%

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils
from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt
import re

from tqdm import tqdm

from functools import partial

device = t.device("cuda" if t.cuda.is_available() else "cpu")

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b", 
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)
model.set_use_split_qkv_input(True)
# %%

class Node:
    def __init__(self, name, layer=-1, head=-1):
        self.tl_name = name
        if head != -1:
            self.name = name + '.' + str(head)
        else:
            self.name = name
        self.head = head
        self.layer = layer
        self.mark = 0
        self.children = []
        self.parents = []
        pass

    def add_child(self, edge):
        self.children.append(edge)

    def add_parent(self, edge):
        self.parents.append(edge)

class Edge:
    def __init__(self, start, end):
        self.frozen = False
        self.start = start
        self.end = end

class Graph:
    def __init__(self, model):
        self.model = model
        cfg = model.cfg

        self.nodes = []
        self.node_dict = {}
        self.edges = []

        for layer in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                node = Node(f"blocks.{layer}.attn.hook_q_input", layer=layer, head=head)
                self.nodes.append(node)
                node = Node(f"blocks.{layer}.attn.hook_k_input", layer=layer, head=head)
                self.nodes.append(node)
                node = Node(f"blocks.{layer}.attn.hook_v_input", layer=layer, head=head)
                self.nodes.append(node)

        end_node = Node("END")
        self.nodes.append(end_node)

        for start_node in self.nodes[:-1]:
            if start_node.layer == cfg.n_layers - 1:
                edge = Edge(start_node, end_node)
                self.edges.append(edge)
                start_node.add_child(edge)
                end_node.add_parent(edge)

            else:
                start_idx = (start_node.layer + 1) * cfg.n_heads * 3
                for end_node in self.nodes[start_idx:]:
                    edge = Edge(start_node, end_node)
                    self.edges.append(edge)
                    start_node.add_child(edge)
                    end_node.add_parent(edge)

    def build_node_dict(self):
        for node in self.nodes:
            self.node_dict[node.name] = node

    def reverse_topo_sort(self):
        sort = []
        
        def visit(n):
            if n.mark == 2:
                return
            elif n.mark == 1:
                print("ERROR: edge cycle")
                exit()
            
            n.mark = 1

            for edge in n.children:
                visit(edge.end)
            
            n.mark = 2
            sort.append(n)
            
        for node in self.nodes:
            visit(node)

        return sort

g = Graph(model)
g.build_node_dict()
# %%

batch = 10
seq_len = 10

prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()

rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
rand_tokens = t.randint(0, model.cfg.d_vocab, (batch, seq_len * 2), dtype=t.int64)

clean_prompt = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=1).to(device)
dirty_prompt = t.cat([prefix, rand_tokens], dim=1).to(device)

# %%

def recieve_hook(val, hook, g, clean_cache, dirty_cache):
    for head in range(cfg.n_heads):
        node = g.node_dict[hook.name + '.' + str(head)]
        for prev_edge in node.parents:
            if prev_edge.frozen:
                prev_node = prev_edge.start
#                print(f"frozen {prev_node.name} -> {node.name}")
                val[..., head, :] += clean_cache[prev_node.tl_name][..., head, :]
                val[..., head, :] -= dirty_cache[prev_node.tl_name][..., head, :]

def send_hook(val, hook, g, first):
    pass

def is_attn_component(str):
    return re.match("blocks.*.attn.hook_q", str) or \
           re.match("blocks.*.attn.hook_k", str) or \
           re.match("blocks.*.attn.hook_v", str)

# %%

def generate_hooks(g, clean_cache, dirty_cache):
    first_recieve = partial(recieve_hook, g=g, clean_cache=clean_cache, dirty_cache=dirty_cache)
#    first_send = partial(send_hook, g=g)
#    second_send = partial(send_hook, g=g)
#    first_hooks = [(is_attn_component, first_recieve), (is_attn_component, first_send)]
#    second_hooks = [(is_attn_component, second_recieve), (is_attn_component, second_send)]
    first_hooks = [(is_attn_component, first_recieve)]
    return first_hooks

def run_model(model, g):
    clean_logits, clean_cache = model.run_with_cache(clean_prompt)
    dirty_logits, dirty_cache = model.run_with_cache(dirty_prompt)

    first_hooks = generate_hooks(g, clean_cache, dirty_cache)

    logits = model.run_with_hooks(clean_prompt, fwd_hooks=first_hooks)
    return logits[:, -2, :]

def disable_edge(g, edge):
    edge.frozen = True
    pass

def enable_edge(g, edge):
    edge.frozen = False
    pass
    
def acdc(model, g, threshold):
    g_logits = run_model(model, g).softmax(dim=-1)

    h_list = g.reverse_topo_sort()
    for v in tqdm(h_list):
        for w_edge in tqdm(v.parents, leave=False):
            w = w_edge.start
#            print("TESTING: " + w.name + ' -> ' + v.name)

            disable_edge(g, w_edge)
            h_new_logits = run_model(model, g).softmax(dim=-1)

            enable_edge(g, w_edge)
            h_logits = run_model(model, g).softmax(dim=-1)

            h_kl = F.kl_div(g_logits, h_logits, reduction='none').sum(dim=-1).mean()
            h_new_kl = F.kl_div(g_logits, h_new_logits, reduction='none').sum(dim=-1).mean()

            kl_diff = h_new_kl - h_kl 
            w_edge.kl_diff = kl_diff.item()
#            print(kl_diff)
#            print(w.name + ' -> ' + v.name + ': ' + str(kl_diff.item()))
            if kl_diff < threshold:
                disable_edge(g, w_edge)

g = Graph(model)
g.build_node_dict()
acdc(model, g, 0.0575)

alive_edges = []
for edge in g.edges:
    if not edge.frozen:
        alive_edges.append(edge)


# %%

alive_edges.sort(key=lambda x: x.kl_diff)
for edge in alive_edges[:10]:
    print(f"{edge.start.name} -> {edge.end.name}: {edge.kl_diff}")


# %%
