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

model = HookedTransformer.from_pretrained('gpt2-small')

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)
# %%

class Node:
    def __init__(self, name, layer=-1, head=-1):
        self.name = name
        self.head = head
        self.layer = layer
        self.mark = 0
        self.edges = []
        pass

    def add_edge(self, edge):
        self.edges.append(edge)

class Edge:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class Graph:
    def __init__(self, model):
        self.model = model
        cfg = model.cfg

        self.nodes = []
        self.edges = []

        for layer in range(cfg.n_layers):
            for head in range(cfg.n_heads):
                node = Node(f"block.{layer}.attn.hook_q", layer=layer, head=head)
                self.nodes.append(node)

        end_node = Node("END")
        self.nodes.append(end_node)

        for start_node in self.nodes[:-1]:
            if start_node.layer == cfg.n_layers - 1:
                edge = Edge(start_node, end_node)
                self.edges.append(edge)
            else:
                start_idx = (start_node.layer + 1) * cfg.n_heads
                for end_node in self.nodes[start_idx:]:
                    edge = Edge(start_node, end_node)
                    self.edges.append(edge)

    def reverse_topo_sort(self):
        sort = []
        
        def visit(n):
            if n.mark == 2:
                return
            elif n.mark == 1:
                print("ERROR: edge cycle")
                exit()
            
            n.mark = 1

            for edge in n.edges:
                visit(edge.end)
            
            n.mark = 2
            sort.append(n)
                
            
        for node in self.nodes:
            visit(node)

        return list(reversed(sort))


g = Graph(model)
sort = g.reverse_topo_sort()
# %%
