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
# %%

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)
model.set_use_split_qkv_input(True)
model.set_use_attn_in(True)
model.set_use_attn_result(True)

# %%

EDGE_QKV_TO_OUTPUT = 1
EDGE_OUTPUT_TO_QKV = 2

class Node:
  def __init__(self, name, layer, head):
    if layer == -1 or head == -1:
      self.full_name = name
    else:
      self.full_name = f"blocks.{layer}.{name}.{head}"
      self.tl_name = f"blocks.{layer}.{name}"
    self.name = name
    self.layer = layer
    self.head = head
    self.parents = []
    self.children = []
    self.mark = 0

class Edge:
  def __init__(self, start, end, type):
    self.type = type
    self.frozen = False
    self.start = start
    self.end = end

class Graph:
  def add_node(self, node):
    self.nodes.append(node)
    self.node_dict[node.full_name] = node
    return node

  def add_edge(self, start, end, type):
    e = Edge(start, end, type)
    self.edges.append(e)
    start.children.append(e)
    end.parents.append(e)
    return e

  def __init__(self, model):
    self.edges = []
    self.nodes = []
    self.model = model
    self.node_dict = {}

    cfg = model.cfg

    end = self.add_node(Node("END", -1, -1))

    for layer in range(cfg.n_layers - 1, -1, -1):
      for head in range(cfg.n_heads):

        q = self.add_node(Node("hook_q_input", layer, head))
        k = self.add_node(Node("hook_k_input", layer, head))
        v = self.add_node(Node("hook_v_input", layer, head))

        output = self.add_node(Node("attn.hook_result", layer, head))

        if layer == cfg.n_layers - 1:
          self.add_edge(output, end, EDGE_OUTPUT_TO_QKV)

        for prev_layer in range(layer + 1, cfg.n_layers):
          for prev_head in range(cfg.n_heads):

            next_q = self.node_dict[f"blocks.{prev_layer}.hook_q_input.{prev_head}"]
            next_k = self.node_dict[f"blocks.{prev_layer}.hook_k_input.{prev_head}"]
            next_v = self.node_dict[f"blocks.{prev_layer}.hook_v_input.{prev_head}"]

            self.add_edge(output, next_q, EDGE_OUTPUT_TO_QKV)
            self.add_edge(output, next_k, EDGE_OUTPUT_TO_QKV)
            self.add_edge(output, next_v, EDGE_OUTPUT_TO_QKV)

      for head in range(cfg.n_heads):
        q = self.node_dict[f"blocks.{layer}.hook_q_input.{head}"]
        k = self.node_dict[f"blocks.{layer}.hook_k_input.{head}"]
        v = self.node_dict[f"blocks.{layer}.hook_v_input.{head}"]

        for subhead in range(cfg.n_heads):
          output = self.node_dict[f"blocks.{layer}.attn.hook_result.{subhead}"]
          self.add_edge(q, output, EDGE_QKV_TO_OUTPUT)
          self.add_edge(k, output, EDGE_QKV_TO_OUTPUT)
          self.add_edge(v, output, EDGE_QKV_TO_OUTPUT)
# %%

def topo_sort(g):
  sort = []

  def visit(n):
    if n.mark == 2:
      return
    if n.mark == 1:
      print("ERROR: edge cycle")
      quit()

    n.mark = 1
    for edge in n.parents:
      m = edge.start
      visit(m)

    n.mark = 2
    sort.append(n)

  for n in g.nodes:
    visit(n)

  for n in g.nodes:
    n.mark = 0

  return sort

# %%

g = Graph(model)
nodes = topo_sort(g)

# %%

batch = 10
seq_len = 10

prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()

rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
rand_tokens = t.randint(0, model.cfg.d_vocab, (batch, seq_len * 2), dtype=t.int64)

clean_prompt = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=1).to(device)
dirty_prompt = t.cat([prefix, rand_tokens], dim=1).to(device)
# %%

clean_logits = g.model.run_with_hooks(clean_prompt, fwd_hooks=[])

log_probs = clean_logits.log_softmax(dim=-1)
# Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
log_probs2 = log_probs[:, :-1].gather(dim=-1, index=clean_prompt[:, 1:].unsqueeze(-1)).squeeze(-1)
log_probs = log_probs2
print(f"Performance on the first half: {log_probs[:, :seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[:, seq_len:].mean():.3f}")

dirty_logits = g.model.run_with_hooks(dirty_prompt, fwd_hooks=[])

log_probs = dirty_logits.log_softmax(dim=-1)
# Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
log_probs2 = log_probs[:, :-1].gather(dim=-1, index=dirty_prompt[:, 1:].unsqueeze(-1)).squeeze(-1)
log_probs = log_probs2
print(f"Performance on the first half: {log_probs[:, :seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[:, seq_len:].mean():.3f}")
# %%

def is_tracked_value(name):
  return bool(re.match("blocks.*.hook_q_input", name)) or \
    bool(re.match("blocks.*.hook_k_input", name)) or \
    bool(re.match("blocks.*.hook_v_input", name))

def recieve_hook1(val, hook, g):
  for head in range(cfg.n_heads):
    name = hook.name + '.' + str(head)
    node = g.node_dict[name]
    for edge in node.parents:
      if edge.frozen:
        start_node = edge.start
        val[:, head, :] += g.dirty_cache[start_node.tl_name][:, start_node.head, :]
        val[:, head, :] -= g.clean_cache[start_node.tl_name][:, start_node.head, :]
  return val


def run_graph_patched(g):
  logits, clean_cache = g.model.run_with_cache(clean_prompt)
  logits, dirty_cache = g.model.run_with_cache(dirty_prompt)
  g.clean_cache = clean_cache
  g.dirty_cache = dirty_cache

  hooks_run1 = [
      (is_tracked_value, partial(recieve_hook1, g=g)),
  ]

  logits = g.model.run_with_hooks(clean_prompt, fwd_hooks=hooks_run1)
  return logits[:, -1, :]

# %%

g = Graph(model)

def kl_div(orig_logits, patched_logits):
    p_log_probs = F.log_softmax(orig_logits, dim=-1)
    q_log_probs = F.log_softmax(patched_logits, dim=-1)
    q_probs = t.exp(q_log_probs)

    kl_div = (q_probs * (t.log(q_probs) - p_log_probs)).sum(dim=-1)
    return kl_div.mean()

real_edges = []

def acdc(g, threshold):
  g_logits = run_graph_patched(g)
  h_list = topo_sort(g)
  print(len(h_list))
  for v in tqdm(h_list[:-1]):
    for edge in v.parents:
      if edge.type == EDGE_QKV_TO_OUTPUT:
        continue
      w = edge.start

      h_logits = run_graph_patched(g)

      edge.frozen = True
      h_new_logits = run_graph_patched(g)

      h_kl = kl_div(g_logits, h_logits)
      h_new_kl = kl_div(g_logits, h_new_logits)

      kl_diff = h_new_kl - h_kl
      edge.kl_diff = kl_diff.item()
      real_edges.append(edge)

      print(f"{w.full_name} -> {v.full_name}: {kl_diff.item()}")
      if kl_diff < threshold:
        print("removed")
      else:
        edge.frozen = False
        print("kept")

acdc(g, 0.5623)
# %%

alive_edges = []
for edge in g.edges:
    if edge.frozen and edge.type == EDGE_OUTPUT_TO_QKV:
        alive_edges.append(edge)

real_edges.sort(key=lambda x: x.kl_diff)
print('\n'.join(list(map(lambda x: f"{x.start.full_name} -> {x.end.full_name} : {x.kl_diff}", real_edges))))


print(len(alive_edges))

# %%
