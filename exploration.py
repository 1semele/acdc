# %%

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils

import matplotlib.pyplot as plt
import re


model = HookedTransformer.from_pretrained('gpt2-small')
cfg = model.cfg
# %%

prompts = ['When John and Mary went to the shops, John gave the bag to', 'When John and Mary went to the shops, Mary gave the bag to', 'When Tom and James went to the park, James gave the ball to', 'When Tom and James went to the park, Tom gave the ball to', 'When Dan and Sid went to the shops, Sid gave an apple to', 'When Dan and Sid went to the shops, Dan gave an apple to', 'After Martin and Amy went to the park, Amy gave a drink to', 'After Martin and Amy went to the park, Martin gave a drink to']
answers = [(' Mary', ' John'), (' John', ' Mary'), (' Tom', ' James'), (' James', ' Tom'), (' Dan', ' Sid'), (' Sid', ' Dan'), (' Martin', ' Amy'), (' Amy', ' Martin')]

prompt_tokens = model.to_tokens(prompts)

answers_tokens = t.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))]).to(model.cfg.device)

corrupted_indices = t.tensor([i+1 if i%2 == 0 else i-1 for i in range(len(prompts))])
corrupted_tokens = prompt_tokens[corrupted_indices]

_, corrupted_cache = model.run_with_cache(corrupted_tokens)


# %%

def compute_logit_diff(logits, answers_tokens):
    correct_logits = logits.gather(1, answers_tokens[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answers_tokens[:, 1].unsqueeze(1))

    return (correct_logits - incorrect_logits).mean()

# %%

def create_empty_mask():
    mask = {}
    for i in range(cfg.n_layers):
        mask[i] = {}
        for j in range(cfg.n_heads):
            mask[i][j] = {}
            mask[i][j]['q'] = True
            mask[i][j]['k'] = True
            mask[i][j]['v'] = True
        mask[i]['mlp'] = True
    return mask

# %%

mask = create_empty_mask()

def q_hook(value, hook : HookPoint):
    global mask
    layer = int(str(hook.name).split('.')[1])
    for i in range(cfg.n_heads):
        if not mask[layer][i]['q']:
            value[:, :, i, :] = corrupted_cache["q", layer, "attn"][..., i, :]

def k_hook(value, hook : HookPoint):
    global mask
    layer = int(str(hook.name).split('.')[1])
    for i in range(cfg.n_heads):
        if not mask[layer][i]['k']:
            value[:, :, i, :] = corrupted_cache["q", layer, "attn"][..., i, :]

def v_hook(value, hook : HookPoint):
    global mask
    layer = int(str(hook.name).split('.')[1])
    for i in range(cfg.n_heads):
        if not mask[layer][i]['v']:
            value[:, :, i, :] = corrupted_cache["v", layer, "attn"][..., i, :]
    
def mlp_hook(value, hook : HookPoint):
    global mask
    layer = int(str(hook.name).split('.')[1])
    if not mask[layer]['mlp']:
        value[:, :, :] = corrupted_cache["mlp_out", layer]


hooks = [
    (lambda name: bool(re.match(r"blocks.*.hook_mlp_out", name)), mlp_hook),
    (lambda name: bool(re.match(r"blocks.*.attn.hook_q", name)), q_hook),
    (lambda name: bool(re.match(r"blocks.*.attn.hook_k", name)), k_hook),
    (lambda name: bool(re.match(r"blocks.*.attn.hook_v", name)), v_hook),
]

# %%

dirty_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=hooks)[..., -1, :]
model.reset_hooks()
clean_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[])[..., -1, :]

print(compute_logit_diff(dirty_logits, answers_tokens))
print(compute_logit_diff(clean_logits, answers_tokens))

# %%

mask = create_empty_mask()

g_logits = model.run_with_hooks(prompt_tokens)[:, -1, :].softmax(-1)

start_h = reversed([(layer, head) for layer in range(cfg.n_layers) for head in range(cfg.n_heads)])

threshold = 0.0575

h_kl = F.kl_div(g_logits, g_logits, reduction='none').sum(dim=-1).mean()

h_logit_diff = compute_logit_diff(g_logits, answers_tokens)

logit_diffs = []

def update_attn_layer(layer, head, item):
    global hooks
    global g_logits
    global h_kl
    global h_logit_diff
    mask[layer][head][item] = False

    h_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=hooks)[:, -1, :].softmax(-1)
    h_new_kl = F.kl_div(g_logits, h_logits, reduction='none').sum(dim=-1).mean()
    logit_diff = compute_logit_diff(h_logits, answers_tokens)

    amt_changed = abs(logit_diff - h_logit_diff)
    if (amt_changed > threshold):
        mask[layer][head][item] = True
    else:
#        h_logit_diff = logit_diff
        
    logit_diffs.append(logit_diff.item())
"""
    amt_changed = abs(h_new_kl - h_kl)
    if (amt_changed > threshold):
        mask[layer][head][item] = True

"""



"""
    if abs(h_logit_diff - logit_diff) < 0.1:
        h_logit_diff = logit_diff
    else:
        mask[layer][head][item] = True
"""


def update_mlp_layer(parent_layer):
    global hooks
    global g_logits
    global h_kl
    mask[parent_layer]['mlp'] = False

    h_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=hooks)[:, -1, :].softmax(-1)

    h_new_kl = F.kl_div(g_logits, h_logits, reduction='none').sum(dim=-1).mean()

    if (h_new_kl - h_kl) < threshold:
        h_kl = h_new_kl
    else:
        mask[parent_layer]['mlp'] = True

    logit_diff = compute_logit_diff(h_logits, answers_tokens)
    logit_diffs.append(logit_diff.item())

for layer, head in start_h:
    if layer == 0:
        break

    print(layer, head)

    parent_layer = layer - 1
    for parent_head in range(cfg.n_heads):
        update_attn_layer(parent_layer, parent_head, 'q')
        update_attn_layer(parent_layer, parent_head, 'k')
        update_attn_layer(parent_layer, parent_head, 'v')
    
    # update_mlp_layer(parent_layer)

plt.figure()
plt.plot(logit_diffs, color='red')
plt.show()

# %%

dirty_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=hooks)[..., -1, :]
model.reset_hooks()
clean_logits = model.run_with_hooks(prompt_tokens, fwd_hooks=[])[..., -1, :]

print(compute_logit_diff(dirty_logits, answers_tokens))
print(compute_logit_diff(clean_logits, answers_tokens))
# %%
