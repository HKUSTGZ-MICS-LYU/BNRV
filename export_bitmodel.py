import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bitmodel import ModelArgs, Transformer

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def bitnet_export(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack('B', int(shared_classifier)))
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # now let's write out all the params
    weights = [
        # *[layer.attention_norm.weight for layer in model.layers],
        # *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
    ]
    # print(weights)
    for w in weights:
        serialize_fp32(out_file, w)
    
    bit_weights = [
        *[layer.attention.wq for layer in model.layers],
        *[layer.attention.wk for layer in model.layers],
        *[layer.attention.wv for layer in model.layers],
        *[layer.attention.wo for layer in model.layers],
        *[layer.feed_forward.w1 for layer in model.layers],
        *[layer.feed_forward.w2 for layer in model.layers],
        *[layer.feed_forward.w3 for layer in model.layers],
    ]

    if not shared_classifier:
        bit_weights.append(model.output)

    for bitlinear in bit_weights:
        ws, wb = bitlinear.export()
        s = ws.detach().to(torch.float32).cpu().numpy()
        # print("scale:", s)
        out_file.write(struct.pack('f', s))
        out_file.write(wb)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")

def load_checkpoint(checkpoint):

    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

if __name__ == "__main__":
    model_path = "outmini_bit_3M/ckpt.pt"
    bin_path = "outmini_bit_3M/bit_model.bin"
    model = load_checkpoint(model_path)
    bitnet_export(model, bin_path)

