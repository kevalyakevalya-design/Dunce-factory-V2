#!/usr/bin/env python3
"""Complete corrected GPT learning script.

Based on the user's learning script and Sebastian Raschka's
Build a Large Language Model (From Scratch), Apache License 2.0.
Copyright (c) Sebastian Raschka for the adapted book code.
Source: https://github.com/rasbt/LLMs-from-scratch
License: https://www.apache.org/licenses/LICENSE-2.0
Modifications: corrected syntax, imports, downloads, device handling and generation;
added command-line controls and isolated execution under main().

Setup and full run:
    python3 omt.py --install-deps
Shorter training run (small RANDOM model; skips pretrained GPT-2):
    python3 omt.py --install-deps --quick --epochs 1
Full architecture without TensorFlow/pretrained downloads:
    python3 omt.py --install-deps --skip-pretrained --epochs 1

Default: all teaching examples, 10 training epochs, checkpoint save/reload,
then download/load pretrained GPT-2 124M and generate text. Full mode needs
substantial RAM/disk and can take a long time on CPU. Training on one story
is an educational demonstration, not a generally capable language model.
Plots are saved without blocking; use --show-plots for interactive windows.
Outputs default to an omt-output directory beside this script.
Initial story, tokenizer and pretrained downloads require internet access.
Dependencies are installed only when --install-deps is explicitly supplied.
Python 3.10+ and PyTorch 2.6+ are required; MPS auto-selection uses 2.9+.
"""

from pathlib import Path
import argparse
import importlib.util
import re
import subprocess
import sys


def select_device(torch, requested):
    if requested != 'auto':
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was selected but is unavailable')
        if requested == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS was selected but is unavailable')
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    match = re.match(r'(\d+)\.(\d+)', torch.__version__)
    recent = match and tuple(map(int, match.groups())) >= (2, 9)
    if torch.backends.mps.is_available() and recent:
        return torch.device('mps')
    return torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--quick', action='store_true', help='Use a small random model and skip pretrained GPT-2')
    parser.add_argument('--skip-pretrained', action='store_true')
    parser.add_argument('--install-deps', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--text-file', type=Path, help='Use your own UTF-8 training text')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'omt-output')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    return args


def prepare_dependencies(args):
    requirements = {'torch': 'torch>=2.6', 'tiktoken': 'tiktoken', 'requests': 'requests',
                    'numpy': 'numpy', 'matplotlib': 'matplotlib', 'tqdm': 'tqdm'}
    if not (args.quick or args.skip_pretrained):
        requirements['tensorflow'] = 'tensorflow'
    if args.install_deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements.values()], check=True)
    missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError('Missing dependencies: ' + ', '.join(missing) + '. Run again with --install-deps.')
    import torch
    match = re.match(r'(\d+)\.(\d+)', torch.__version__)
    if not match or tuple(map(int, match.groups())) < (2, 6):
        raise RuntimeError('PyTorch 2.6+ is required. Run again with --install-deps.')
    import matplotlib
    if not args.show_plots:
        matplotlib.use('Agg')


def main(args):
    print('at ur service !')
    from pathlib import Path
    import requests
    BASE_DIR = args.output_dir.resolve()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = args.text_file.resolve() if args.text_file else BASE_DIR / 'the-verdict.txt'
    if args.text_file and not file_path.is_file():
        raise FileNotFoundError(f'Text file does not exist: {file_path}')
    if not file_path.exists():
        url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print('total number of charecter:', len(raw_text))
    print(raw_text[:99])
    import re
    text = 'hello, world . i hope the factory treats me nice.'
    result = re.split('(\\s)', text)
    print(result)
    result = re.split('([,.]|\\s)', text)
    print(result)
    text = 'hello, world . i hope the factory treats me nice.'
    result = re.split('([,.!?:;"()]|--|\\s)', text)
    result = [item.strip() for item in result if item.strip()]
    print(result)
    preprocessed = re.split('([,.:;?_!"()\\\']|--|\\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])
    print(len(preprocessed))
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break

    class simpletokenizerV1:

        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split('([,.:;?_!"()\\\']|--|\\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = ' '.join([self.int_to_str[i] for i in ids])
            text = re.sub('\\s+([,.?!"()\\\'])', '\\1', text)
            return text
    tokenizer = simpletokenizerV1(vocab)
    text = 'It\'s the last he painted, you know,"\n           Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(text)
    print(ids)
    tokenizer.decode(ids)
    tokenizer.decode(tokenizer.encode(text))
    tokenizer = simpletokenizerV1(vocab)
    text = 'Hello, do you like tea. Is this-- a test?'
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(['<|endoftext|>', '<|unk|>'])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    len(vocab.items())
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    class simpletokenizerv2:

        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split('([,.:;?_!"()\\\']|--|\\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = ' '.join([self.int_to_str[i] for i in ids])
            text = re.sub('\\s+([,.:;?!"()\\\'])', '\\1', text)
            return text
    tokenizer = simpletokenizerv2(vocab)
    text1 = 'Hello, do you like tea?'
    text2 = 'In the sunlit terraces of the palace.'
    text = ' <|endoftext|> '.join((text1, text2))
    print(text)
    tokenizer.encode(text)
    tokenizer.decode(tokenizer.encode(text))
    import importlib.metadata
    import tiktoken
    print('tiktoken version:', importlib.metadata.version('tiktoken'))
    tokenizer = tiktoken.get_encoding('gpt2')
    text = 'Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.'
    integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    print(integers)
    strings = tokenizer.decode(integers)
    print(strings)
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f'x: {x}')
    print(f'y:    {y}')
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, '-->', desired)
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), '-->', tokenizer.decode([desired]))
    import torch
    from torch.utils.data import Dataset, DataLoader

    class GPTDatasetV1(Dataset):

        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []
            token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
            if max_length < 1 or stride < 1:
                raise ValueError('max_length and stride must be positive')
            if len(token_ids) <= max_length:
                raise ValueError(f'Need more than {max_length} tokens in this data split; got {len(token_ids)}. Use more text or a smaller context length.')
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1:i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return (self.input_ids[idx], self.target_ids[idx])

    def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        tokenizer = tiktoken.get_encoding('gpt2')
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    input, targets = next(data_iter)
    print('inputs\n', input)
    print('\nTargets:\n', targets)
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3
    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))
    print(embedding_layer(input_ids))
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    input, targets = next(data_iter)
    print('token IDs:\n', input)
    print('\nInputs shape\n', input.shape)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    token_embedding_layer = torch.nn.Embedding(tokenizer.n_vocab, output_dim)
    token_embeddings = token_embedding_layer(input)
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    from importlib.metadata import version
    print('torch version:', version('torch'))
    inputs = torch.tensor([[0.43, 0.15, 0.89], [0.55, 0.87, 0.66], [0.57, 0.85, 0.64], [0.22, 0.58, 0.33], [0.77, 0.25, 0.1], [0.05, 0.8, 0.55]])
    query = inputs[1]
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    print(attn_scores_2)
    res = 0
    for idx, element in enumerate(inputs[0]):
        res += inputs[0][idx] * query[idx]
    print(res)
    print(torch.dot(inputs[0], query))
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print('attention weights:', attn_weights_2_tmp)
    print('sum:', attn_weights_2_tmp.sum())

    def softmax_naive(x):
        exp_x = torch.exp(x - x.max(dim=0, keepdim=True).values)
        return exp_x / exp_x.sum(dim=0)
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print('attention weights:', attn_weights_2_naive)
    print('sum:', attn_weights_2_naive.sum())
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print('attentionw eights:', attn_weights_2)
    print('sum:', attn_weights_2.sum())
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print('Attention weights:', attn_weights_2)
    print('Sum:', attn_weights_2.sum())
    query = inputs[1]
    context_vec_2 = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(attn_scores)
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)
    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.124, 0.1082, 0.1581])
    print('Row 2 sum', row_2_sum)
    print('all row sums:', attn_weights.sum(dim=-1))
    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    print('Previous 2nd context vector:', context_vec_2)
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)
    keys = inputs @ W_key
    values = inputs @ W_value
    print('keys.shape:', keys.shape)
    print('values.shape:', values.shape)
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)
    attn_scores_2 = query_2 @ keys.T
    print(attn_scores_2)
    d_k = keys.shape[1]
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
    print(attn_weights_2)
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)
    import torch.nn as nn

    class SelfAttention_v1(nn.Module):

        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value
            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            context_vec = attn_weights @ values
            return context_vec
    torch.manual_seed(123)
    sal_v1 = SelfAttention_v1(d_in, d_out)
    print(sal_v1(inputs))
    queries = inputs @ sal_v1.W_query
    keys = inputs @ sal_v1.W_key
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)
    masked_simple = attn_weights * mask_simple
    print(masked_simple)
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill_(mask.bool(), -torch.inf)
    print(masked)
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))
    torch.manual_seed(123)
    print(dropout(attn_weights))
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)

    class CasualAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            attn_scores = queries @ keys.transpose(1, 2)
            attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = attn_weights @ values
            return context_vec
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CasualAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print(context_vecs)
    print('context_vecs.shape:', context_vecs.shape)

    class multiheadattentionwrapper(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)
    torch.manual_seed(123)
    context_length = batch.shape[1]
    d_in, d_out = (3, 2)
    mha = multiheadattentionwrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print('context_vecs.shape:', context_vecs.shape)

    class MultiHeadAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)
            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)
            return context_vec
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print('context_vecs.shape:', context_vecs.shape)
    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], [0.8993, 0.039, 0.9268, 0.7388], [0.7179, 0.7058, 0.9156, 0.434]], [[0.0772, 0.3565, 0.1479, 0.5331], [0.4066, 0.2318, 0.4545, 0.9737], [0.4606, 0.5159, 0.422, 0.5786]]]])
    print(a @ a.transpose(2, 3))
    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print('\nsecond head:\n', second_res)

    from importlib.metadata import version
    print('matplotlib version', version('matplotlib'))
    print('torch version:', version('torch'))
    print('tiktoken:', version('tiktoken'))
    GPT_CONFIG_124M = {'vocab_size': 50257, 'context_length': 64 if args.quick else 1024, 'emb_dim': 64 if args.quick else 768, 'n_heads': 4 if args.quick else 12, 'n_layers': 2 if args.quick else 12, 'drop_rate': 0.1, 'qkv_bias': False}
    import torch
    import torch.nn as nn

    class DipLoyalServent(nn.Module):

        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
            self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
            self.drop_emb = nn.Dropout(cfg['drop_rate'])
            self.trf_blocks = nn.Sequential(*[DipTransformerBlock(cfg) for _ in range(cfg['n_layers'])])
            self.final_norm = DipLayerNorm(cfg['emb_dim'])
            self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    class DipTransformerBlock(nn.Module):

        def __init__(self, cfg):
            super().__init__()

        def forward(self, x):
            return x

    class DipLayerNorm(nn.Module):

        def __init__(self, normalized_shape, eps=1e-05):
            super().__init__()

        def forward(self, x):
            return x
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    batch = []
    txt1 = 'Every effort moves you'
    txt2 = 'Every day holds a'
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    torch.manual_seed(123)
    model = DipLoyalServent(GPT_CONFIG_124M)
    logits = model(batch)
    print('output shape:', logits.shape)
    print(logits)
    del model, logits
    torch.manual_seed(123)
    batch_example = torch.rand(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True, unbiased=False)
    print('mean:\n', mean)
    print('variance:\n', var)
    out_norm = (out - mean) / torch.sqrt(var + 1e-05)
    print('normalized layer output:\n', out_norm)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True, unbiased=False)
    print('mean:\n', mean)
    print('variance:\n', var)
    torch.set_printoptions(sci_mode=False)
    print('mean:\n', mean)
    print('variance:\n', var)

    class LayerNorm(nn.Module):

        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-05
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift
    ln = LayerNorm(emb_dim=6)
    out_ln = ln(out)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print('Mean:\n', mean)
    print('Variance:\n', var)

    class GELU(nn.Module):

        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh((2.0 / torch.pi) ** 0.5 * (x + 0.044715 * torch.pow(x, 3))))
    import matplotlib.pyplot as plt
    gelu, relu = (GELU(), nn.ReLU())
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = (gelu(x), relu(x))
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'ReLU']), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f'{label} activation funtion')
        plt.xlabel(f'x')
        plt.ylabel(f'{label}(x)')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'activation-plot.pdf')
    plt.show() if args.show_plots else plt.close()

    class FeedForward(nn.Module):

        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), GELU(), nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']))

        def forward(self, x):
            return self.layers(x)
    print(GPT_CONFIG_124M['emb_dim'])

    class ExampleDeepNeuralNetwork(nn.Module):

        def __init__(self, layer_sizes, use_shortcut):
            super().__init__()
            self.use_shortcut = use_shortcut
            self.layers = nn.ModuleList([nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()), nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()), nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()), nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()), nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())])

        def forward(self, x):
            for layer in self.layers:
                layer_output = layer(x)
                if self.use_shortcut and x.shape == layer_output.shape:
                    x = x + layer_output
                else:
                    x = layer_output
            return x

    def print_gradients(model, x):
        output = model(x)
        target = torch.zeros_like(output)
        loss = nn.MSELoss()(output, target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if 'weight' in name and param.grad is not None:
                print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)

    class TransformerBlock(nn.Module):

        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], context_length=cfg['context_length'], num_heads=cfg['n_heads'], dropout=cfg['drop_rate'], qkv_bias=cfg['qkv_bias'])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg['emb_dim'])
            self.norm2 = LayerNorm(cfg['emb_dim'])
            self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

        def forward(self, x):
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)
            x = self.drop_shortcut(x)
            x = x + shortcut
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut
            return x
    torch.manual_seed(123)
    x = torch.rand(2, 4, GPT_CONFIG_124M['emb_dim'])
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print('input shape:', x.shape)
    print('output shape:', output.shape)

    class GPTModel(nn.Module):

        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
            self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
            self.drop_emb = nn.Dropout(cfg['drop_rate'])
            self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
            self.final_norm = LayerNorm(cfg['emb_dim'])
            self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
    torch.manual_seed(123)
    if 'model' in locals():
        del model
    model = GPTModel(GPT_CONFIG_124M)
    with torch.no_grad():
        out = model(batch)
    print('Input batch:\n', batch)
    print('\nOutput shape:', out.shape)
    print(out)
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'Total number of parameters: {total_params:,}')
    print('Token embedding layer shape:', model.tok_emb.weight.shape)
    print('Output layer shape:', model.out_head.weight.shape)
    total_params_gpt2 = total_params - sum((p.numel() for p in model.out_head.parameters()))
    print(f'Hypothetical parameter count if weights were tied (not enabled): {total_params_gpt2:,}')
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f'Total size of the model: {total_size_mb:.2f} MB')

    def generate_text_simple(model, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    start_context = 'Hello, I am'
    encoded = tokenizer.encode(start_context)
    print('encoded:', encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print('encoded_tensor.shape:', encoded_tensor.shape)
    model.eval()
    out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M['context_length'])
    print('Output:', out)
    print('Output length:', len(out[0]))
    print('Generated text:', tokenizer.decode(out[0].tolist()))
    from importlib.metadata import version, PackageNotFoundError
    pkgs = ['matplotlib', 'numpy', 'tiktoken', 'torch', 'tensorflow']
    for p in pkgs:
        try:
            print(f'{p} version: {version(p)}')
        except PackageNotFoundError:
            print(f'{p} is not installed (optional for this script section)')
    import torch
    GPT_CONFIG_124M = {'vocab_size': 50257, 'context_length': 64 if args.quick else 256, 'emb_dim': 64 if args.quick else 768, 'n_heads': 4 if args.quick else 12, 'n_layers': 2 if args.quick else 12, 'drop_rate': 0.1, 'qkv_bias': False}
    torch.manual_seed(123)
    if 'model' in locals():
        del model
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    import tiktoken

    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0)
        return tokenizer.decode(flat.tolist())
    start_context = 'Every effort moves you'
    tokenizer = tiktoken.get_encoding('gpt2')
    token_ids = generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=GPT_CONFIG_124M['context_length'])
    print('Output text:\n', token_ids_to_text(token_ids, tokenizer))

    inputs = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])
    with torch.no_grad():
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print('Token IDs:\n', token_ids)
    print(f'Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}')
    print(f'Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}')
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print('Text 1:', target_probas_1)
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print('Text 2:', target_probas_2)
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    print('Logits shape:', logits.shape)
    print('Targets shape:', targets.shape)
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print('Flattened logits:', logits_flat.shape)
    print('Flattened targets:', targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    perplexity = torch.exp(loss)
    print(perplexity)
    import os
    import requests
    file_path = args.text_file.resolve() if args.text_file else BASE_DIR / 'the-verdict.txt'
    url = 'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt'
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()
    print(text_data[:99])
    print(text_data[:99])
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print('Characters:', total_characters)
    print('Tokens:', total_tokens)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'], stride=GPT_CONFIG_124M['context_length'], drop_last=True, shuffle=True, num_workers=0)
    val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'], stride=GPT_CONFIG_124M['context_length'], drop_last=False, shuffle=False, num_workers=0)
    if total_tokens * train_ratio < GPT_CONFIG_124M['context_length']:
        print("Not enough tokens for the training loader. Try to lower the `GPT_CONFIG_124M['context_length']` or increase the `training_ratio`")
    if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M['context_length']:
        print("Not enough tokens for the validation loader. Try to lower the `GPT_CONFIG_124M['context_length']` or decrease the `training_ratio`")
    print('Train loader:')
    for x, y in train_loader:
        print(x.shape, y.shape)
    print('\nValidation loader:')
    for x, y in val_loader:
        print(x.shape, y.shape)
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()
    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()
    print('Training tokens:', train_tokens)
    print('Validation tokens:', val_tokens)
    print('All tokens:', train_tokens + val_tokens)

    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = (input_batch.to(device), target_batch.to(device))
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            raise ValueError('Data loader has no batches; increase text length or reduce batch/context size.')
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches
    device = select_device(torch, args.device)
    print(f'Using {device} device.')
    model.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print('Training loss:', train_loss)
    print('Validation loss:', val_loss)

    def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
        train_losses, val_losses, track_tokens_seen = ([], [], [])
        tokens_seen, global_step = (0, -1)
        for epoch in range(num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f'Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}')
            generate_and_print_sample(model, tokenizer, device, start_context)
        return (train_losses, val_losses, track_tokens_seen)

    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return (train_loss, val_loss)

    def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace('\n', ' '))
        model.train()
    torch.manual_seed(123)
    if 'model' in locals():
        del model
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = args.epochs
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context='Every effort moves you', tokenizer=tokenizer)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(epochs_seen, train_losses, label='Training loss')
        ax1.plot(epochs_seen, val_losses, linestyle='-.', label='Validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)
        ax2.set_xlabel('Tokens seen')
        fig.tight_layout()
        plt.savefig(BASE_DIR / 'loss-plot.pdf')
        plt.show() if args.show_plots else plt.close()
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    inference_device = torch.device('cpu')
    model.to(inference_device)
    model.eval()
    tokenizer = tiktoken.get_encoding('gpt2')
    token_ids = generate_text_simple(model=model, idx=text_to_token_ids('Every effort moves you', tokenizer).to(inference_device), max_new_tokens=25, context_size=GPT_CONFIG_124M['context_length'])
    print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
    vocab = {'closer': 0, 'every': 1, 'effort': 2, 'forward': 3, 'inches': 4, 'moves': 5, 'pizza': 6, 'toward': 7, 'you': 8}
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor([4.51, 0.89, -1.9, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    def print_sampled_tokens(probas):
        torch.manual_seed(123)
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1000)]
        sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
        for i, freq in enumerate(sampled_ids):
            print(f'{freq} x {inverse_vocab[i]}')
    print_sampled_tokens(probas)

    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'temperature-plot.pdf')
    plt.show() if args.show_plots else plt.close()
    print_sampled_tokens(scaled_probas[1])
    print_sampled_tokens(scaled_probas[2])
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print('Top logits:', top_logits)
    print('Top positions:', top_pos)
    new_logits = torch.where(condition=next_token_logits < top_logits[-1], input=torch.tensor(float('-inf')), other=next_token_logits)
    print(new_logits)
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

    def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError('Provide a nonempty token sequence shaped (batch, tokens)')
        if context_size <= 0 or max_new_tokens < 0 or temperature < 0:
            raise ValueError('Invalid generation length, context size or temperature')
        context_size = min(context_size, model.pos_emb.num_embeddings)
        vocab_size = model.out_head.out_features
        if top_k is not None and not 1 <= top_k <= vocab_size:
            raise ValueError(f'top_k must be between 1 and {vocab_size}')
        if eos_id is not None and not 0 <= eos_id < vocab_size:
            raise ValueError('eos_id is outside the vocabulary')
        was_training = model.training
        model.eval()
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits = model(idx[:, -context_size:])[:, -1, :]
                    if top_k is not None:
                        threshold = torch.topk(logits, top_k).values[:, -1:]
                        logits = logits.masked_fill(logits < threshold, float('-inf'))
                    if temperature > 0:
                        logits = logits / temperature
                        logits = logits - logits.max(dim=-1, keepdim=True).values
                        idx_next = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                    else:
                        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                    if eos_id is not None:
                        idx_next = torch.where(finished[:, None], eos_id, idx_next)
                        finished |= idx_next.squeeze(1).eq(eos_id)
                    idx = torch.cat((idx, idx_next), dim=1)
                    if eos_id is not None and finished.all().item():
                        break
        finally:
            model.train(was_training)
        return idx
    torch.manual_seed(123)
    token_ids = generate(model=model, idx=text_to_token_ids('Every effort moves you', tokenizer).to(inference_device), max_new_tokens=15, context_size=GPT_CONFIG_124M['context_length'], top_k=25, temperature=1.4)
    print('Output text:\n', token_ids_to_text(token_ids, tokenizer))
    torch.save(model.state_dict(), BASE_DIR / 'model.pth')
    if 'model' in locals():
        del model
    model = GPTModel(GPT_CONFIG_124M)
    device = select_device(torch, args.device)
    print('Device:', device)
    model.to(device)
    model.load_state_dict(torch.load(BASE_DIR / 'model.pth', map_location=device, weights_only=True))
    model.eval()
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, BASE_DIR / 'model_and_optimizer.pth')
    checkpoint = torch.load(BASE_DIR / 'model_and_optimizer.pth', map_location=device, weights_only=True)
    if 'model' in locals():
        del model
    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
    if args.skip_pretrained or args.quick:
        print('Finished. Pretrained GPT-2 stage skipped by the selected option.')
        return
    del model, optimizer, checkpoint
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from importlib.metadata import version
    print('TensorFlow version:', version('tensorflow'))
    print('tqdm version:', version('tqdm'))
    import os
    import requests
    import json
    import numpy as np
    import tensorflow as tf
    from tqdm import tqdm

    def download_and_load_gpt2(model_size, models_dir):
        allowed_sizes = ('124M', '355M', '774M', '1558M')
        if model_size not in allowed_sizes:
            raise ValueError(f'Model size not in {allowed_sizes}')
        model_dir = os.path.join(models_dir, model_size)
        base_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models'
        backup_base_url = 'https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2'
        filenames = ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001', 'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']
        os.makedirs(model_dir, exist_ok=True)
        for filename in filenames:
            file_url = f'{base_url}/{model_size}/{filename}'
            backup_url = f'{backup_base_url}/{model_size}/{filename}'
            file_path = os.path.join(model_dir, filename)
            download_file(file_url, file_path, backup_url)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        if tf_ckpt_path is None:
            raise FileNotFoundError(f'No TensorFlow checkpoint found in {model_dir}')
        with open(os.path.join(model_dir, 'hparams.json'), encoding='utf-8') as file:
            settings = json.load(file)
        params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
        return (settings, params)

    def download_file(url, destination, backup_url=None):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.part')
        last_error = None
        for download_url in (url, backup_url):
            if download_url is None:
                continue
            try:
                with requests.get(download_url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    expected = int(response.headers.get('Content-Length', 0))
                    if destination.exists() and expected and destination.stat().st_size == expected:
                        return
                    written = 0
                    with temporary.open('wb') as file, tqdm(total=expected or None, unit='B', unit_scale=True, desc=destination.name) as bar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                                written += len(chunk)
                                bar.update(len(chunk))
                    if written == 0 or (expected and written != expected):
                        raise IOError(f'Incomplete download: {written} bytes; expected {expected}')
                    temporary.replace(destination)
                    return
            except (requests.RequestException, OSError, ValueError) as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                print(f'Download attempt failed for {destination.name}: {exc}')
        raise RuntimeError(f'Could not download {destination.name} from any source') from last_error

    def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
        params = {'blocks': [{} for _ in range(settings['n_layer'])]}
        for name, _ in tf.train.list_variables(ckpt_path):
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
            variable_name_parts = name.split('/')[1:]
            target_dict = params
            if variable_name_parts[0].startswith('h'):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params['blocks'][layer_number]
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array
        return params
    settings, params = download_and_load_gpt2(model_size='124M', models_dir=BASE_DIR / 'gpt2')
    print('Settings:', settings)
    print('Parameter dictionary keys:', params.keys())
    print(params['wte'])
    print('Token embedding weight tensor dimensions:', params['wte'].shape)
    model_configs = {'gpt2-small (124M)': {'emb_dim': 768, 'n_layers': 12, 'n_heads': 12}, 'gpt2-medium (355M)': {'emb_dim': 1024, 'n_layers': 24, 'n_heads': 16}, 'gpt2-large (774M)': {'emb_dim': 1280, 'n_layers': 36, 'n_heads': 20}, 'gpt2-xl (1558M)': {'emb_dim': 1600, 'n_layers': 48, 'n_heads': 25}}
    model_name = 'gpt2-small (124M)'
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({'context_length': 1024, 'qkv_bias': True})
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f'Shape mismatch. Left: {left.shape}, Right: {right.shape}')
        return torch.nn.Parameter(torch.tensor(right))
    import numpy as np

    def load_weights_into_gpt(gpt, params):
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
        for b in range(len(params['blocks'])):
            q_w, k_w, v_w = np.split(params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)
            q_b, k_b, v_b = np.split(params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)
            gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params['blocks'][b]['attn']['c_proj']['w'].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params['blocks'][b]['attn']['c_proj']['b'])
            gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params['blocks'][b]['mlp']['c_fc']['w'].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params['blocks'][b]['mlp']['c_fc']['b'])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params['blocks'][b]['mlp']['c_proj']['w'].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params['blocks'][b]['mlp']['c_proj']['b'])
            gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params['blocks'][b]['ln_1']['g'])
            gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params['blocks'][b]['ln_1']['b'])
            gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params['blocks'][b]['ln_2']['g'])
            gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params['blocks'][b]['ln_2']['b'])
        gpt.final_norm.scale = assign(gpt.final_norm.scale, params['g'])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params['b'])
        gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    torch.manual_seed(123)
    token_ids = generate(model=gpt, idx=text_to_token_ids('Every effort moves you', tokenizer).to(device), max_new_tokens=25, context_size=NEW_CONFIG['context_length'], top_k=50, temperature=1.5)
    print('Output text:\n', token_ids_to_text(token_ids, tokenizer))


if __name__ == '__main__':
    args = parse_args()
    try:
        prepare_dependencies(args)
        main(args)
    except KeyboardInterrupt:
        print('Stopped by user.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'Run failed: {exc}', file=sys.stderr)
        raise
