
print("at ur service !")

import os #got this from git
import requests #ehhh also fromgit

if not os.path.exists("the-verdict.txt"):
    url = ( 
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"

    )
    file_path = "the-verdict.txt"

    response = requests.get(url,timeout = 30)
    response.raise_for_status()
    with open(file_path, "wb") as f: #im not sure why we use file_path
        f.write(response.content)
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("total number of charecter:", len(raw_text))
print(raw_text[:99])
# we are building tokens example: each word has to be its own "token" or else its to big 
#"punctuation""words" all has to be tokenizedTM
import re
text = "hello, world . i hope the factory treats me nice."
result = re.split(r'(\s)', text)
print(result)
result = re.split(r'([,.])|\s',text)
print (result)
text = "hello, world . i hope the factory treats me nice."
result = re.split((r'([,.!?:;"()]--|\s)'), text)
result = [item.strip() for item in result if item.strip()]
print (result)
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))
# tokenization has ended the code is set up now im moving on to building vocabulary that consists of tokens
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
vocab = {token:integer for integer, token in enumerate(all_words)}
#add first 50 words this box is litrially learning puncuationa & words rn
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break
class simpletokenizerV1:
        def __init__(self,  vocab):
            self.str_to_int = vocab
            self.int_to_str = {i:s for s,i in vocab.items()}
        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            preprocessed = [
                item.strip() for item in preprocessed if item.strip()
            ]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids
        def decode(self, ids):
            text ="". join([self.int_to_str[i] for i in ids])
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return text
        #incode = text --> token IDs
        #decode --> toekn IDs back into text
tokenizer = simpletokenizerV1(vocab)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
tokenizer.decode(ids)
tokenizer.decode(tokenizer.encode(text))
#adding sepecial context tokens
tokenizer = simpletokenizerV1(vocab)
text = "Hello, do you like tea. Is this-- a test?"
tokenizer.encode(text)
all_tokens = sorted(list(set(preprocessed)))
all_token.extend(["<|endoftext|>", "<|unk|>"])
vocab = { token:integer for integer,token in enumerate(all_tokens)}
len(vocab.items())
for i, item in enumerate(list(vocab.items())[-5]):
    print(item)
#new class function introduced
class simpletokenizerv2:
    def __init__(self,vocab):
        self.str_to_int =vocab
        self.int_to_str ={i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_int_int
            else "<|unk|>" for item in preprocessed

        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = "". join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    #modified tokenizer has been built
    tokenizer = simpletokenizerv2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    print(text)
tokenizer.encode(text)
tokenizer.decode(tokenizer.encode(text))
#bytepair encoding inprocess perchance
import importlib
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text,allowed_special = {"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
#now the dunceulator is able to break down words perchance
with open("the-verdict.txt", "r", encoding = "utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:    {y}")
#the x and y cooridinates help out with the arrays

for i in range(1, context_size+1):
    context = enc_sample[:1]
    desired = enc_sample[1]

    print(context, "-->",)
for i in range(1, context_size+1):
    context = enc_sample[:1]
    desired = enc_sample[1]
    print(tokenizer.decode(context),"-->",tokenizer.decode([desired]))

#importing torch
import torch

from torch.utils.data import dataset, dataloader
class GPTDatasetV1(dataset):
    def _init_(self, txt, tokenizer, maxlength,stride):
        self.input_ids = []
        self.target_ids = []
        #tokenizer the entire t
        token_ids = tokenizer.encode(text, allowed_special ={"<|endoftext|>"})
        assert len(token_ids)> max_length, "number token imputs must equal +1"
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1: i + max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def _len_(self):
        return len(self.input_ids)
    def _getitem_(self,idx):
        return self.input_ids[idx], self.target_ids[idx]
def creat_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(
        txt,
        tokenizer,
        max_length,
        stride
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
with open("the - verdict.txt","r", encoding = "utf-8") as f:
    raw_text = f.read()
dataloader = creat_dataloader_v1(
    raw_text, batch_size = 1, maxleangth = 4, stride = 1, shuffe= False
) 
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch= next(data_iter)
print(second_batch)

dataloader = creat_dataloader_v1(raw_text, batch_size=8, max_length = 4, stride = 4, shuffle = False)

data_iter = iter(dataloader)
input, targets = next(data_iter)
print("inputs\n", input)
print("\nTargets:\n", targets)
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.embedding(vocab_size,output_dim)
print(embedding_layer.weight)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))
max_length = 4
dataloader = creat_dataloader_v1(
    raw_text, batch_size=8, max_length= max_length,
    stride= max_length, shuffle = False

)
data_iter = iter(dataloader)
input, targets - next(data_iter)
print("token IDs:\n", input)
print("\nInputs shape\n", input.shape)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

#part 3

from importlib.metadata import version
print("torch version:", version("torch"))
inputs = torch.tonsor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
query = input[1]
attn_scores_2 = torch.empty(input.shape[0])
for i, x_i in enumerate(input):
    attn_scores_2[1] = torch.dot(x_i, query)
print(attn_scores_2)

res = 0
for idx, element in enumerate(input[0]):
    res += input[0][idx] * query[idx]
print(res)
print(torch.dot(input[0], query))

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("attention weights:", attn_weights_2_tmp)
print("sum:", attn_weights_2_tmp.sum())

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("attention weights:", attn_weights_2_naive)
print("sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim = 0)

print("attentionw eights:", attn_weights_2)
print("sum:", attn_weights_2.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

query = input[1]

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(input):
    context_v_2 += attn_weights_2[i]*x_i
print(context_vec_2)

attn_scores = torch.empty(6,6)
for i, x_i in enumerate(input):
    for j, x_j in enumerate(input):
        attn_scores[i , j] = torch.dot(x_i, x_j)
print(attn_scores)

attn_scores = input @ input.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum", row_2_sum)
print("all row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ input
print(all_context_vecs)

print("Previous 2nd context vector:", context_vec_2)

x_2 = input[1]
d_in = input.shape[1]
d_out = 2

torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1] # Python starts index at 0
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2= attn_weights_2 @ values
print(context_vec_2)

import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def _init_(self, d_in, d_out):
        super()._init_()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec
torch.manual_seed(123)
sal_v1 = SelfAttention_v1(d_in, d_out)
print(sal_v1(input))
