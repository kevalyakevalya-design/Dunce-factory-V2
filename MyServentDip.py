
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

dataloader = creat_dataloader_v1(raw_text, batch_size=8, max_length = 4, stride = 4, stride = 4, shuffle = False)

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


