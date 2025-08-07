# train.py - Complete BERT Pretraining Script (Original Working Version)
# References:
# https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891
# https://ai.plainenglish.io/bert-pytorch-implementation-prepare-dataset-part-1-efd259113e5a

import torch
from torch import nn
from pathlib import Path
from tokenizers import Tokenizer
from huggingface_hub import PyTorchModelHubMixin
import os
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import itertools
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import math
from tqdm import tqdm
from datasets import load_dataset
import wandb
from torchinfo import summary

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
n_warmup_steps = 1000
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
n_segments = 3
block_size = 64
batch_size = 64
embeddings_dims = 128
attn_dropout = 0.1
no_of_heads = 2
dropout = 0.1
epochs = 20
max_lr = 2.5e-5
no_of_encoder_layers = 2

os.makedirs("datasets", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("bert-ur-1", exist_ok=True)

print("Loading dataset...")
raw_ds = load_dataset("El-chapoo/Urdu-1M-news-text", split="train")

def split_urdu(text: str, max_len=30):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

pairs = []
for doc in tqdm(raw_ds):
    sents = [s.strip() for s in split_urdu(doc["News Text"]) if s.strip()]
    for i in range(len(sents) - 1):
        pairs.append([
            " ".join(sents[i].split()[:block_size]),
            " ".join(sents[i + 1].split()[:block_size])
        ])

# print("Urdu pairs:", len(pairs))

# # Tokenization
# text_data = []
# file_count = 0

# def clean_text(text):
#     return text.encode('utf-8', 'ignore').decode('utf-8')

# for sample in tqdm([x[0] for x in pairs]):
#     text_data.append(sample)

# with open(f'./datasets/text.txt', 'w', encoding='utf-8') as fp:
#     fp.write('\n'.join(text_data))

# paths = 'datasets/text.txt'

# # Training own tokenizer
# tokenizer = BertWordPieceTokenizer(
#     clean_text=True,
#     handle_chinese_chars=False,
#     strip_accents=False,
#     lowercase=True
# )

# tokenizer.train(
#     files=paths,
#     vocab_size=10000,
#     min_frequency=5,
#     wordpieces_prefix='##',
#     special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
# )

tokenizer = BertTokenizer.from_pretrained('./bert-ur-1/bert-ur-vocab.txt', local_files_only=True)
vocab_size = tokenizer.vocab_size

class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=block_size):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
    
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, item):
        sent1, sent2, is_next = self.get_nsp(item)
        
        sent1_masked, label1 = self.get_masked_sentences(sent1)
        sent2_masked, label2 = self.get_masked_sentences(sent2)
        
        sent1_masked_cls_and_sep_aded = [self.tokenizer.vocab['[CLS]']] + sent1_masked + [self.tokenizer.vocab['[SEP]']]
        sent2_masked_cls_and_sep_aded = sent2_masked + [self.tokenizer.vocab['[SEP]']]
        label1_padding_added = [self.tokenizer.vocab['[PAD]']] + label1 + [self.tokenizer.vocab['[PAD]']]
        label2_padding_added = label2 + [self.tokenizer.vocab['[PAD]']]
        
        segment_ids = [1 for _ in range(len(sent1_masked_cls_and_sep_aded))] + [2 for _ in range(len(sent2_masked_cls_and_sep_aded))]
        
        combined_sentence = sent1_masked_cls_and_sep_aded + sent2_masked_cls_and_sep_aded
        combined_labels = label1_padding_added + label2_padding_added
        
        if(len(combined_sentence) > self.seq_len):
            combined_sentence = combined_sentence[:self.seq_len]
            combined_labels = combined_labels[:self.seq_len]
            segment_ids = segment_ids[:self.seq_len]
        elif (len(combined_sentence) < self.seq_len):
            while(len(combined_sentence) < self.seq_len):
                combined_sentence = [self.tokenizer.vocab['[PAD]']] + combined_sentence
                segment_ids = [0] + segment_ids
                combined_labels = [0] + combined_labels
        
        values = {
            'bert_input_masked': combined_sentence,
            'bert_input_labels': combined_labels,
            'segment_ids': segment_ids,
            'is_next': is_next
        }
        
        assert len(combined_labels) == len(combined_sentence)
        return {key: torch.tensor(value) for key, value in values.items()}
    
    def get_nsp(self, index):
        t1, t2 = self.lines[index][0], self.lines[index][1]
        prob = random.random()
        if(prob < 0.5):
            return t1, t2, 1
        else:
            return t1, self.lines[random.randrange(len(pairs))][1], 0
    
    def get_masked_sentences(self, sentence):
        tokens = self.tokenizer(sentence)['input_ids'][1:-1]
        mask_label = []
        output = []
        for token in tokens:
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    output.append(self.tokenizer.vocab['[MASK]'])
                elif prob < 0.9:
                    output.append(random.randrange(len(self.tokenizer.vocab)))
                else:
                    output.append(token)
                mask_label.append(token)
            else:
                output.append(token)
                mask_label.append(0)
        assert len(output) == len(mask_label)
        return output, mask_label

dataset = BERTDataset(data_pair=pairs, tokenizer=tokenizer, seq_len=block_size)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=os.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=os.cpu_count())

# Test
sample_data = next(iter(train_loader))
print(sample_data)

# Text embeddings
class TextEmbeddings(nn.Module):
    def __init__(self, vocab_size=vocab_size, embeddings_dims=embeddings_dims):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, device=device, padding_idx=0)
    
    def forward(self, x):
        return self.embeddings_table(x)

# Segment embeddings
class SegmentEmbeddings(nn.Module):
    def __init__(self, n_segments=n_segments, embeddings_dims=embeddings_dims):
        super().__init__()
        self.seg_embds = nn.Embedding(num_embeddings=n_segments, embedding_dim=embeddings_dims, device=device, padding_idx=0)
    
    def forward(self, x):
        return self.seg_embds(x)

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embeddings_dims, device=device)
    
    def forward(self, x):
        return self.layer_norm(x)

# FeedForward Neural Network
class MLPBlock(nn.Module):
    def __init__(self, dropout=dropout, embeddings_size=embeddings_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(device=device, in_features=embeddings_size, out_features=4 * embeddings_size),
            nn.ReLU(),
            nn.Linear(device=device, in_features=4 * embeddings_size, out_features=embeddings_size),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        return self.mlp(x)

# Single Attention Head
class AttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.query = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.keys = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.values = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout)
    
    def forward(self, x, mask=None):
        k = self.keys(x)
        q = self.query(x)
        v = self.values(x)
        
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        if(mask != None):
            masked_values = weights.masked_fill(mask == 0, float('-inf'))
            weights_normalized = nn.functional.softmax(masked_values, dim=-1)
            out = weights_normalized @ v
            out = self.dropout(out)
            return out
        else:
            weights_normalized = nn.functional.softmax(weights, dim=-1)
            out = weights_normalized @ v
            out = self.dropout(out)
            return out

# MHA
class MHA(nn.Module):
    def __init__(self, mask=None, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads) for _ in range(no_of_heads)])
        self.dropout = nn.Dropout(p=attn_dropout)
        self.linear = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims, device=device, bias=False)
    
    def forward(self, x, mask):
        concat = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        linear_layer = self.linear(concat)
        out = self.dropout(linear_layer)
        return out

import math

class PositionEmbeddings(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims, block_size=block_size):
        super().__init__()
        self.pos_embd = torch.ones((block_size, embeddings_dims), device=device, requires_grad=False)
        for pos in range(block_size):
            for i in range(0, embeddings_dims, 2):
                self.pos_embd[pos, i] = math.sin(pos/(10000**((2*i)/embeddings_dims)))
                if i + 1 < embeddings_dims:
                    self.pos_embd[pos, i + 1] = math.cos(pos/(10000**((2*(i + 1))/embeddings_dims)))
    
    def forward(self, x):
        pos_embd = self.pos_embd
        pos_embd = pos_embd.unsqueeze(0)
        return pos_embd

# Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout, mask=None):
        super().__init__()
        self.mha = MHA(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads)
        self.layer_norm1 = LayerNormalization(embeddings_dims=embeddings_dims)
        self.layer_norm2 = LayerNormalization(embeddings_dims=embeddings_dims)
        self.mlp_block = MLPBlock(dropout=dropout, embeddings_size=embeddings_dims)
    
    def forward(self, x, mask):
        x = self.layer_norm1(x + self.mha(x, mask))
        x = self.layer_norm2(x + self.mlp_block(x))
        return x

# Encoder 
class EncoderModel(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, 
                 block_size=block_size, dropout=dropout, no_of_encoder_layers=no_of_encoder_layers, 
                 vocab_size=vocab_size, n_segments=n_segments, mask=None):
        super().__init__()
        self.encoder_layer_stacked = []
        self.positional_embeddings = PositionEmbeddings(block_size=block_size, embeddings_dims=embeddings_dims)
        self.text_embds = TextEmbeddings(vocab_size=vocab_size, embeddings_dims=embeddings_dims)
        self.no_of_encoder_layers = no_of_encoder_layers
        self.layer_norm = LayerNormalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p=dropout)
        self.seg_embds = SegmentEmbeddings(n_segments=n_segments, embeddings_dims=embeddings_dims)
        self.encoder_layer = TransformerEncoderBlock(embeddings_dims=embeddings_dims, attn_dropout=attn_dropout, no_of_heads=no_of_heads, dropout=dropout)
        
        for _ in range(no_of_encoder_layers):
            self.encoder_layer_stacked.append(self.encoder_layer)
    
    def forward(self, x, segment_ids):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)
        x = (self.text_embds(x) + self.seg_embds(segment_ids) + self.positional_embeddings(x)) * math.sqrt(embeddings_dims)
        x = self.dropout(x)
        for layer in self.encoder_layer_stacked:
            x = layer(x, mask=mask)
        out = self.layer_norm(x)
        return x

# NSP
class NSP(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=2, device=device)
    
    def forward(self, x, isnext):
        logits = self.linear_layer(x[:, 0])
        loss = nn.functional.cross_entropy(logits, isnext, ignore_index=0)
        return loss, logits

# MLM
class MLM(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims, vocab_size=vocab_size):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=vocab_size, device=device)
    
    def forward(self, x, mask_labels):
        logits = self.linear_layer1(x)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        mask_labels = mask_labels.view(-1)
        loss = nn.functional.cross_entropy(logits, mask_labels, ignore_index=0)
        return loss, logits

# BERT
class BERT(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads,
                 block_size=block_size, dropout=dropout, vocab_size=vocab_size, n_segments=n_segments):
        super().__init__()
        self.mlm = MLM(embeddings_dims=embeddings_dims, vocab_size=vocab_size)
        self.nsp = NSP(embeddings_dims=embeddings_dims)
        self.encoder_layer = EncoderModel(attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, 
                                        no_of_heads=no_of_heads, no_of_encoder_layers=no_of_encoder_layers, 
                                        block_size=block_size, dropout=dropout, n_segments=n_segments)
    
    def forward(self, x, segment_ids, labels, isnext):
        x = self.encoder_layer(x, segment_ids)
        mlm_loss, mlm_logits = self.mlm(x, labels)
        nsp_loss, nsp_logits = self.nsp(x, isnext)
        return mlm_loss, mlm_logits, nsp_loss, nsp_logits

model = BERT(embeddings_dims=embeddings_dims, vocab_size=vocab_size)
model = model.to(device)

sample_data = {key: value.to(device) for key, value in sample_data.items()}
summary(model=model,
        input_data=(sample_data['bert_input_masked'], sample_data['segment_ids'], 
                   sample_data['bert_input_labels'], sample_data['is_next']),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, embeddings_dims, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(embeddings_dims, -0.5)
    
    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()
    
    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad(set_to_none=True)
    
    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
    
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, eps=epsilon, betas=(beta_1, beta_2))
lr_scheduler = ScheduledOptim(optimizer=optimizer, embeddings_dims=embeddings_dims, n_warmup_steps=n_warmup_steps)


@torch.inference_mode()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, mlm_loss_total, nsp_loss_total = [], [], []
    for batch in val_loader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        mlm_loss, _, nsp_loss, _ = model(
            inputs['bert_input_masked'],
            inputs['segment_ids'],
            inputs['bert_input_labels'],
            inputs['is_next']
        )
        total = mlm_loss + nsp_loss
        total_loss.append(total.item())
        mlm_loss_total.append(mlm_loss.item())
        nsp_loss_total.append(nsp_loss.item())
    model.train()
    return {
        "val_loss": np.mean(total_loss),
        "val_mlm_loss": np.mean(mlm_loss_total),
        "val_nsp_loss": np.mean(nsp_loss_total)
    }

wandb.init(
    project="bert-ur",
    name=f"bert_pretrain_run",
    config={
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": max_lr,
        "warmup_steps": n_warmup_steps,
        "optimizer": "Adam",
        "scheduler": "CustomWarmup",
        "embedding_dim": embeddings_dims
    }
)

def train(model, train_loader, val_loader, device, epochs, lr_scheduler, gradient_clip=1.0, validate_every=100, save_path=None):
    model.to(device)
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss, mlm_loss_list, nsp_loss_list = [], [], []
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            global_step += 1
            inputs = {key: value.to(device) for key, value in batch.items()}
            
            mlm_loss, _, nsp_loss, _ = model(
                inputs['bert_input_masked'],
                inputs['segment_ids'],
                inputs['bert_input_labels'],
                inputs['is_next']
            )
            
            loss = mlm_loss + nsp_loss
            lr_scheduler.zero_grad()
            loss.backward()
            
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            lr_scheduler.step_and_update_lr()
            
            total_loss.append(loss.item())
            mlm_loss_list.append(mlm_loss.item())
            nsp_loss_list.append(nsp_loss.item())
            
            wandb.log({
                "step": global_step,
                "train/total_loss": loss.item(),
                "train/mlm_loss": mlm_loss.item(),
                "train/nsp_loss": nsp_loss.item(),
                "learning_rate": lr_scheduler._optimizer.param_groups[0]['lr']
            }, step=global_step)
            
            if (step + 1) % validate_every == 0 or (step + 1) == len(train_loader):
                val_metrics = evaluate(model, val_loader, device)
                print(f"Step {step+1}:")
                print(f" Train MLM Loss: {np.mean(mlm_loss_list):.4f}, NSP Loss: {np.mean(nsp_loss_list):.4f}, Total Loss: {np.mean(total_loss):.4f}")
                print(f" Val MLM Loss: {val_metrics['val_mlm_loss']:.4f}, NSP Loss: {val_metrics['val_nsp_loss']:.4f}, Total Loss: {val_metrics['val_loss']:.4f}")
                
                wandb.log({
                    "val/total_loss": val_metrics['val_loss'],
                    "val/mlm_loss": val_metrics['val_mlm_loss'],
                    "val/nsp_loss": val_metrics['val_nsp_loss']
                }, step=global_step)
                
                total_loss.clear()
                mlm_loss_list.clear()
                nsp_loss_list.clear()
        
        if save_path:
            ckpt_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved at: {ckpt_path}")
            wandb.save(ckpt_path)

if __name__ == "__main__":
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr_scheduler=lr_scheduler,
        gradient_clip=1.0,
        validate_every=100,
        save_path="./checkpoints"
    )
    
    print("Training completed successfully!")
    wandb.finish()

