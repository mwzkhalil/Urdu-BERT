import torch
from torch import nn
from pathlib import Path
from tokenizers import Tokenizer
from huggingface_hub import PyTorchModelHubMixin
import os
import torch
import math
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
from torch.cuda.amp import autocast, GradScaler
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#Hyperparameters
n_warmup_steps = 1000
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
n_segments = 3
block_size = 64
batch_size = 128 
accumulation_steps = 2  
embeddings_dims = 128
attn_dropout = 0.1
no_of_heads = 2
dropout = 0.1
epochs = 4  
max_lr = 3e-5 
no_of_encoder_layers = 2
patience = 3000  
min_delta = 0.001

# Dynamic masking 
initial_mask_prob = 0.20  #higher
final_mask_prob = 0.12   #lower
mask_decay_steps = 10000


#DIRs
os.makedirs("datasets", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("bert-ur-1", exist_ok=True)

text_data = []
file_count = 0

def clean_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

for sample in tqdm([x[0] for x in pairs]):
    text_data.append(sample)

with open(f'./datasets/test.txt', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(text_data))
        # text_data = []
        # file_count += 1

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train( 
    files=paths,
    vocab_size=10000, 
    min_frequency=5,
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )


# tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert-ur-1/bert-ur-vocab.txt', local_files_only=True)
vocab_size = tokenizer.vocab_size

class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=block_size):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
        self.current_step = 0
    
    def set_step(self, step):
        """Set current training step for dynamic masking"""
        self.current_step = step
    
    def get_dynamic_mask_prob(self):
        """Dynamic masking probability that decreases over time"""
        if self.current_step >= mask_decay_steps:
            return final_mask_prob
        
        progress = self.current_step / mask_decay_steps
        mask_prob = initial_mask_prob - (initial_mask_prob - final_mask_prob) * progress
        return mask_prob
    
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
        
        #Segment 
        segment_ids = [1 for _ in range(len(sent1_masked_cls_and_sep_aded))] + [2 for _ in range(len(sent2_masked_cls_and_sep_aded))]
        
        combined_sentence = sent1_masked_cls_and_sep_aded + sent2_masked_cls_and_sep_aded
        combined_labels = label1_padding_added + label2_padding_added
        
        #Padding 
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
        dynamic_prob = self.get_dynamic_mask_prob()
        
        for token in tokens:
            prob = random.random()
            if prob < dynamic_prob:
                prob /= dynamic_prob
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         pin_memory=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                       pin_memory=True, num_workers=4, persistent_workers=True)

#Test
sample_data = next(iter(train_loader))
print("Sample batch shape:", {k: v.shape for k, v in sample_data.items()})

class TextEmbeddings(nn.Module):
    def __init__(self, vocab_size=vocab_size, embeddings_dims=embeddings_dims):
        super().__init__()
        self.embeddings_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims, device=device, padding_idx=0)
    
    def forward(self, x):
        return self.embeddings_table(x)

class SegmentEmbeddings(nn.Module):
    def __init__(self, n_segments=n_segments, embeddings_dims=embeddings_dims):
        super().__init__()
        self.seg_embds = nn.Embedding(num_embeddings=n_segments, embedding_dim=embeddings_dims, device=device, padding_idx=0)
    
    def forward(self, x):
        return self.seg_embds(x)

class LayerNormalization(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embeddings_dims, device=device)
    
    def forward(self, x):
        return self.layer_norm(x)

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

#MHA
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

#Encoder 
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

#Model
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

#NSP
class NSP(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=2, device=device)
    
    def forward(self, x, isnext):
        logits = self.linear_layer(x[:, 0])
        loss = nn.functional.cross_entropy(logits, isnext, ignore_index=0)
        return loss, logits

#MLM
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

#BERT
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
    def __init__(self, optimizer, embeddings_dims, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(embeddings_dims, -0.5)
    
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()
    
    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)
    
    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])
    
    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=patience, min_delta=min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, eps=epsilon, betas=(beta_1, beta_2))
lr_scheduler = ScheduledOptim(optimizer=optimizer, embeddings_dims=embeddings_dims, n_warmup_steps=n_warmup_steps)
scaler = GradScaler()
early_stopping = EarlyStopping()

#Warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
@torch.inference_mode()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, mlm_loss_total, nsp_loss_total = [], [], []
    
    for batch in val_loader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        
        with autocast():
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
    name=f"bert_pretrain",
    config={
        "epochs": epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "learning_rate": max_lr,
        "warmup_steps": n_warmup_steps,
        "optimizer": "Adam",
        "scheduler": "CustomWarmup",
        "embedding_dim": embeddings_dims,
        "mixed_precision": "fp16",
        "dynamic_masking": True,
        "initial_mask_prob": initial_mask_prob,
        "final_mask_prob": final_mask_prob,
        "early_stopping_patience": patience
    }
)

def train(model, train_loader, val_loader, device, epochs, lr_scheduler, scaler, early_stopping, 
          accumulation_steps=2, gradient_clip=1.0, validate_every=500, save_path=None):
    model.to(device)
    model.train()
    global_step = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss, mlm_loss_list, nsp_loss_list = [], [], []
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            inputs = {key: value.to(device) for key, value in batch.items()}
            
            if hasattr(train_loader.dataset.dataset, 'set_step'):
                train_loader.dataset.dataset.set_step(global_step)
            
            with autocast():
                mlm_loss, _, nsp_loss, _ = model(
                    inputs['bert_input_masked'],
                    inputs['segment_ids'],
                    inputs['bert_input_labels'],
                    inputs['is_next']
                )
                loss = (mlm_loss + nsp_loss) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            #Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                global_step += 1
                
                if gradient_clip:
                    scaler.unscale_(lr_scheduler._optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                scaler.step(lr_scheduler._optimizer)
                scaler.update()
                lr_scheduler.zero_grad()
                lr_scheduler.n_current_steps += 1
                
                lr = lr_scheduler.init_lr * lr_scheduler._get_lr_scale()
                for param_group in lr_scheduler._optimizer.param_groups:
                    param_group['lr'] = lr
            
            actual_loss = loss.item() * accumulation_steps
            total_loss.append(actual_loss)
            mlm_loss_list.append(mlm_loss.item())
            nsp_loss_list.append(nsp_loss.item())
            
            if (step + 1) % accumulation_steps == 0:
                current_mask_prob = train_loader.dataset.dataset.get_dynamic_mask_prob() if hasattr(train_loader.dataset.dataset, 'get_dynamic_mask_prob') else 0.15
                wandb.log({
                    "step": global_step,
                    "train/total_loss": actual_loss,
                    "train/mlm_loss": mlm_loss.item(),
                    "train/nsp_loss": nsp_loss.item(),
                    "learning_rate": lr_scheduler._optimizer.param_groups[0]['lr'],
                    "dynamic_mask_prob": current_mask_prob,
                    "training_time_hours": (time.time() - start_time) / 3600
                }, step=global_step)
            
            if (step + 1) % (validate_every * accumulation_steps) == 0 or (step + 1) == len(train_loader):
                val_metrics = evaluate(model, val_loader, device)
                elapsed_time = (time.time() - start_time) / 3600
                
                print(f"Step {global_step}:")
                print(f" Train MLM Loss: {np.mean(mlm_loss_list):.4f}, NSP Loss: {np.mean(nsp_loss_list):.4f}, Total Loss: {np.mean(total_loss):.4f}")
                print(f" Val MLM Loss: {val_metrics['val_mlm_loss']:.4f}, NSP Loss: {val_metrics['val_nsp_loss']:.4f}, Total Loss: {val_metrics['val_loss']:.4f}")
                print(f" Time elapsed: {elapsed_time:.2f} hours")
                
                wandb.log({
                    "val/total_loss": val_metrics['val_loss'],
                    "val/mlm_loss": val_metrics['val_mlm_loss'],
                    "val/nsp_loss": val_metrics['val_nsp_loss'],
                    "training_time_hours": elapsed_time
                }, step=global_step)
                
                if early_stopping(val_metrics['val_loss']):
                    print(f"Early stopping triggered at step {global_step}")
                    break
                
                if val_metrics['val_loss'] == early_stopping.best_loss:
                    if save_path:
                        best_path = os.path.join(save_path, f"best_model_step_{global_step}.pt")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': lr_scheduler._optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'step': global_step,
                            'val_loss': val_metrics['val_loss']
                        }, best_path)
                        print(f"Best model saved at: {best_path}")
                        wandb.save(best_path)
                
                total_loss.clear()
                mlm_loss_list.clear()
                nsp_loss_list.clear()
                
                if elapsed_time > 6.5:
                    print(f"Approaching 7-hour limit. Stopping training early.")
                    return model
        
        if save_path:
            ckpt_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': lr_scheduler._optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch + 1,
                'step': global_step
            }, ckpt_path)
            print(f"Checkpoint saved at: {ckpt_path}")
            wandb.save(ckpt_path)
        
        if early_stopping.counter > 0:
            print(f"Early stopping counter: {early_stopping.counter}/{early_stopping.patience}")
    
    return model

def calculate_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb, param_size + buffer_size

def save_final_model(model, tokenizer, save_path):
    final_dir = os.path.join(save_path, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
    
    config = {
        "vocab_size": vocab_size,
        "hidden_size": embeddings_dims,
        "num_hidden_layers": no_of_encoder_layers,
        "num_attention_heads": no_of_heads,
        "intermediate_size": 4 * embeddings_dims,
        "hidden_dropout_prob": dropout,
        "attention_probs_dropout_prob": attn_dropout,
        "max_position_embeddings": block_size,
        "type_vocab_size": n_segments,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "model_type": "bert"
    }
    
    import json
    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Final model saved to: {final_dir}")
    return final_dir

if __name__ == "__main__":
    model_size_mb, model_size_bytes = calculate_model_size(model)
    print(f"Model size: {model_size_mb:.2f} MB ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    start_time = time.time()
    
    try:
        trained_model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            accumulation_steps=accumulation_steps,
            gradient_clip=1.0,
            validate_every=500,
            save_path="./checkpoints"
        )
        
        final_model_path = save_final_model(trained_model, tokenizer, "./checkpoints")
        
        total_time = (time.time() - start_time) / 3600

        wandb.log({
            "training_completed": True,
            "total_training_time_hours": total_time,
            "final_model_size_mb": model_size_mb,
            "total_parameters": sum(p.numel() for p in model.parameters())
        })
        
    except KeyboardInterrupt:
        emergency_path = os.path.join("./checkpoints", "emergency_checkpoint.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': lr_scheduler._optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'interrupted': True
        }, emergency_path)
        print(f"Emergency checkpoint saved: {emergency_path}")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        wandb.finish()
        print("Training session ended.")
