import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import time
from torch.utils.data import DataLoader
import wandb
import warnings
warnings.filterwarnings("ignore")



from dataloader import SpeechDataset, TestSpeechDataset
from models import LAS
from train_test import train, validate, inference, save_model, plot_attention
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB = ['<pad>', '<sos>', '<eos>', 'A',   'B',    'C',    'D', 'E',   'F',    'G',    'H',    
         'I',   'J',    'K',    'L', 'M',   'N',    'O',    'P', 'Q',   'R',    'S',    'T', 
         'U',   'V',    'W',    'X', 'Y',   'Z',    "'",    ' ',]

VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}
PAD_TOKEN = VOCAB_MAP["<pad>"]

config = {
  'batch_size': 128,
  'lr':1e-4,
  'epochs': 50,
  'listener_hidden_size':320,
  'speller_embedding_dim': 256,
  'speller_hidden_size' : 512,
  'speller_hidden_dim': 512,
  'projection_size': 128,
  'max_timesteps':550,
  'checkpoint_path': '/home/ubuntu/model/best_las.pth',
  'epoch_checkpoint_path': '/home/ubuntu/model/epoch_las.pth'
}


# Get dataloaders
train_dataset = SpeechDataset(VOCAB)
dev_dataset   = SpeechDataset(VOCAB, partition='dev-clean')
test_dataset  = TestSpeechDataset()
train_loader  = DataLoader(dataset=train_dataset, num_workers=4, batch_size=config['batch_size'], shuffle=False, collate_fn=train_dataset.collate_fn)
dev_loader    = DataLoader(dataset=dev_dataset, num_workers=4, batch_size=config['batch_size'], shuffle=False, collate_fn=dev_dataset.collate_fn)
test_loader   = DataLoader(dataset=test_dataset, num_workers=4, batch_size=config['batch_size'], shuffle=False, collate_fn=test_dataset.collate_fn)

# Define the model and the optimizer
model = LAS(listener_hidden_size=config['listener_hidden_size'],
            speller_embedding_dim=config['speller_embedding_dim'],
            speller_hidden_dim=config['speller_hidden_dim'],
            speller_hidden_size=config['speller_hidden_size'],
            projection_size=config['projection_size'],
            max_timesteps=config['max_timesteps'])

optimizer   = torch.optim.Adam(model.parameters(), lr= config['lr'])
criterion   = torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=PAD_TOKEN)
scaler      = torch.cuda.amp.GradScaler()
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', factor =0.5, patience=3)
tf_rate     = 1.0

wandb.login(key = "")
run = wandb.init(
    name = "LAS model",
    reinit=True,
    id= '',
    resume= 'must',
    project='LAS',
    config = config
)

wandb.save('LAS.txt')
wandb.watch(model, log="all")
best_lev_dist = float("inf")


model.to(DEVICE)
print(model)

for epoch in config['epochs']:
    for param_group in optimizer.param_groups:
        curr_lr = param_group['lr']
        print("Current lr: \t {}".format(curr_lr))
        
    print("Start Train \t{ Epoch}".format(epoch))
    startTime = time.time()
    running_loss, running_perplexity, attention_plot= train(model, train_loader, criterion, optimizer, scaler, tf_rate)
    print("Start Dev \t{} Epoch".format(epoch))
    lev_dist    = validate(model, dev_loader)
    print("*** Saving Checkpoint ***")
    save_model(model, optimizer, scheduler, ['lev_dist', lev_dist], epoch, path=config['epoch_checkpoint_path'])
    # Print your metrics
    print("\tTrain Loss {:.04f}%\trunning_perplexity {:.04f}\t Learning Rate {:.07f}\t TF ratio {:.04f}".format(running_loss, running_perplexity, curr_lr, tf_rate))
    print("\tVal Levenshtein distance {:.04f}".format(lev_dist))
    
    # Plot Attention for a single item in the batch
    plot_attention(attention_plot[0].cpu().detach().numpy())
    
    # Log metrics to Wandb
    wandb.log({'Running loss': running_loss, 'Running perplexity': running_perplexity, 
               'Levenshtein distance': lev_dist, 'lr': curr_lr, 'tf_rate': tf_rate})
    
    scheduler.step(lev_dist)
    
    if lev_dist < best_lev_dist:
        best_lev_dist = lev_dist
        save_model(model, optimizer, scheduler, ['lev_dist', lev_dist], epoch, config['checkpoint_path'])
        wandb.save(config['checkpoint_path'])
        print('Saved best model')
        
wandb.finish()

state_dict = torch.load(config['checkpoint_path'])
model.load_state_dict(state_dict['model_state_dict'])
model.to(DEVICE)

df = inference(model, test_loader)