import torch
import yaml
from addict import Dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = ['<pad>', '<sos>', '<eos>', 'A',   'B',    'C',    'D', 'E',   'F',    'G',    'H',    
         'I',   'J',    'K',    'L', 'M',   'N',    'O',    'P', 'Q',   'R',    'S',    'T', 
         'U',   'V',    'W',    'X', 'Y',   'Z',    "'",    ' ',]

VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}


PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]


def load_config(config_file_path = 'config/config.yaml')->Dict:
    with open(config_file_path, 'r') as cfg:
        config = yaml.safe_load(cfg)
        
    return Dict(config)