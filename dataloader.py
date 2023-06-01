import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


VOCAB = ['<pad>', '<sos>', '<eos>', 'A',   'B',    'C',    'D', 'E',   'F',    'G',    'H',    
         'I',   'J',    'K',    'L', 'M',   'N',    'O',    'P', 'Q',   'R',    'S',    'T', 
         'U',   'V',    'W',    'X', 'Y',   'Z',    "'",    ' ',]

VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}

PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]

root = "/home/ubuntu"


class SpeechDataset(torch.utils.data.Dataset):
    '''
    Feel free to add arguments, additional functions, this is the 
    bare-minimum template.
    '''
    def __init__(self,VOCAB, partition ='train-clean-100'):
        mfcc_dir                = os.path.join(root, partition, 'mfcc')
        transcript_dir          = os.path.join(root, partition, 'transcript')

        mfcc_files         = sorted(os.listdir(mfcc_dir))
        transcript_files   = sorted(os.listdir(transcript_dir))

        self.VOCAB              = VOCAB

        assert len(mfcc_files)  == len(transcript_files)

        self.length             = len(mfcc_files)

        self.mfccs              = list()
        self.transcripts        = list()

        for i in range(self.length):
            # Load the mfcc files from the numpy file 
            mfcc                = np.load(os.path.join(mfcc_dir,mfcc_files[i]))
            # Normalize the mfccs by subtracting the mean and dividing by the std
            mfcc                = (mfcc - np.mean(mfcc, axis =0))/np.std(mfcc, axis =0)
            # Load the transcript files from the numpy folder
            transcript          = np.load(os.path.join(transcript_dir, transcript_files[i]))
            # Convert the transcript files into an array
            transcript          = np.array([VOCAB.index(i) for i in transcript])
            # Append the transcript and mfcc to the list created earlier
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

    
    def __len__(self,):
        # Return the lenth of the mfcc files
        return self.length
    
    def __getitem__(self,index):
        # Convert the mfcc and transcripts into tensors
        mfcc                   = torch.FloatTensor(self.mfccs[index])
        transcript             = torch.LongTensor(self.self.transcripts[index])

        return mfcc, transcript
    
    def collate_fn(self,batch):
        # Extract the mfcc and transcripts as lists
        batch_mfcc, batch_transcript = zip(*batch)

        # Use pad sequence to create arrays of the same length
        batch_mfcc_pad              = pad_sequence(batch_mfcc, batch_first=True, padding_value= PAD_TOKEN)
        mfcc_length                 = [element.size(dim=0) for element in batch_mfcc]

        batch_transcript_pad        = pad_sequence(batch_transcript, batch_first=True,padding_value= PAD_TOKEN)
        transcript_length           = [element.size(dim=0) for element in batch_transcript_pad]

        return batch_mfcc_pad, batch_transcript_pad, torch.tensor(mfcc_length), torch.tensor(transcript_length)
  
  
class TestSpeechDataset(torch.utils.data.Dataset):
    '''
    Feel free to add arguments, additional functions, this is the 
    bare-minimum template.
    '''
    def __init__(self, partition ='test-clean'):
        mfcc_dir                = os.path.join(root, partition, 'mfcc')

        mfcc_files              = sorted(os.listdir(mfcc_dir))

        self.length             = len(mfcc_files)

        self.mfccs              = list()

        for i in range(self.length):
            # Load the mfcc files from the numpy file 
            mfcc                = np.load(os.path.join(mfcc_dir,mfcc_files[i]))
            # Normalize the mfccs by subtracting the mean and dividing by the std
            mfcc                = (mfcc - np.mean(mfcc, axis =0))/np.std(mfcc, axis =0)
            # Append the mfcc to the list created earlier
            self.mfccs.append(mfcc)

    
    def __length__(self,):
        # Return the lenth of the mfcc files
        return self.length
    
    def __getitem__(self,index):
        # Convert the mfcc and transcripts into tensors
        mfcc                   = torch.FloatTensor(self.mfccs[index])
        return mfcc
    
    def collate_fn(self,batch):
        # Extract the mfcc and transcripts as lists
        batch_mfcc                  = batch

        # Use pad sequence to create arrays of the same length
        batch_mfcc_pad              = pad_sequence(batch_mfcc, batch_first=True, padding_value= PAD_TOKEN)
        mfcc_length                 = [element.size(dim=0) for element in batch_mfcc]
        
        return batch_mfcc_pad, torch.tensor(mfcc_length)