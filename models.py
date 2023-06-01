import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torchinfo as t_summary
import math
from matplotlib import pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB = ['<pad>', '<sos>', '<eos>', 'A',   'B',    'C',    'D', 'E',   'F',    'G',    'H',    
         'I',   'J',    'K',    'L', 'M',   'N',    'O',    'P', 'Q',   'R',    'S',    'T', 
         'U',   'V',    'W',    'X', 'Y',   'Z',    "'",    ' ',]

VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}


PAD_TOKEN = VOCAB_MAP["<pad>"]
SOS_TOKEN = VOCAB_MAP["<sos>"]
EOS_TOKEN = VOCAB_MAP["<eos>"]


class pBLSTM(torch.nn.Module):

    
    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)# TODO: Initialize a single layer bidirectional LSTM with the given input_size and hidden_size

    def forward(self, x_packed):

        x, x_lens = pad_packed_sequence(x_packed, batch_first=True)
        
        x, x_lens = self.trunc_reshape(x, x_lens)
        packed_in = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        packed_out, packed_length = self.blstm(packed_in)

        return packed_out

    def trunc_reshape(self, x, x_lens): 
        if x.size(1) % 2 == 1:
            x = F.pad(x, (0,0,0,1,0,0), "constant", 0)
        x = torch.reshape(x, (x.size(0), x.size(1)//2, x.size(2)*2))
        x_lens  = torch.clamp(x_lens, max=x.shape[1])
        return x, x_lens
    
    
class Listener(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Listener, self).__init__()

        self.lstm             = torch.nn.LSTM(27, hidden_dim, batch_first =True, num_layers=3, bidirectional =True, dropout =0.2)

        self.pblstm_encoder   = torch.nn.Sequential(
                                pBLSTM(hidden_dim*4, hidden_dim),
                                pBLSTM(hidden_dim*4, hidden_dim),
                                pBLSTM(hidden_dim*4, hidden_dim)
        )
    
    def forward(self, x, lx):
        pack_padded_sequence    = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)

        rnn, _                  = self.lstm(pack_padded_sequence)

        encoding                = self.pblstm_encoder(rnn)

        output, output_length   = pad_packed_sequence(encoding, batch_first= True)

        return output, output_length
    

class Attention(torch.nn.Module):
    def __init__(self, listener_hidden_size, speller_hidden_size, projection_size):
        super(Attention, self).__init__()
        self.VW               = torch.nn.Linear(listener_hidden_size*2, projection_size)
        self.KW               = torch.nn.Linear(listener_hidden_size*2, projection_size)
        self.QW               = torch.nn.Linear(speller_hidden_size, projection_size)
        self.dropout          = torch.nn.Dropout(p=0.2)          
    
    def set_key_value(self, encoder_outputs, encoder_output_length):
        '''
        In this function we take the encoder embeddings and make key and values from it.
        key.shape   = (batch_size, timesteps, projection_size)
        value.shape = (batch_size, timesteps, projection_size)
        '''
        self.key            = self.KW(encoder_outputs)
        self.value          = self.VW(encoder_outputs)
        self.mask           = torch.arange(encoder_outputs.size(1)).unsqueeze(0) >= encoder_output_length.unsqueeze(1)

    def compute_context(self, decoder_context):
        '''
            In this function from decoder context, we make the query, and then we
            multiply the queries with the keys to find the attention logits, 
            finally we take a softmax to calculate attention energy which gets 
            multiplied to the generted values and then gets summed.

        key.shape   = (batch_size, timesteps, projection_size)
        value.shape = (batch_size, timesteps, projection_size)
        query.shape = (batch_size, projection_size)

        You are also recomended to check out Abu's Lecture 19 to understand Attention better.
        '''
        query             = self.QW(decoder_context)

        q_dim             = query.size(-1)

        raw_weights       = torch.bmm(self.key, query.unsqueeze(2)).squeeze()/math.sqrt(q_dim)

        raw_weights.masked_fill_(self.mask.to('cuda'), float('-inf'))

        attention_weights = F.softmax(raw_weights, dim=1)

        attention_weights = self.dropout(attention_weights)
        
        attention_context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)

        return attention_context, attention_weights
    
    
class LockedDropout(torch.nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()


    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'
            
            
class Speller(torch.nn.Module):

  # Refer to your HW4P1 implementation for help with setting up the language model.
  # The only thing you need to implement on top of your HW4P1 model is the attention module and teacher forcing.

  def __init__(self, attender:Attention, vocab_size, embedding_dim, hidden_dim,projection_size, max_timesteps,  tie_weights):
    super(). __init__()


    self.embedding_dim = embedding_dim
    self.hidden_dim    = hidden_dim
    self.tie_weights   = tie_weights

    self.attend = attender # Attention object in speller
    self.max_timesteps = max_timesteps # Max timesteps
    # self.batch_size    = config['batch_size']

    self.embedding =  torch.nn.Embedding(vocab_size, embedding_dim,padding_idx=PAD_TOKEN)    # Embedding layer to convert token to latent space
    self.lstm_cells =  torch.nn.Sequential(
        torch.nn.LSTMCell(embedding_dim + projection_size, hidden_dim), # TODO: Enter the parameters for the LSTMCells the 128 is the attention projection size
        # You can add multiple LSTMCells too if you want
        torch.nn.LSTMCell(hidden_dim , hidden_dim),
        torch.nn.LSTMCell(hidden_dim, hidden_dim)
        
    )# Create a sequence of LSTM Cells
    
    # For CDN (Feel free to change)
    self.output_to_char = torch.nn.Linear(hidden_dim + projection_size, self.embedding_dim)#just before the cdn mean to add context changed this to embedding dim for weight tying
    # Needs clarification Linear module to convert outputs to correct hidden size (Optional: TO make dimensions match)
    self.activation = torch.nn.GELU()# Check which activation is suggested
    self.char_prob =  torch.nn.Linear(self.embedding_dim, vocab_size) #Actual cdn
    # Linear layer to convert hidden space back to logits for token classification
    if self.tie_weights:
        self.char_prob.weight = self.embedding.weight # Weight tying (From embedding layer)

    self.locked_dropout = LockedDropout()


  def lstm_step(self, input_word, hidden_state):

    for i in range(len(self.lstm_cells)):
        hidden_state[i]       = self.lstm_cells[i](input_word, (hidden_state[i]))
        input_word            = hidden_state[i][0]
        input_word            = self.locked_dropout(input_word.unsqueeze(0)).squeeze()
        
  
    return input_word, hidden_state 
    
  def CDN(self,encoder_context):

      output_to_char    = self.output_to_char(encoder_context)
      output_to_char  = self.activation(output_to_char)
      project = self.char_prob(output_to_char)
      return  project
    
  def forward (self, x, y=None, teacher_forcing_ratio=1, hidden_states_list = None):

    batch_size    = x.shape[0]

    attn_context = torch.zeros((batch_size, 128)).to(DEVICE)   # initial context tensor for time t = 0 self.batch_size 
    output_symbol = torch.zeros(batch_size, dtype = torch.long).fill_(SOS_TOKEN).to(DEVICE) # Set it to SOS for time t = 0 self.batch_size
    raw_outputs = []  
    attention_plot = []
      
    if y is None:
        timesteps = self.max_timesteps
        teacher_forcing_ratio = 0 #Why does it become zero?

    else:
        timesteps = y.shape[1] #raise NotImplementedError # How many timesteps are we predicting for?

    # Initialize your hidden_states list here similar to HW4P1
    hidden_states_list = [None]*len(self.lstm_cells) if hidden_states_list == None else hidden_states_list

    if hidden_states_list == None:
        for i in range(len(self.lstm_cells)):
            hidden_states_list[i] = (torch.nn.init.xavier_normal_(torch.zeros(batch_size, self.hidden_dim).to(DEVICE)), torch.nn.init.xavier_normal_(torch.zeros(batch_size, self.hidden_dim).to(DEVICE)))
            
            
    for t in range(timesteps):
        p = torch.rand(1) # generate a probability p between 0 and 1

        if y is not None:
            if p < teacher_forcing_ratio and t > 0: # Why do we consider cases only when t > 0? What is considered when t == 0? Think.
                output_symbol = y[:,t-1]  # Take from y, else draw from probability distribution


        char_embed = self.embedding(output_symbol)   #raise NotImplementedError # Embed the character symbol
        # Concatenate the character embedding and context from attention, as shown in the diagram
        lstm_input = torch.cat([char_embed, attn_context], dim =1)  
        rnn_out, hidden_states_list = self.lstm_step(lstm_input, hidden_states_list) # Feed the input through LSTM Cells and attention.

        attn_context, attn_weights = self.attend.compute_context(rnn_out) # Feed the resulting hidden state into attention
      

        cdn_input = torch.cat([rnn_out, attn_context], dim = 1)  # TODO: You need to concatenate the context from the attention module with the LSTM output hidden state, as shown in the diagram

        raw_pred = self.CDN(cdn_input) #raise NotImplementedError # call CDN with cdn_input

        
        output_symbol = torch.argmax(raw_pred, dim =-1)#

        raw_outputs.append(raw_pred) # for loss calculation
        attention_plot.append(attn_weights) # for plotting attention plot
   
    attention_plot = torch.stack(attention_plot, dim=1)
    raw_outputs = torch.stack(raw_outputs, dim=1)

    return raw_outputs, attention_plot


    
class LAS(torch.nn.Module):
    def __init__(self,listener_hidden_size, speller_embedding_dim, speller_hidden_dim,speller_hidden_size, projection_size, max_timesteps): # add parameters
        super(LAS, self).__init__()

        # Pass the right parameters here
        self.listener = Listener(listener_hidden_size)
        self.attend = Attention(listener_hidden_size, speller_hidden_dim, projection_size)
        self.speller = Speller(self.attend, len(VOCAB), speller_embedding_dim, speller_hidden_size,projection_size, max_timesteps, True)


    def forward(self, x, lx, y=None, teacher_forcing_ratio=1):
        # Encode speech features
        encoder_outputs, encoder_length = self.listener(x,lx)

        # We want to compute keys and values ahead of the decoding step, as they are constant for all timesteps
        # Set keys and values using the encoder outputs
        self.attend.set_key_value(encoder_outputs, encoder_length)

        # Decode text with the speller using context from the attention
        raw_outputs, attention_plots = self.speller(x,y=y,teacher_forcing_ratio=teacher_forcing_ratio)

        return raw_outputs, attention_plots