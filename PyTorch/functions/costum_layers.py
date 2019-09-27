#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:22:38 2018

@author: danny
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from load_embeddings import load_word_embeddings
import torch
import torch.nn as nn
import numpy as np

####### Costum implementation of Recurrent Highway Networks ###################
# Rather slow, and not much better than regular gru/lstms. 

# implementation of recurrent highway networks using existing PyTorch layers
class RHN(nn.Module):
    def __init__(self, in_size, hidden_size, n_steps, batch_size):
        super(RHN, self).__init__()
        self.n_steps = n_steps
        self.initial_state = torch.autograd.Variable(torch.rand(1, batch_size, 
                                                                hidden_size)).gpu()
        # create 3 linear layers serving as the hidden, transform and carry 
        #gate, one each for each microstep. 
        self.H, self.T, self.C = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # linear layers for the input in the first microstep
        self.init_h = (nn.Linear(in_size, hidden_size))
        self.init_t = (nn.Linear(in_size, hidden_size))
        self.init_c = (nn.Linear(in_size, hidden_size))
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        # linear layers for the history in the microsteps
        layer_list = [self.H, self.T, self.C]
        for steps in range(self.n_steps):
            for layers, lists in zip(self.create_microstep(hidden_size), 
                                     layer_list):
                lists.append(layers)
                
    # initialise linear layers for the microsteps
    def create_microstep(self, n_nodes):       
        H = nn.Linear(n_nodes,n_nodes)
        T = nn.Linear(n_nodes,n_nodes)
        C = nn.Linear(n_nodes,n_nodes)
        return(H,T,C)
    # the input is only used in the first microstep. For the first step the 
    # history is a random initial state.
    def calc_htc(self, x, hx, step, non_linearity):
        if step == 0:
            return non_linearity(((step + 1) // 1 * self.init_h(x)) + \
                                 self.H[step](hx))
        else:
            return non_linearity(self.H[step](hx))
                                 
    def perform_microstep(self, x, hx, step):
        output = self.calc_htc(x, hx, step, self.tan) * \
                 self.calc_htc(x, hx, step, self.sig) + hx * \
                 (1 - self.calc_htc(x, hx, step, self.sig))
        return(output)
        
    def forward(self, input):
        # list to append the output of each time step to
        output = []
        hx = self.initial_state
        # loop through all time steps
        for x in input:
            # apply the microsteps to the hidden state of the GRU
            for step in range(self.n_steps):
                hx = self.perform_microstep(x, hx, step)
            # append the hidden state of time step n to the output. 
            output.append(hx)
        return torch.cat(output)

###############################################################################
        
# class for making multi-headed attenders. 
class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss 
            # function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)
    
# perceptron self attention layer
class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(nn.functional.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x 
    
    # alternative to the forward function. This does not allow the attention
    # to peek at the future, allowing application in e.g. next word prediction
    def no_peek(self, input):
        # calculate the attention weights
        self.alpha = self.out(nn.functional.tanh(self.hidden(input)))
        x = self.apply_attention(input)
        return x   
    
    def apply_attention(self, input):
        att_applied = torch.zeros(input.shape)
        for x in range(1, input.shape[1]):
            _alpha = self.softmax(self.alpha[:,:x,:])
            att_applied[:, x - 1, :] = torch.sum(_alpha * input[:,:x,:], 1)
        return att_applied

################################ Transformer Layers ###########################
# Implementation of the Transformer encoder & decoder. Can be instantiated and 
# used seperately (e.g. in next word prediction). This script implements 
# encoder and decoder cells, their attention and linear layer blocks and 
# classes for stacking multiple encoder/decoder layers. The transformer 
# superclass contains useful functions for creating the masks, positional 
# embeddings and costum 'forward' functions

# single encoder transformer cell with h attention heads fully connected layer 
# block and residual connections
class transformer_encoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_encoder_cell, self).__init__()
        # assert input size is compatible with the number of attention heads
        assert in_size % h == 0
        # the encoder applies an attention block and a linear block, with
        # a residual connection and layer normalisation after each block
        self.att_heads = transformer_att(in_size, h)

        self.ff = transformer_ff(in_size, fc_size)
    
        self.norm_att = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, mask = None):
        # attention block, for encoder q = k = v = input
        att = self.att_heads(input, input, input, mask)
        norm_att = self.norm_att(self.dropout(att) + input)        

        lin = self.ff(norm_att)
        out = self.norm_ff(self.dropout(lin) + norm_att)
        return out

# decoder cell, has an extra attention block which receives encoder output as 
# its input
class transformer_decoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_decoder_cell, self).__init__()
        # assert input size is compatible with the number of attention heads
        assert in_size % h == 0
        # the decoder applies two attention blocks, followed by linear block
        # with a residual connection and layer normalisation after each block
        self.att_one = transformer_att(in_size, h)
        self.att_two = transformer_att(in_size, h)

        self.ff = transformer_ff(in_size, fc_size)
   
        self.norm_att_one = nn.LayerNorm(in_size)
        self.norm_att_two = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)
        
        self.dropout = nn.Dropout(0.1)
        
    # the decoder has different masks for the first and second attention layer.
    # the encoder input is optional, if not provided the decoder acts 
    # basically as an encoder with two attention layers.
    def forward(self, input, dec_mask = None, enc_mask = None, 
                enc_input = None):
        
        # in the first layer q = k = v = input.
        att = self.att_one(input, input, input, dec_mask)
        norm_att = self.norm_att_one(self.dropout(att) + input)
        
        # in the second att block, q is the intermediate dec output, k and v 
        # are the final states of the encoder. 
        if enc_input is None:
            # if no enc_input is provided default to using the intermediate 
            # dec output and the decoder mask
            enc_input = norm_att
            enc_mask = dec_mask
        att_2 = self.att_two(norm_att, enc_input, enc_input, enc_mask)
        norm_att_2 = self.norm_att_two(self.dropout(att_2) + norm_att)

        lin = self.ff(norm_att_2)
        out = self.norm_ff(self.dropout(lin) + norm_att_2)
        
        return out
    
# the linear layer block (feed forward) of the transformer applies 2 linear 
# layers with relu activation in between
class transformer_ff(nn.Module):
    def __init__(self, in_size, fc_size):
        super(transformer_ff, self).__init__()
        self.ff_1 = nn.Linear(in_size, fc_size)
        self.ff_2 = nn.Linear(fc_size, in_size)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        
        output = self.ff_2(self.relu(self.ff_1(input)))
        
        return output
    
# transformer attention head with in_size equal to transformer input size and 
# hidden size equal to in_size/h (number of attention heads) 
class transformer_att(nn.Module):
    def __init__(self, in_size, h):
        super(transformer_att, self).__init__()
        self.att_size = int(in_size/h)
        # create the Query, Key and Value parts of the attention head
        self.Q = nn.Linear(in_size, in_size, bias = False)
        self.K = nn.Linear(in_size, in_size, bias = False)
        self.V = nn.Linear(in_size, in_size, bias = False)
        # att block linear output layer
        self.fc = nn.Linear(in_size, in_size, bias = False)
        
        self.softmax = nn.Softmax(dim = -1)
        self.h = h
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask = None):
        # scaling factor for the attention scores
        scale = torch.sqrt(torch.FloatTensor([self.h])).item() 
        batch_size = q.size(0)
        # apply the linear transform to the query, key and value and reshape 
        # the result into h attention heads
        Q = self.Q(q).view(batch_size, -1, self.h, self.att_size).transpose(1,2)
        K = self.K(k).view(batch_size, -1, self.h, self.att_size).transpose(1,2)
        V = self.V(v).view(batch_size, -1, self.h, self.att_size).transpose(1,2)
        # multiply and scale q and v to get the attention scores
        self.alpha = torch.matmul(Q,K.transpose(-2,-1))/scale
        # apply mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1)
            self.alpha = self.alpha.masked_fill(mask == 0, -1e9)
        # apply softmax to the (masked)attention scores and apply them to V
        self.alpha = self.softmax(self.alpha)
        att_applied = torch.matmul(self.dropout(self.alpha), V)   
        
        # reshape the attention heads and finally pass them through a fully 
        # connected layer
        att = att_applied.transpose(1, 2).reshape(batch_size, -1, 
                                                  self.att_size * self.h)
        output = self.fc(att)   
        return output

# the transformer encoder capable of stacking multiple transformer cells. 
class transformer_encoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_encoder, self).__init__()
        # create one or more multi-head attention layers
        self.tf_stack = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.tf_stack.append(transformer_encoder_cell(in_size, fc_size, h))
            
    def forward(self, input, mask = None):
        # apply the (stacked) transformer
        for tf in self.tf_stack:
            input = tf(self.dropout(input), mask)
        return(input)

# the transformer decoder capable of stacking multiple transformer cells. 
class transformer_decoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_decoder, self).__init__()
        # create one or more multi-head attention layers
        self.tf_stack = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.tf_stack.append(transformer_decoder_cell(in_size, fc_size, h))
            
    def forward(self, input, dec_mask = None, enc_mask = None, 
                enc_input = None):
        # apply the (stacked) transformer
        for tf in self.tf_stack:
            input = tf(self.dropout(input), dec_mask, enc_mask, enc_input)
        return(input)

# super class which transformer architectures should inherit. Contains
# costum 'forward' functions for the Transformer: encoder-decoder training, 
# encoder-decoder prediction/beam search and encoder training. Also contains 
# functions to load pre-trained embeddings, create encoder/decoder masks and
# positional embeddings  
class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        pass    
    # option to load pretrained word embeddings. Takes the dictionary of words
    # occuring in the training data add the file location of the embeddings.
    def load_embeddings(self, dict_loc, embedding_loc):
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)
    
    # function to create the positional embeddings
    def pos_embedding(self, sent_len, d_model):
        pos_emb = torch.zeros(int(sent_len), d_model)
        for x in range(0, sent_len):
            for y in range (0, d_model, 2):
                pos_emb[x, y] = torch.sin(torch.Tensor([x / (10000 ** \
                                                       ((2 * y) / d_model))])
                                          )
                pos_emb[x, y + 1] = torch.cos(torch.Tensor([x / (10000 ** \
                                              ((2 * (y + 1)) / d_model))])
                                              )
        if self.is_cuda == True:
            pos_emb = pos_emb.cuda()
        return pos_emb
    
    # create the encoder mask, masking the padding indices 
    def create_enc_mask(self, input):
        return (input != 0).unsqueeze(1)
    
    # create the decoder mask, masking the padding and for each timestep
    # all future timesteps
    def create_dec_mask(self, input):
        seq_len = input.size(1)
        # create a mask which masks the padding indices
        mask = (input != 0).unsqueeze(1).byte()
        # create a mask which masks for each time-step the future time-steps
        triu = (np.triu(np.ones([1, seq_len, seq_len]), k = 1) == 0).astype('uint8')
        
        if self.is_cuda == True:
            dtype = torch.cuda.ByteTensor
        else:
            dtype = torch.ByteTensor
        # combine the two masks
        return dtype(triu) & dtype(mask)
            
    # Function used for training a transformer encoder-decoder, use in 
    # the forward function of your network
    def encoder_decoder_train(self, enc_input, dec_input):
        # create the targets for the loss function (decoder input, shifted to 
        # the left, padded with a zero)
        targs = torch.nn.functional.pad(dec_input[:, 1:], [0, 1]).long()

        # create the encoder mask which is 0 where the input is padded along 
        # the time dimension
        e_mask = self.create_enc_mask(enc_input)

        # retrieve and scale embeddings 
        e_emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)

        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(e_emb + self.pos_emb[:enc_input.size(1), :], 
                              mask = e_mask)  

        # create the decoder mask for padding, which also prevents the decoder 
        # from looking into the future.
        d_mask = self.create_dec_mask(dec_input)

        # retrieve and scale embeddings
        d_emb = self.embed(dec_input.long()) * np.sqrt(self.embed.embedding_dim)

        # apply the (stacked) decoder transformer
        decoded = self.TF_dec(d_emb + self.pos_emb[:dec_input.size(1), :],
                              dec_mask = d_mask, enc_mask = e_mask, 
                              enc_input = encoded)

        return decoded, targs

    # Function used for a transformer with an encoder only without additional
    # context input e.g. for next word prediction. 
    def encoder_train(self, enc_input):
        # create the targets
        targs = torch.nn.functional.pad(enc_input[:,1:], [0,1]).long()
        
        # n.b. an encoder only setup will most likely use a decoder mask, which
        # also prevents peeking at future time steps. 
        e_mask = self.create_dec_mask(enc_input)
        
        # retrieve and scale embeddings 
        e_emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)
        
        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(e_emb + self.pos_emb[:enc_input.size(1), :], 
                              mask = e_mask)
        
        return encoded, targs 
   
    # Function to use a trained encoder-decoder for prediction. Encodes the
    # context and performs beamsearch over the decoder output. decoder input
    # is optional, as it is only used to validate the prediction not make it.
    def encoder_decoder_test(self, enc_input, dec_input = None, max_len = 64, 
                             beam_width = 1):
        if self.is_cuda == True:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        # create the targets if dec_input is given
        if not dec_input is None:
            targs = torch.nn.functional.pad(dec_input[:, 1:], [0, max_len - \
                                            dec_input[:, 1:].size()[-1]
                                                               ]
                                            ).long()
        else:
            targs = dtype([0])
        # create the encoder mask 
        e_mask = self.create_enc_mask(enc_input)
        # retrieve and scale embeddings
        emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)
        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(emb + self.pos_emb[:enc_input.size(1), :], 
                              mask = e_mask)  
        # set the decoder input to the beggining of sentence token    
        dec_input = enc_input[:, 0:1]
        # create the initial candidate sentence
        candidates = [[dec_input, 0]]
        # perform beam search
        for x in range(1, max_len + 1):
            candidates = self.beam_search(candidates, encoded, e_mask, 
                                          beam_width, dtype)
            
        # get the probability of the top prediction by passing it through the 
        # decoder
        d_mask = self.create_dec_mask(candidates[0][0][:, :-1])
        emb = self.embed(candidates[0][0].long()) * np.sqrt(self.embed.embedding_dim)
        decoded = self.TF_dec(emb[:, :-1, :] + self.pos_emb[:candidates[0][0].size(1), :], 
                              dec_mask = d_mask, enc_mask = e_mask, 
                              enc_input = encoded)
        
        top_pred = self.linear(decoded)
        return candidates, top_pred, targs   
    
    # beam search algorithm for finding translations
    def beam_search(self, candidates, encoded, e_mask, beam_width, dtype):
        new_candidates = []
        for input, score in candidates:
            d_mask = self.create_dec_mask(input)
            emb = self.embed(input.long()) * np.sqrt(self.embed.embedding_dim)
            decoded = self.TF_dec(emb + self.pos_emb[:input.size(1), :], 
                                  dec_mask = d_mask, enc_mask = e_mask, 
                                  enc_input = encoded)
            # pass the data through the prediction layer and a softmax
            pred = torch.nn.functional.softmax(self.linear(decoded), 
                                               dim = -1
                                               ).squeeze(0)
            # get the top k predictions for the next word, calculate the new
            # probability of the sentence and append to the list of new candidates
            for value, idx in zip(*pred[-1, :].cpu().data.topk(beam_width)):
                new_candidates.append([torch.cat([input, 
                                                 dtype([idx.item()]).unsqueeze(0)
                                                  ], 1
                                                 ), -np.log(value) + score
                                       ]
                                      )
        sorted(new_candidates, key=lambda s: s[1])
        return new_candidates[:beam_width]       
