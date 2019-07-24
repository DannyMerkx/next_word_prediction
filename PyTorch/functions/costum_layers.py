#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:22:38 2018

@author: danny
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

############################# Costum implementation of Recurrent Highway Networks #####################
# Rather slow, not much better than regular gru/lstms 

# implementation of recurrent highway networks using existing PyTorch layers
class RHN(nn.Module):
    def __init__(self, in_size, hidden_size, n_steps, batch_size):
        super(RHN, self).__init__()
        self.n_steps = n_steps
        self.initial_state = torch.autograd.Variable(torch.rand(1, batch_size, hidden_size)).gpu()
        # create 3 linear layers serving as the hidden, transform and carry gate,
        # one each for each microstep. 
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
            for layers, lists in zip(self.create_microstep(hidden_size), layer_list):
                lists.append(layers)
                
    # initialise linear layers for the microsteps
    def create_microstep(self, n_nodes):       
        H = nn.Linear(n_nodes,n_nodes)
        T = nn.Linear(n_nodes,n_nodes)
        C = nn.Linear(n_nodes,n_nodes)
        return(H,T,C)
    # the input is only used in the first microstep. For the first step the history is a random initial state.
    def calc_htc(self, x, hx, step, non_linearity):
        if step == 0:
            return non_linearity(((step+1)//1 * self.init_h(x)) + self.H[step](hx))
        else:
            return non_linearity(self.H[step](hx))
                                 
    def perform_microstep(self, x, hx, step):
        output = self.calc_htc(x, hx, step, self.tan) * self.calc_htc(x, hx, step, self.sig) + hx * (1 - self.calc_htc(x, hx, step,self.sig))
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
        
# class for making multi headed attenders. 
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
            # save the attention matrices to be able to use them in a loss function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)
    
# attention layer for audio encoders
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
        x = self.alpha * input
        # return the resulting embedding
        return x   
    
################################ Transformer Layers ###########################
        
# single encoder transformer cell with h attention heads fully connected layer block and residual connections
class transformer_encoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_encoder_cell, self).__init__()
        # set the input size for the attention heads
        assert in_size % h == 0
        # create the attention layers
        self.att_heads = transformer_att(in_size, h)
        # create the linear layer block
        self.ff = transformer_ff(in_size, fc_size)
        # the layernorm and dropout functions      
        self.norm_att = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, mask = None):
        # apply the attention block to the input
        att = self.att_heads(input, input, input, mask)
        # apply the residual connection to the input and apply layer normalisation
        norm_att = self.norm_att(self.dropout(att) + input)        
        # apply the linear layer block
        lin = self.ff(norm_att)
        # apply the residual connection to the att block and apply layer normalisation
        out = self.norm_ff(self.dropout(lin) + norm_att)
        return out

# decoder cell, has an extra attention block which recieves encoder output as its input
class transformer_decoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_decoder_cell, self).__init__()
        # set the input size for the attention heads
        assert in_size % h == 0
        # create the attention layers
        self.att_one = transformer_att(in_size, h)
        self.att_two = transformer_att(in_size, h)
        # linear layer block
        self.ff = transformer_ff(in_size, fc_size)
        # the layernorm and dropout functions     
        self.norm_att_one = nn.LayerNorm(in_size)
        self.norm_att_two = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, dec_mask = None, enc_mask = None, enc_input = None):
        # apply the first attention block to the input
        att = self.att_one(input, input, input, dec_mask)
        # apply the residual connection to the input and apply layer normalisation
        norm_att = self.norm_att_one(self.dropout(att) + input)
        #norm_att = att + input
        # apply the second attention block to the input
        if enc_input is None:
            # in case you want to use the decoder architecture without enc-dec setup, use 
            # the dec input queries and keys
            enc_input = norm_att
            enc_mask = dec_mask
        att_2 = self.att_two(norm_att, enc_input, enc_input, enc_mask)

        # apply the residual connection and layer normalisation
        norm_att_2 = self.norm_att_two(self.dropout(att_2) + norm_att)
        # apply the linear layer block
        lin = self.ff(norm_att_2)
        # apply the residual connection to the att block and apply layer normalisation
        out = self.norm_ff(self.dropout(lin) + norm_att_2)
        #out = lin + norm_att_2
        return out
    
# the linear layer block of the transformer
class transformer_ff(nn.Module):
    def __init__(self, in_size, fc_size):
        super(transformer_ff, self).__init__()
        # the linear layers of feed forward block
        self.ff_1 = nn.Linear(in_size, fc_size)
        self.ff_2 = nn.Linear(fc_size, in_size)
        # rectified linear unit activation function
        self.relu = nn.ReLU()
    def forward(self, input):
        output = self.ff_2(self.relu(self.ff_1(input)))
        return output
    
# transformer attention head with insize equal to transformer input size and hidden size equal to
# insize/h 
class transformer_att(nn.Module):
    def __init__(self, in_size, h):
        super(transformer_att, self).__init__()
        self.att_size = int(in_size/h)
        # create the Q, K and V parts of the attention head
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
        # apply the linear transform to the query, key and value and reshape the result into
        # h attention heads
        Q = self.Q(q).view(batch_size, -1, self.h, self.att_size).transpose(1,2)
        K = self.K(k).view(batch_size, -1, self.h, self.att_size).transpose(1,2)
        V = self.V(v).view(batch_size, -1, self.h, self.att_size).transpose(1,2)
        # multiply and scale q and v to get the attention scores
        self.alpha = torch.matmul(Q,K.transpose(-2,-1))/scale
        # apply mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1)
            self.alpha = self.alpha.masked_fill(mask == 0, -1e9)
        # apply softmax to the attention scores
        self.alpha = self.softmax(self.alpha)
        # apply the att scores to the value v
        att_applied = torch.matmul(self.dropout(self.alpha), V)    
        # reshape the attention heads and finally pass through a fully connected layer
        att = att_applied.transpose(1, 2).reshape(batch_size, -1, self.att_size * self.h)
        output = self.fc(att)   
        return output

# the transformer encoder layer capable of stacking multiple transformer cells. 
class transformer_encoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_encoder, self).__init__()
        # create one or more multi-head attention layers
        self.transformers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.transformers.append(transformer_encoder_cell(in_size, fc_size, h))
    def forward(self, input, mask = None):
        # apply the (stacked) transformer
        for tf in self.transformers:
            input = tf(self.dropout(input), mask)
        return(input)

# the transformer decoder layer capable of stacking multiple transformer cells. 
class transformer_decoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_decoder, self).__init__()
        # create one or more multi-head attention layers
        self.transformers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.transformers.append(transformer_decoder_cell(in_size, fc_size, h))
    def forward(self, input, dec_mask = None, enc_mask = None, enc_input = None):
        # apply the (stacked) transformer
        for tf in self.transformers:
            input = tf(self.dropout(input), dec_mask, enc_mask, enc_input)
        return(input)
