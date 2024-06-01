#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_utils.py    
@Contact :   hkustudy@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/9/18 1:07   Armor(hku)    1.0          None
'''
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  BertModel,BertConfig,RobertaConfig,RobertaModel

class Mdrop(nn.Module):
    def __init__(self):
        super(Mdrop,self).__init__()
        self.dropout_0 = nn.Dropout(p=0.0)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_3 = nn.Dropout(p=0.3)
    def forward(self,x):
        output_0 = self.dropout_0(x)
        output_1 = self.dropout_1(x)
        output_2 = self.dropout_2(x)
        output_3 = self.dropout_3(x)
        return [output_0,output_1,output_2,output_3]

class XFBert(nn.Module):
    def __init__(self, MODEL_NAME, intent_dim, dropout=None, n_ambda=0., enable_mdrop=False):
        super(XFBert, self).__init__()
        self.enable_mdrop = enable_mdrop
        self.model = BertModel.from_pretrained(MODEL_NAME)
        if n_ambda > 0.:
            print(n_ambda)
            for name, para in self.model.named_parameters():
                self.model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * n_ambda * torch.std(para)
        self.config = BertConfig.from_pretrained(MODEL_NAME)
        self.intent_num_labels = intent_dim

        self.lstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=self.config.hidden_size, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size=self.config.hidden_size*2, hidden_size=self.config.hidden_size, batch_first=True, bidirectional=True)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.config.hidden_size*2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(self.config.hidden_size*2, 64)
        if self.enable_mdrop:
            self.dropout = Mdrop()
        else:
            self.dropout = nn.Dropout(dropout if dropout is not None else self.config.hidden_dropout_prob)

        self.intent_classifier = nn.Linear(64, self.intent_num_labels)

    def forward(self, input_ids, attention_mask=None,use_last4hidden=False):
        outputs = self.model(input_ids, attention_mask=attention_mask,output_hidden_states=True)
        if use_last4hidden:
            hidden_state = torch.stack(outputs.hidden_states[-4:], dim=-1).mean(-1)
        else:
            hidden_state = outputs.last_hidden_state
        hidden_state, _ = self.lstm(hidden_state)
        hidden_state, _ = self.gru(hidden_state)

        M = self.tanh1(hidden_state)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = hidden_state*alpha

        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        # intent_logits = self.intent_classifier(out)

        # pooled_output_intent = hidden_state.mean(dim=1)
        # pooled_output_intent = self.dropout(out)
        # if self.enable_mdrop:
        #     pooled_output_intent = torch.mean(torch.stack(pooled_output_intent,dim=0),dim=0)
        intent_logits = self.intent_classifier(out)
        return intent_logits

class XFRoberta(nn.Module):
    def __init__(self, MODEL_NAME, intent_dim, dropout=None, n_ambda=0.):
        super(XFRoberta, self).__init__()

        self.model = RobertaModel.from_pretrained(MODEL_NAME)
        if n_ambda > 0.:
            print(n_ambda)
            for name, para in self.model.named_parameters():
                self.model.state_dict()[name][:] += (torch.rand(para.size()) - 0.5) * n_ambda * torch.std(para)
        self.config = RobertaConfig.from_pretrained(MODEL_NAME)
        self.intent_num_labels = intent_dim

        self.intent_dropout = nn.Dropout(dropout if dropout is not None else self.config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(self.config.hidden_size, self.intent_num_labels)

    def forward(self, input_ids, attention_mask=None,use_last4hidden=False):
        outputs = self.model(input_ids, attention_mask=attention_mask,output_hidden_states=True)
        if use_last4hidden:
            hidden_state = torch.stack(outputs.hidden_states[-4:], dim=-1).mean(-1)
        else:
            hidden_state = outputs.last_hidden_state

        pooled_output_intent = hidden_state.mean(dim=1)
        pooled_output_intent = self.intent_dropout(pooled_output_intent)
        intent_logits = self.intent_classifier(pooled_output_intent)
        return intent_logits



