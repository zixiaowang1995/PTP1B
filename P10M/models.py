import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import init
import math
batch_size = 36
class Global_Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
    def forward(self, x):
        attention_weights = torch.softmax(x, dim=-1)
        return x * attention_weights
class SMILESFeatureExtractor(nn.Module):
    def __init__(self, model, tokenizer, output_dim=384, device='cuda'):
        super(SMILESFeatureExtractor, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.linear = nn.Linear(model.config.vocab_size, output_dim)
        self.fc1 = nn.Linear(output_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
        self.pool = Global_Attention(output_dim)  # 使用参数而非硬编码
        self.to(device)

    def forward(self, smiles_list):
        encoded = self.tokenizer(
            smiles_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded['attention_mask'].to(self.model.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # 简化CLS token提取
        cls_token_embedding = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'last_hidden_state') else outputs[0][:, 0, :]
        
        transformed_output = self.linear(cls_token_embedding)
        transformed_output = self.pool(transformed_output)  # batch_size不需要
        out = self.fc1(transformed_output)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sig(out)
        # 确保输出维度正确，避免squeeze后维度不匹配
        return out.view(-1, 1)


class Model_TGCN(nn.Module):
    def __init__(self, p_dropout):
        super(Model_TGCN, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 150)
        self.bn3 = nn.BatchNorm1d(150)
        self.fc1 = nn.Linear(150, 1)
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x_l):
        x_l = x_l.to("cuda")
        x_l = self.linear1(self.dropout(x_l))
        x_l = self.bn1(x_l)
        x_l = F.relu(x_l)
        x_l = self.linear2(self.dropout(x_l))
        x_l = self.bn2(x_l)
        x_l = F.relu(x_l)
        x_l = self.linear3(self.dropout(x_l))
        x_l = self.bn3(x_l)
        x_l = F.relu(x_l)
        out = self.fc1(x_l)
        out = self.sig(out)
        return out