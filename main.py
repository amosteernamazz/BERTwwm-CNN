# # import torch
# # from transformers import BertConfig, BertTokenizer, BertModel,AutoModelForMaskedLM,AutoTokenizer
# #
# # model_name = "/home/workspace/ranqiguolu/bertproject/home/workspace/ranqiguolu/bertproject/chinese-bert-wwm-ext"
# # tokenizer = BertTokenizer.from_pretrained(model_name)   # 导入词典导入分词器
# # model_config = BertConfig.from_pretrained(model_name)   # 导入配置文件
# # model = AutoModelForMaskedLM.from_pretrained(model_name)
# # tokenizer.encode("我爱你")
# # # Out :[101, 2769, 4263, 872, 102]
# # print(tokenizer.encode("你好"))
# # # encode_plus返回所有编码信息
# # # input_id = tokenizer.encode_plus("我爱你", "你也爱我")
# # # Out :{'input_ids': [101, 2769, 4263, 872, 102, 872, 738, 4263, 2769, 102],
# # #      'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
# # #      'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
# # # # 添加batch维度并转化为tensor
# # # input_ids = torch.tensor(input_id['input_ids'])
# # # token_type_ids = torch.tensor(input_id['token_type_ids'])
# # # attention_mask_ids=torch.tensor(input_id['attention_mask'])
# # #
# # # # 将模型转化为eval模式
# # # model.eval()
# # # # print(model)
# # # # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
# # # # print(torch.cuda.is_available())
# # # # device = 'cuda:1'
# # # # tokens_tensor = input_ids.to(device).unsqueeze(0)
# # # # segments_tensors = token_type_ids.to(device).unsqueeze(0)
# # # # attention_mask_ids_tensors = attention_mask_ids.to(device).unsqueeze(0)
# # # # model.to(device)
# # # #
# # # # # 进行编码
# # # # with torch.no_grad():
# # # #     # See the models docstrings for the detail of the inputs
# # # #     outputs = model(tokens_tensor, segments_tensors, attention_mask_ids_tensors)
# # # #     # Transformers models always output tuples.
# # # #     # See the models docstrings for the detail of all the outputs
# # # #     # In our case, the first element is the hidden state of the last layer of the Bert model
# # # #     encoded_layers = outputs
# # # #
# # # # s_a, s_b = "李白拿了个锤子", "锤子？"
# # # # # 分词是tokenizer.tokenize, 分词并转化为id是tokenier.encode
# # # # # 简单调用一下, 不作任何处理经过transformer
# # # # input_id = tokenizer.encode(s_a)
# # # # # input_id.to(device)
# # # # print(input_id)
# # # # input_id = torch.tensor([input_id])  # 输入数据是tensor且batch形式的
# # # # input_id.to(device)
# # #
# # #
# # # # print(input_id)
# # # # 得到最终的编码结果encoded_layers
#
# # -*- coding: utf-8 -*-
#
# import random
#
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import numpy as np
#
# import transformers
# import argparse
#
# from models.bert_CNN import bert_cnn, bert_cnn_Config
# from models.bert_lr import bert_lr, bert_lr_Config
# from models.bert_lr_last4layer import bert_lr_last4layer, bert_lr_last4layer_Config
#
# from tools.config import Config
# from transformers import AdamW
# from transformers import BertConfig, BertForSequenceClassification
# from tools.utils import SentimentDataset, convert_text_to_ids, seq_padding
#
