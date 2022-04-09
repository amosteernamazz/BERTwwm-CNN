# -*- coding: utf-8 -*-
"""
手动实现transformer.models.bert.BertModel函数 + CNN


"""
import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import torch.nn.functional as F


class bert_cnn_Config(nn.Module):
    def __init__(self):
        self.bert_path = "./chinese-bert-wwm-ext"
        self.config_path = "./chinese-bert-wwm-ext/config.json"

        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_labels = 2
        self.dropout_bertout = 0.2
        self.mytrainedmodel = "./result/bert_clf_model.bin"

        """
        current loss: 0.11688137799501419 	 current acc: 0.96875
        current loss: 0.10488582722638821 	 current acc: 0.9643967661691543
        current loss: 0.10978878578147909 	 current acc: 0.9628273067331671
        train loss:  0.11139155609225695 	 train acc: 0.9632994833422412
        valid loss:  0.13825790914283556 	 valid acc: 0.9526290857413549
        """

class bert_cnn(nn.Module):

    def __init__(self,config):
        super(bert_cnn, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config = config.config_path)
        self.dropout_bertout = nn.Dropout(config.dropout_bertout)
        self.num_labels = config.num_labels

        self.conv1 = nn.Sequential(         # input shape # [32, 1, 100, 768] [batch_size, channel, height, width]
            nn.Conv2d(
                in_channels=1,              # input height // input channel
                out_channels=64,            # n_filters    // output channel
                kernel_size=(11,768),       # filter size !!如果要调整这个卷积核大小，一定要调整padding第一项，是2n+1的关系
                stride=1,                   # filter movement/step
                padding=(5,0),                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape
            nn.ReLU(),                     # activation
            # nn.MaxPool2d(kernel_size = (100, 1))
        )

        self.classifier = nn.Linear(64, config.num_labels)
        self.Softmaxdim1 = nn.Softmax(dim=1)
        for param in self.bert.parameters():
            param.requires_grad = True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        bert_output = outputs[0] # [1]是pooled的结果 [0]是last_hidden_state ((batch_size, sequence_length, hidden_size))
            # [32,100,768]
        bert_output = self.dropout_bertout(bert_output)

        bert_output = bert_output.unsqueeze(1) # [32, 1, 100, 768]
        # print(bert_output.shape) # [32, 1, 100, 768]
        batch_size = bert_output.shape[0] # 32
        seq_len = bert_output.shape[2] # 100


        cnn_output = self.conv1(bert_output)

        cnn_output = F.max_pool2d(cnn_output, kernel_size=(seq_len, 1))
        # print(cnn_output.shape) # [32, 64, 1, 1]

        flatten = cnn_output.view(batch_size, -1)
        # print(flatten.shape) # [32, 64]

        logits = self.classifier(flatten)
        # print(logits)
        logits = self.Softmaxdim1(logits)
        # print(logits)
        # print(logits.shape) [64,2]
        loss = None
        # print(labels.shape)  [64,1]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss,logits


if __name__ == "__main__":
    model = bert_cnn(config=bert_cnn_Config())
    print(model)