"""
Author: Wenyu Ouyang
Date: 2024-04-17 12:32:26
LastEditTime: 2024-04-17 12:33:34
LastEditors: Xinzhuo Wu
Description: 
FilePath: /torchhydro/torchhydro/models/seq2seq.py
Copyright (c) 2021-2024 Wenyu Ouyang. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SMEncoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        input_channels=None,
        mode="single",
        num_layers=1,
    ):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.mode = mode
        if self.mode == "single":
            self.sm_encoder = SMEncoder(input_channels, output_channels=1)

    def forward(self, x):
        if self.mode == "single":
            src1, src2 = x
            sm_encoded = self.sm_encoder(src2)
            src_combined = torch.cat((src1, sm_encoded), dim=2)
            outputs, (hidden, cell) = self.lstm(src_combined)
        else:
            outputs, (hidden, cell) = self.lstm(x)
        outputs = self.fc(outputs)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return F.softmax(energy.squeeze(2), dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            hidden_dim + output_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.attention = Attention(hidden_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        attention_weights = self.attention(encoder_outputs, hidden.squeeze(0))
        weighted_context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((input, weighted_context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class GeneralSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        forecast_length,
        cnn_size=None,
        model_mode="single",
    ):
        super(GeneralSeq2Seq, self).__init__()
        self.mode = model_mode
        self.trg_len = forecast_length
        self.encoder1 = Encoder(
            input_dim=input_size,
            hidden_dim=hidden_size,
            input_channels=cnn_size,
            mode=self.mode,
        )
        self.decoder1 = Decoder(output_dim=output_size, hidden_dim=hidden_size)
        self.encoder2 = (
            Encoder(input_dim=1, hidden_dim=hidden_size, mode=self.mode)
            if self.mode != "single"
            else None
        )
        self.decoder2 = (
            Decoder(output_dim=1, hidden_dim=hidden_size)
            if self.mode != "single"
            else None
        )

    def forward(self, *src):
        if self.mode != "single":
            return self.process_dual(src[:2], self.trg_len, src[2])
        return self.process_single(src[:2], self.trg_len, src[2])

    def process_single(self, src, trg_len, trg_start_token):
        encoder_outputs, hidden, cell = self.encoder1(src)
        current_input = trg_start_token

        outputs = []
        for _ in range(trg_len):
            output, hidden, cell = self.decoder1(
                current_input, hidden, cell, encoder_outputs
            )
            outputs.append(output)
            current_input = output.unsqueeze(1)
        outputs = torch.stack(outputs, dim=0)
        return outputs.permute(1, 0, 2)

    def process_dual(self, src, trg_len, trg_start_token):
        src1, src2 = src
        encoder_outputs1, hidden1, cell1 = self.encoder1(src1)
        current_input = trg_start_token

        outputs1 = []
        for _ in range(trg_len):
            output1, hidden1, cell1 = self.decoder1(
                current_input, hidden1, cell1, encoder_outputs1
            )
            outputs1.append(output1)
            current_input = output1.unsqueeze(1)
        outputs1 = torch.stack(outputs1, dim=0)

        encoder_outputs2, hidden2, cell2 = self.encoder2(src2)
        current_input = trg_start_token

        outputs2 = []
        for _ in range(trg_len):
            output2, hidden2, cell2 = self.decoder2(
                current_input, hidden2, cell2, encoder_outputs2
            )
            outputs2.append(output2)
            current_input = output2.unsqueeze(1)
        outputs2 = torch.stack(outputs2, dim=0)
        runoff_coefficients = torch.sigmoid(outputs2)
        final_outputs = outputs1 * runoff_coefficients
        return final_outputs.permute(1, 0, 2)
