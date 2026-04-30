# models.py

import torch
import torch.nn as nn


class POSTagger(nn.Module):

    def __init__(self, vocab_size, tag_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,128)

        self.lstm = nn.LSTM(
            128,256,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(512, tag_size)

    def forward(self,x):

        x = self.embedding(x)
        out,_ = self.lstm(x)

        return self.fc(out)



class ErrorModel(nn.Module):

    def __init__(self,vocab_size,num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,128)

        self.lstm = nn.LSTM(
            128,256,
            batch_first=True,
            bidirectional=True
        )

        self.detector = nn.Linear(512,2)
        self.classifier = nn.Linear(512,num_classes)

    def forward(self,x):

        x = self.embedding(x)
        out,_ = self.lstm(x)

        return self.detector(out), self.classifier(out)



class Encoder(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,128)
        self.lstm = nn.LSTM(128,256,batch_first=True)

    def forward(self,x):

        x = self.embedding(x)
        out,(h,c) = self.lstm(x)

        return out,h,c


class Attention(nn.Module):

    def forward(self,hidden,encoder_outputs):

        scores = torch.bmm(
            encoder_outputs,
            hidden.unsqueeze(2)
        ).squeeze(2)

        weights = torch.softmax(scores,dim=1)

        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context


class Decoder(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,128)
        self.lstm = nn.LSTM(128+256,256,batch_first=True)
        self.fc = nn.Linear(256,vocab_size)
        self.attn = Attention()

    def forward(self,x,h,c,encoder_outputs):

        x = self.embedding(x)

        context = self.attn(h[-1],encoder_outputs)

        x = torch.cat([x,context.unsqueeze(1)],dim=2)

        out,(h,c) = self.lstm(x,(h,c))

        return self.fc(out),h,c


class Seq2Seq(nn.Module):

    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,y):

        encoder_outputs,h,c = self.encoder(x)

        outputs = []
        input_token = y[:,0].unsqueeze(1)

        for t in range(1,y.size(1)):

            out,h,c = self.decoder(
                input_token,h,c,encoder_outputs
            )

            outputs.append(out)

            input_token = y[:,t].unsqueeze(1)

        return torch.cat(outputs,dim=1) 