
import torch
import torch
import torch.optim as optim
import torch.nn as nn
from preprocessing import encode, pad

def train_seq2seq(model, loader, vocab_size):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):

        total_loss = 0

        for x,y in loader:

            optimizer.zero_grad()

            output = model(x,y)

            loss = loss_fn(
                output.view(-1,vocab_size),
                y[:,1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print("Epoch",epoch,"Loss",total_loss)
        
        
        
def correct(sentence, model, word_to_idx):

    model.eval()

    encoded = encode(sentence, word_to_idx)
    padded = pad(encoded, word_to_idx["<PAD>"])

    x = torch.tensor(padded).unsqueeze(0)

    encoder_outputs,h,c = model.encoder(x)

    input_token = x[:,0].unsqueeze(1)

    result = []

    for _ in range(len(sentence)):

        out,h,c = model.decoder(
            input_token,h,c,encoder_outputs
        )

        pred = out.argmax(2)

        result.append(pred.item())

        input_token = pred

    return result