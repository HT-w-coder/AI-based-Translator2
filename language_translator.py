import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define special tokens matching vocab indices
SOS_token = 0
EOS_token = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq).view(1, input_seq.size(0), -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq).view(1, input_seq.size(0), -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(1)
        target_len = target_seq.size(0)
        target_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden = self.encoder.init_hidden(batch_size)
        encoder_output, hidden = self.encoder(input_seq, hidden)

        decoder_input = torch.tensor([SOS_token] * batch_size, device=device)

        for t in range(target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target_seq[t] if teacher_force and t < target_len else top1

        return outputs

class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]

def train(model, optimizer, criterion, input_seq, target_seq):
    model.train()
    optimizer.zero_grad()

    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    output = model(input_seq, target_seq)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    target_seq = target_seq[1:].reshape(-1)

    loss = criterion(output, target_seq)
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    # Define example vocabularies (replace with your real vocab sets)
    input_vocab = {'<sos>': 0, '<eos>': 1, 'hello': 2, 'world': 3}
    output_vocab = {'<sos>': 0, '<eos>': 1, 'hola': 2, 'mundo': 3}

    # Example tokenized tensor pairs (sequence length x batch size)
    data_pairs = [(
        torch.tensor([input_vocab['<sos>'], input_vocab['hello'], input_vocab['world'], input_vocab['<eos>']], dtype=torch.long),
        torch.tensor([output_vocab['<sos>'], output_vocab['hola'], output_vocab['mundo'], output_vocab['<eos>']], dtype=torch.long)
    )]

    dataset = TranslationDataset(data_pairs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_size = len(input_vocab)
    output_size = len(output_vocab)
    hidden_size = 256
    learning_rate = 0.01
    num_epochs = 10
    teacher_forcing_ratio = 0.5

    encoder = Encoder(input_size, hidden_size).to(device)
    decoder = Decoder(hidden_size, output_size).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.view(-1, 1)  # (seq_len, batch_size)
            target_seq = target_seq.view(-1, 1)
            loss = train(model, optimizer, criterion, input_seq, target_seq)
            total_loss += loss
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'translation_model.pt')

if __name__ == "__main__":
    main()
