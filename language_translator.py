import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import io  # For capturing print output if needed

# Define constants and dummies (these would typically be loaded from data preprocessing)
SOS_token = 0  # Start-of-sequence token
EOS_token = 1  # End-of-sequence token (added for inference)
PAD_token = 2  # Padding token (added for completeness)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dummy vocabularies (in a real scenario, these would be built from your training data)
input_vocab_size = 10000  # Example size for input vocabulary
output_vocab_size = 10000  # Example size for output vocabulary
input_vocab = {f"word_{i}": i for i in range(3, input_vocab_size)}  # Start from 3 to leave room for special tokens
input_vocab['<PAD>'] = PAD_token
input_vocab['<SOS>'] = SOS_token
input_vocab['<EOS>'] = EOS_token
output_vocab = {f"word_{i}": i for i in range(3, output_vocab_size)}
output_vocab['<PAD>'] = PAD_token
output_vocab['<SOS>'] = SOS_token
output_vocab['<EOS>'] = EOS_token

# Reverse vocab for decoding output
input_index_to_word = {i: word for word, i in input_vocab.items()}
output_index_to_word = {i: word for word, i in output_vocab.items()}

# Dummy data: list of (input_seq, target_seq) where each is a tensor of shape (seq_len,)
# For simplicity, assume all sequences are length 10, batch_size=32 will pad if needed, but here fixed length
seq_len = 10
num_samples = 1000  # Enough for a few batches
data = []
for _ in range(num_samples):
    input_seq = torch.randint(3, input_vocab_size, (seq_len,))  # Random input sequence (avoid special tokens)
    target_seq = torch.randint(3, output_vocab_size, (seq_len,))  # Random target sequence
    data.append((input_seq, target_seq))

# Define the Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        # input_seq: (batch_size, src_len)
        embedded = self.embedding(input_seq)  # (batch_size, src_len, hidden_size)
        embedded = embedded.transpose(0, 1)  # (src_len, batch_size, hidden_size) for GRU
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

# Define the Decoder model
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        # input_seq: (batch_size,) for single timestep
        embedded = self.embedding(input_seq)  # (batch_size, hidden_size)
        embedded = embedded.unsqueeze(0)  # (1, batch_size, hidden_size)
        output, hidden = self.gru(embedded, hidden)  # output: (1, batch_size, hidden_size)
        output = self.softmax(self.out(output[0]))  # output[0]: (batch_size, hidden_size) -> (batch_size, output_size)
        return output, hidden

# Define the Seq2Seq model that combines the Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        encoder_output, hidden = self.encoder(input_seq, torch.zeros(1, batch_size, self.encoder.hidden_size).to(self.device))

        decoder_input = torch.full((batch_size,), SOS_token, dtype=torch.long).to(self.device)
        
        for t in range(1, target_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            decoder_input = target_seq[:, t] if teacher_force else top1

        return outputs

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Define the training function
def train(model, optimizer, criterion, input_seq, target_seq):
    optimizer.zero_grad()

    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    output = model(input_seq, target_seq)
    output = output[1:].view(-1, output.shape[-1])
    target_seq = target_seq[:, 1:].reshape(-1)

    loss = criterion(output, target_seq)
    loss.backward()

    optimizer.step()

    return loss.item()

# Define the main training function (modified for Streamlit progress)
@st.cache_data  # Cache to avoid retraining on every rerun
def train_model(_input_size, _output_size, _hidden_size, _learning_rate, _batch_size, _num_epochs, _teacher_forcing_ratio):
    # Prepare the dataset
    dataset = TranslationDataset(data)
    dataloader = DataLoader(dataset, batch_size=_batch_size, shuffle=True)

    # Initialize the models
    encoder = Encoder(_input_size, _hidden_size)
    decoder = Decoder(_hidden_size, _output_size)
    model = Seq2Seq(encoder, decoder, device)

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=_learning_rate)

    # Training loop with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_loss_epoch = 0

    for epoch in range(_num_epochs):
        epoch_loss = 0
        for i, (input_seq, target_seq) in enumerate(dataloader):
            loss = train(model, optimizer, criterion, input_seq, target_seq)
            epoch_loss += loss
            if (i + 1) % 10 == 0:
                status_text.text(f"Epoch [{epoch+1}/{_num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        progress_bar.progress((epoch + 1) / _num_epochs)
        status_text.text(f"Epoch [{epoch+1}/{_num_epochs}] Average Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'translation_model.pt')
    return model

# Simple dummy tokenizer (for demo; in real app, use proper tokenizer like from torchtext or Hugging Face)
def tokenize_input(text, vocab, max_len=10):
    # Dummy: split into words, map to vocab indices (if not in vocab, use random or PAD)
    words = text.lower().split()[:max_len-2]  # Leave room for SOS/EOS
    indices = [SOS_token]  # Start with SOS
    for word in words:
        if word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(torch.randint(3, len(vocab), (1,)).item())  # Dummy random token
    indices.append(EOS_token)  # End with EOS
    # Pad to max_len
    while len(indices) < max_len:
        indices.append(PAD_token)
    return torch.tensor([indices], dtype=torch.long)  # Batch size 1

# Inference function: Translate input sequence
def translate(model, input_tensor, max_len=10):
    model.eval()
    with torch.no_grad():
        # Encode
        encoder_output, hidden = model.encoder(input_tensor, torch.zeros(1, 1, model.encoder.hidden_size).to(device))
        
        # Decode
        decoder_input = torch.tensor([[SOS_token]], device=device)
        outputs = []
        for _ in range(max_len):
            output, hidden = model.decoder(decoder_input, hidden)
            top1 = output.max(1)[1]
            if top1.item() == EOS_token:
                break
            outputs.append(top1.item())
            decoder_input = top1.unsqueeze(0)
        
        # Convert to words
        translation = [output_index_to_word.get(idx, '<UNK>') for idx in outputs]
        return ' '.join(translation)

# Streamlit App
st.title("🗣️ Language Translator")
st.write("This is a demo Seq2Seq model for machine translation using GRU. It uses dummy random data, so translations are not meaningful—use for demonstration only!")

# Sidebar for hyperparameters (optional, for tweaking)
st.sidebar.header("Training Hyperparameters")
input_size = st.sidebar.number_input("Input Vocab Size", value=input_vocab_size, min_value=1000)
output_size = st.sidebar.number_input("Output Vocab Size", value=output_vocab_size, min_value=1000)
hidden_size = st.sidebar.number_input("Hidden Size", value=256, min_value=64)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, min_value=0.001)
batch_size = st.sidebar.number_input("Batch Size", value=32, min_value=1)
num_epochs = st.sidebar.number_input("Num Epochs", value=3, min_value=1)  # Reduced for demo speed
teacher_forcing_ratio = st.sidebar.slider("Teacher Forcing Ratio", 0.0, 1.0, 0.5)

# Train button
if st.button("🚀 Train the Model"):
    with st.spinner("Training the model... This may take a few minutes."):
        model = train_model(input_size, output_size, hidden_size, learning_rate, batch_size, num_epochs, teacher_forcing_ratio)
        st.session_state.model = model
        st.session_state.trained = True
    st.success("✅ Model trained and saved! Now you can translate text.")

# Translation section (only if trained)
if 'trained' in st.session_state and st.session_state.trained:
    st.header("Translate Text")
    input_text = st.text_input("Enter source text (e.g., 'hello world'):", "hello world")
    
    if st.button("Translate"):
        if 'model' in st.session_state:
            input_tensor = tokenize_input(input_text, input_vocab, seq_len)
            input_tensor = input_tensor.to(device)
            translation = translate(st.session_state.model, input_tensor, seq_len)
            st.write(f"**Input:** {input_text}")
            st.write(f"**Translation:** {translation}")
        else:
            st.error("Model not loaded. Please train first.")

    # Option to load saved model if needed
    if st.button("Reload Saved Model"):
        try:
            encoder = Encoder(input_size, hidden_size)
            decoder = Decoder(hidden_size, output_size)
            model = Seq2Seq(encoder, decoder, device)
            model.load_state_dict(torch.load('translation_model.pt', map_location=device))
            st.session_state.model = model
            st.success("Model reloaded!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

else:
    st.info("👆 Click 'Train the Model' to start. Training uses dummy data for demo purposes.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**Notes:**\n- This is a basic GRU-based Seq2Seq model.\n- For real translation, use proper datasets (e.g., WMT) and tokenizers.\n- Dummy data means outputs are random—model doesn't learn meaningful mappings.")
