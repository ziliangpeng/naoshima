import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.datasets import IMDB
# from torchtext.data import Field, LabelField, BucketIterator
# from torchtext.legacy.data import BucketIterator

import dataloader
from dataloader import VOCAB_SIZE, MAX_LENGTH

# Define the fields for preprocessing the data
# TEXT = Field(tokenize='spacy', lower=True)
# LABEL = LabelField(dtype=torch.float)

# Load the IMDB dataset
# train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary
# TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
# LABEL.build_vocab(train_data)


X_train, y_train, X_test, y_test = dataloader.load()

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        prediction = self.fc(hidden[-1])
        return prediction.squeeze(0)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the hyperparameters
# input_dim = len(TEXT.vocab)
input_dim = VOCAB_SIZE
embedding_dim = 128
hidden_dim = 128
output_dim = 1
batch_size = 256
num_epochs = 42
learning_rate = 1e-3

# Create the iterators
# train_iterator, test_iterator = BucketIterator.splits(
#     (train_data, test_data),
#     batch_size=batch_size,
#     device=device)

# Initialize the model and optimizer
model = LSTM(input_dim, embedding_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# Train the model
for epoch in range(num_epochs):
    for batch in train_iterator:
        text, label = batch.text.to(device), batch.label.to(device)
        optimizer.zero_grad()
        prediction = model(text).squeeze(1)
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_iterator:
        text, label = batch.text.to(device), batch.label.to(device)
        prediction = model(text).squeeze(1)
        rounded_preds = torch.round(torch.sigmoid(prediction))
        correct += (rounded_preds == label).sum().item()
        total += label.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.3f}')