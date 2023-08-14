import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch import nn

import dataloader

# Generate synthetic regression data
X, y = dataloader.generate_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert data to tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the model
model = nn.Sequential(nn.Linear(X_train.shape[1], 1))

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
epochs = 100
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(x_batch)

        # Calculate loss
        loss = loss_function(y_pred, y_batch)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model on the test set
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = loss_function(y_test_pred, y_test_tensor)
    print(f"Test loss: {test_loss.item()}")

# Print the learned weights and bias
weights, bias = model[0].weight.detach().numpy(), model[0].bias.detach().numpy()
print("Learned weight:", weights[0][0])
print("Learned bias:", bias[0])
