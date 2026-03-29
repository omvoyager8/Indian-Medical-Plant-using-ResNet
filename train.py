import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loaders
from model import get_model
from utils import train, evaluate

DATA_DIR = r"C:\Users\ACER\Downloads\Indian Medicinal Leaves Image Datasets\Medicinal Leaf and Plant dataset"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, classes = get_data_loaders(DATA_DIR)

model = get_model(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")

import json

with open("classes.json", "w") as f:
    json.dump(classes, f)