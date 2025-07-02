import torch.nn as nn
import torch.optim as optim
import time
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 64
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        h = 32
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, h),
            nn.ReLU(),
            nn.Linear(h, 10)
        )

    def forward(self, x):
        return self.model(x)
#
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),   # 1x28x28 → 4x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 4x14x14
            nn.Conv2d(4, 8, kernel_size=3, padding=1),   # 8x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 8x7x7
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, 10)                     # Direct to output
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1x28x28 → 8x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 8x14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 16x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 16x7x7
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)


class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 → 32x28x28
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), # 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 32x14x14
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 64x7x7
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(loader)  # average over batches
    return correct / total, avg_loss

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(model_class, name):
    model = model_class().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print(f"\n{name} ({count_params(model)} parameters)")
    start = time.time()
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, criterion)
    duration = time.time() - start

    acc, loss = test(model, test_loader)
    print(f"Test Accuracy: {acc * 100:.2f}% | Loss: {loss:.2f} | Learning time: {duration:.1f}s")

# run_experiment(MLP, "MLP")
# run_experiment(CNN, "CNN")
# run_experiment(TinyCNN, "Tiny CNN")
# run_experiment(StrongCNN, "Strong CNN")
