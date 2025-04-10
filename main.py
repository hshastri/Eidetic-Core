import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import customlayers
import logging
import numpy as np
import db
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(filename='benchmark.log', filemode='a', level=logging.DEBUG)
logging.info("Started")

# ----------------------
# Custom Dataset Loader
# ----------------------
class MNISTCSVLoader(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype('float32')
        self.images = self.images / 255.0  # normalize to 0-1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(self.labels[idx]).long()
        return image, label

# ----------------------
# Neural Network Model
# ----------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 36)

        task_b_cardinality = int(os.getenv("TASK_B_SUBSET_CARDINALITY"))
        num_quantiles = int(os.getenv("NUM_QUANTILES"))

        self.eidetic = customlayers.EideticLinearLayer(36, 36, 1.0, task_b_cardinality, 1)
        self.eideticIndexed = customlayers.EideticIndexedLinearLayer(36, 36, 1.0, task_b_cardinality, num_quantiles, 2)
        self.indexed = customlayers.IndexedLinearLayer(36, 36, num_quantiles)

        self.indexed_layers = {"1": self.eideticIndexed, "2": self.indexed}
        self.eidetic_layers = {"1": self.eidetic, "2": self.eideticIndexed}

    def forward(self, x, calculate_distribution, get_indices, use_db):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        [x, idxs] = self.eidetic(x, calculate_distribution[0], get_indices[0], use_db[0])
        [x, idxs] = self.eideticIndexed(x, idxs, calculate_distribution[1], get_indices[1], use_db[1])
        x = self.indexed(x, idxs)
        return F.log_softmax(x, dim=1)

    def unfreeze_eidetic_layers(self):
        self.indexed.unfreeze_params()

    def use_indices(self, val, table_number):
        self.indexed_layers[table_number].set_use_indices(val)

    def calculate_n_quantiles(self, num_quantiles, use_db, table_number):
        self.eidetic_layers[table_number].calculate_n_quantiles(num_quantiles, use_db)

    def index_layers(self, num_quantiles, table_number):
        self.indexed_layers[table_number].build_index(num_quantiles)

# ----------------------
# Training function
# ----------------------
def train(args, model, device, train_loader, optimizer, epoch, calculate_distribution, use_db, get_indices, val_to_add_to_target):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target + val_to_add_to_target
        optimizer.zero_grad()
        output = model(data, calculate_distribution, get_indices, use_db)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break

# ----------------------
# Test function
# ----------------------
def test(model, device, test_loader, calculate_distribution, use_db, get_indices, val_to_add_to_target, test_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target + val_to_add_to_target
            output = model(data, calculate_distribution, get_indices, use_db)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\n{test_name} Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    logging.info(f'{test_name} Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# ----------------------
# Utility functions
# ----------------------
def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_eidetic_layers(model, num_quantiles, layer_number):
    for i, param in enumerate(model.indexed_layers[layer_number].parameters()):
        if num_quantiles == 1 and i == 0:
            param.requires_grad = True
        if i >= 2:
            param.requires_grad = True

# ----------------------
# Main training workflow
# ----------------------
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-mps', action='store_true', default=False)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False, 'num_workers': 0}

    # Use your local MNIST CSV files
    dataset1 = MNISTCSVLoader('mnist_train.csv')
    dataset2 = MNISTCSVLoader('mnist_test.csv')

    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    # Use subset for Task A and Task B
    subset_indices = np.arange(1, int(os.getenv("TASK_B_SUBSET_CARDINALITY")))
    degradation_subset = DataLoader(torch.utils.data.Subset(dataset1, subset_indices), batch_size=1, shuffle=True)

    subset_indices = np.arange(1, int(os.getenv("TASK_A_SUBSET_CARDINALITY")))
    train_subset = DataLoader(torch.utils.data.Subset(dataset1, subset_indices), batch_size=1, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    round_ = 1
    num_quantiles = int(os.getenv("NUM_QUANTILES"))
    use_indices = num_quantiles != 1
    use_db = os.getenv("USE_DB") == "True"

    if use_db:
        db.database.recreate_tables(num_quantiles, 1)
        db.database.recreate_tables(num_quantiles, 2)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if round_ == 1:
            logging.info("Layer 1")
            train(args, model, device, train_subset, optimizer, epoch, [False, False], [False, False], [False, False], 26)

            logging.info("Layer 2 Pre-Training")
            test(model, device, train_subset, [False, False], [False, False], [False, False], 26, "Layer 2, Task A Pre-Training")
            test(model, device, degradation_subset, [False, False], [False, False], [False, False], 0, "Layer 2, Task B Pre-Training")

            test(model, device, degradation_subset, [False, use_indices], [False, use_db], [False, False], 0, "Layer 2, Task B Storing Activations")

            if use_indices:
                print("Layer 2, Calculating Quantiles...")
                model.calculate_n_quantiles(num_quantiles, use_db, "2")
                print("Layer 2, Indexing Layers...")
                model.index_layers(num_quantiles, "2")
                model.use_indices(True, "2")

            print("Layer 2, Freezing non-eidetic layers...")
            freeze_layers(model)
            unfreeze_eidetic_layers(model, num_quantiles, "2")

            print("Layer 2, Training model with eidetic parameters...")
            train(args, model, device, degradation_subset, optimizer, epoch, [False, False], [False, False], [False, use_indices], 0)

            test(model, device, degradation_subset, [False, False], [False, False], [False, use_indices], 0, "Layer 2, Task B")
            test(model, device, train_subset, [False, False], [False, False], [False, use_indices], 26, "Layer 2, Task A")

            print("Epoch finished...")

        round_ += 1
        scheduler.step()

    logging.info(f"--- {time.time() - start_time} seconds ---")

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()
