import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import customlayers
import logging 
import numpy as np
import db

from dotenv import load_dotenv
load_dotenv()

import os
import time

logging.basicConfig(filename='benchmark.log', filemode='a', level=logging.DEBUG)
logging.info("Started")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 36)
        self.eidetic= customlayers.EideticLinearLayer(36, 36, 1.0, int(os.getenv("TASK_B_SUBSET_CARDINALITY")), 1)
        self.eideticIndexed= customlayers.EideticIndexedLinearLayer(36, 36, 1.0, int(os.getenv("TASK_B_SUBSET_CARDINALITY")), int(os.getenv("NUM_QUANTILES")), 2)
        self.indexed= customlayers.IndexedLinearLayer(36, 36, int(os.getenv("NUM_QUANTILES")))
        self.indexed_layers = {}
        self.indexed_layers["1"] = self.eideticIndexed
        self.indexed_layers["2"] = self.indexed

        self.eidetic_layers = {}
        self.eidetic_layers["1"] = self.eidetic
        self.eidetic_layers["2"] = self.eideticIndexed

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
        
        output = F.log_softmax(x, dim=1)
        return output

    def unfreeze_eidetic_layers(self):
        self.indexed.unfreeze_params()

    def use_indices(self, val, table_number):
        self.indexed_layers[table_number].set_use_indices(val)

    def calculate_n_quantiles(self, num_quantiles, use_db, table_number):
      self.eidetic_layers[table_number].calculate_n_quantiles(num_quantiles, use_db)

    def index_layers(self, num_quantiles, table_number):
        # self.eidetic.build_index(num_quantiles)
        self.indexed_layers[table_number].build_index(num_quantiles)

        

def train(args, model, device, train_loader, optimizer, epoch, calculate_distribution, use_db, get_indices, val_to_add_to_target):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target + val_to_add_to_target
        optimizer.zero_grad()
        output = model(data, calculate_distribution, get_indices, use_db)
        loss = F.nll_loss(output, target)
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, calculate_distribution, use_db, get_indices, val_to_add_to_target, test_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target + val_to_add_to_target
            output = model(data, calculate_distribution, get_indices, use_db)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logging.info(test_name + 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

def freeze_eidetic_layers(model):
    model.indexed.freeze_params()

def print_trainable_params(model):
    for param in model.parameters():
        if param.requires_grad == True:
            print(param)

def unfreeze_eidetic_layers(model, num_quantiles, layer_number):
    
    i = 0
    for param in model.indexed_layers[layer_number].parameters():
        if num_quantiles == 1 and i == 0:
            param.requires_grad = True
        if i >= 2:
            param.requires_grad = True
        i = i + 1

    
        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()


    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    #cpu or cuda
    device = os.getenv("DEVICE")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    use_cude = True
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    dataset3 = datasets.EMNIST('../data', train=True,
                       transform=transform, split="letters", download=True)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    extension_train_loader = torch.utils.data.DataLoader(dataset3,**train_kwargs)

    subset_indices = np.arange(1,int(os.getenv("TASK_B_SUBSET_CARDINALITY"))) # select your indices here as a list

    subset = torch.utils.data.Subset(extension_train_loader.dataset, subset_indices)
    degradation_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=True)
    #test_subset_a
    #test_subset_b

    
    subset_indices = np.arange(1,int(os.getenv("TASK_A_SUBSET_CARDINALITY"))) # select your indices here as a list

    subset = torch.utils.data.Subset(train_loader.dataset, subset_indices)
    train_subset = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=True)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    round_ = 1
    num_quantiles = int(os.getenv("NUM_QUANTILES"))

    use_indices = True

    if num_quantiles == 1:
        use_indices = False

    use_db = False
    

    if os.getenv("USE_DB") == "True":
        use_db = True
        db.database.recreate_tables(num_quantiles, 1)
        db.database.recreate_tables(num_quantiles, 2)
        
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        if round_ == 1:

            logging.info("\n\n\nLayer 1")
            train(args, model, device, train_subset, optimizer, epoch, [False, False], [False, False], [False, False], 26)
  
            logging.info("\n\n\nLayer 2")
            test(model, device, train_subset, [False, False], [False, False], [False, False], 26, "Layer 2, Task A Pre-Training ")
            test(model, device, degradation_subset, [False, False], [False, False], [False, False], 0, "Layer 2, Task B Pre-Training ")
            #Storing Activations
            test(model, device, degradation_subset, [False, use_indices], [False, use_db], [False, False], 0, "Layer 2, Task B Storing Activations ")

            if use_indices == True:
                print("Layer 2, Calculating Quantiles...")
                model.calculate_n_quantiles(num_quantiles, use_db, "2")
                print("Layer 2, Indexing Layers...")
                model.index_layers(num_quantiles, "2")
                model.use_indices(True, "2")
            print("Layer 2, Freezing non eidetic layers...")
            freeze_layers(model)
            unfreeze_eidetic_layers(model, num_quantiles, "2")
            print("Layer 2, Training model with eidetic parameters...")
            
            train(args, model, device, degradation_subset, optimizer, epoch, [False, False], [False, False], [False, use_indices], 0)
            
            test(model, device, degradation_subset, [False, False], [False, False], [False, use_indices], 0, "Layer 2, Task B")
            test(model, device, train_subset, [False, False], [False, False], [False, use_indices], 26, "Layer 2, Task A")
            print("Epoch finished...")
        round_ = round_ + 1
        scheduler.step()
    logging.info("--- %s seconds ---" % (time.time() - start_time))
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
