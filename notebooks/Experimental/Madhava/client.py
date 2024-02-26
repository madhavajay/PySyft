"""
model definition
param init / load weights
fit locally
send back weights to aggregate
send back to the clients to eval and send back the result

weighted_average metrics?
send in model

TODO:
create model
send in model definition
load weights (action object jax example)
assign dataset


Calling get_parameters with  <class 'dict'> {}
Calling fit with  <class 'list'> <class 'dict'> {}
Calling set_parameters with  <class 'list'>

Training set
Test set


Weights MutableObject




Training: 100%|█████████████████| 417/417 [00:06<00:00, 68.55it/s]
Calling get_parameters with  <class 'dict'> {}
Calling evaluate with  <class 'list'> <class 'dict'> {}
Calling set_parameters with  <class 'list'>
Testing: 100%|█████████████████| 105/105 [00:00<00:00, 183.10it/s]
Calling fit with  <class 'list'> <class 'dict'> {}
Calling set_parameters with  <class 'list'>
Training: 100%|█████████████████| 417/417 [00:07<00:00, 56.45it/s]
Calling get_parameters with  <class 'dict'> {}
Calling evaluate with  <class 'list'> <class 'dict'> {}
Calling set_parameters with  <class 'list'>
Testing: 100%|█████████████████| 105/105 [00:00<00:00, 175.10it/s]
Calling fit with  <class 'list'> <class 'dict'> {}
Calling set_parameters with  <class 'list'>
Training: 100%|█████████████████| 417/417 [00:07<00:00, 58.35it/s]
Calling get_parameters with  <class 'dict'> {}
Calling evaluate with  <class 'list'> <class 'dict'> {}
Calling set_parameters with  <class 'list'>
Testing: 100%|█████████████████| 105/105 [00:00<00:00, 185.46it/s]
DEBUG flwr 2024-02-19 14:59:43,076 | connection.py:220 | gRPC channel closed
INFO flwr 2024-02-19 14:59:43,076 | app.py:398 | Disconnect and shut down
 ~/d/fl                                 ✔  46s  fl-_kL-QqcT Py

"""


import argparse
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data(node_id):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(node_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get node id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--node-id",
    choices=[0, 1, 2],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
node_id = parser.parse_args().node_id

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data(node_id=node_id)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print("Calling get_parameters with ", type(config), config)
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        print("Calling set_parameters with ", type(parameters))
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Calling fit with ", type(parameters), type(config), config)
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Calling evaluate with ", type(parameters), type(config), config)
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)