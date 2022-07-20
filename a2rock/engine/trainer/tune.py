from functools import partial
from typing import List, Type

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def load_data(data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_test_metrics(model, test_set):
    pass

def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = Net(config['l1'], config['l2'])

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint'))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print('Finished Training')


def train_model(
        config,
        model,
        train_data,
        split_data = True,
        device: str = 'cuda:0',
):
    net = model(**config)
    net.to(device)

    net.fit(train_data, split_data=split_data)


def log_best_trial(trial, metrics):

    print(f'Best trial config: {trial.config}')
    print(f'Best trial final validation loss: {trial.last_result["loss"]}')
    for metric in metrics:
        print(
            f'Best trial final validation {metric}: {trial.last_result[metric]}'
        )

def log_best_test_metrics(metrics):
    pass


def fine_tune(
        model,
        train_data,
        config: dict,
        max_num_epochs: int,
        metrics: List[str],
        num_samples: int,
        split_data: bool = True
):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(
        # parameter_columns=['l1', 'l2', 'lr', 'batch_size'],
        metric_columns=['loss', *metrics, 'training_iteration']
    )
    result = tune.run(
        partial(
            train_model,
            model=model,
            train_data=train_data,
            split_data=split_data,
            device=device
        ),
        resources_per_trial={'cpu': 2, 'gpu': 0},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )
    best_trial = result.get_best_trial('loss', 'min', 'last')
    log_best_trial(best_trial, metrics)

    best_trained_model = model(**best_trial.config)
    best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(
    #     os.path.join(best_checkpoint_dir, 'checkpoint')
    # )
    # best_trained_model.load_state_dict(model_state)

    # test_metrics = compute_test_metrics(best_trained_model, test_set)
    # log_best_test_metrics(test_metrics)


if __name__ == '__main__':
    # You can change the number of GPUs per trial here:
    fine_tune(num_samples=10, max_num_epochs=10)
