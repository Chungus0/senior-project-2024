#! /usr/bin/env python
#! coding:utf-8
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

from dataloader.custom_loader import load_waving_data, WavingDataGenerator, WavingConfig
from models.DDNet_Custom import DDNet_Waving as DDNet  # Import your custom model
from utils import makedir
import sys
import time
import numpy as np
import logging

sys.path.insert(0, "./pytorch-summary/torchsummary/")
from torchsummary import summary  # noqa

# Create directory for experiments
savedir = Path("experiments") / Path(str(int(time.time())))
makedir(savedir)
logging.basicConfig(filename=savedir / "train.log", level=logging.INFO)
history = {"train_loss": [], "test_loss": [], "test_acc": []}


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2, target) in enumerate(tqdm(train_loader)):
        M, P, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(M, P)
        loss = criterion(output, target)
        train_loss += loss.detach().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            msg = "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data1),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
            print(msg)
            logging.info(msg)
            if args.dry_run:
                break
    history["train_loss"].append(train_loss)
    return train_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for _, (data1, data2, target) in enumerate(tqdm(test_loader)):
            M, P, target = data1.to(device), data2.to(device), target.to(device)
            output = model(M, P)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(correct / len(test_loader.dataset))
    msg = "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
    )
    print(msg)
    logging.info(msg)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=199, metavar="N", help="number of epochs to train (default: 199)"
    )
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument(
        "--gamma", type=float, default=0.5, metavar="M", help="Learning rate step gamma (default: 0.5)"
    )
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()
    logging.info(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True})

    # Load the waving dataset
    Config = WavingConfig()  # Use custom configuration
    Train, Test, le = load_waving_data()  # Load the dataset using your custom loader

    # Generate training and test data
    X_0_train, X_1_train, Y_train = WavingDataGenerator(Train, Config, le)
    X_0_train = torch.from_numpy(X_0_train).type("torch.FloatTensor")
    X_1_train = torch.from_numpy(X_1_train).type("torch.FloatTensor")
    Y_train = torch.from_numpy(Y_train).type("torch.LongTensor")

    X_0_test, X_1_test, Y_test = WavingDataGenerator(Test, Config, le)
    X_0_test = torch.from_numpy(X_0_test).type("torch.FloatTensor")
    X_1_test = torch.from_numpy(X_1_test).type("torch.FloatTensor")
    Y_test = torch.from_numpy(Y_test).type("torch.LongTensor")

    # Prepare DataLoader for training and testing
    trainset = torch.utils.data.TensorDataset(X_0_train, X_1_train, Y_train)
    train_loader = torch.utils.data.DataLoader(trainset, **kwargs)

    testset = torch.utils.data.TensorDataset(X_0_test, X_1_test, Y_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size)

    # Initialize the model
    model = DDNet(Config.frame_l, Config.joint_n, Config.joint_d, Config.feat_d, Config.filters, Config.clc_num)
    model = model.to(device)

    # Display model summary
    print(f"Shape of input M: {(Config.frame_l, Config.feat_d)}")
    print(f"Shape of input P: {(Config.frame_l, Config.joint_n, Config.joint_d)}")

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=5, cooldown=0.5, min_lr=5e-6, verbose=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(model, device, test_loader)
        scheduler.step(train_loss)

    # Plot training results
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(history["train_loss"])
    ax1.plot(history["test_loss"])
    ax1.legend(["Train", "Test"], loc="upper left")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Loss")

    ax2.set_title("Model accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.plot(history["test_acc"])
    xmax = np.argmax(history["test_acc"])
    ymax = np.max(history["test_acc"])
    text = "x={}, y={:.3f}".format(xmax, ymax)
    ax2.annotate(text, xy=(xmax, ymax))

    ax3.set_title("Confusion matrix")
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_0_test.to(device), X_1_test.to(device)).cpu().numpy()
    Y_test_np = Y_test.numpy()
    cnf_matrix = confusion_matrix(Y_test_np, np.argmax(Y_pred, axis=1))
    ax3.imshow(cnf_matrix)
    fig.tight_layout()
    fig.savefig(str(savedir / "perf.png"))

    # Save the model if required
    if args.save_model:
        torch.save(model.state_dict(), str(savedir / "model.pt"))


if __name__ == "__main__":
    main()
