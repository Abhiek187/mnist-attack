# Inputs: HxWxCxN, Filters: RxSxCxM, Output: ExFxMxN, P = padding, T = stride
# E = (H + 2P - R)/T + 1
# F = (W + 2P - S)/T + 1
# params = (RSC + 1)M
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

mpl.use("Agg")


class LeNet(nn.Module):
    """
    LeNet model:
    Input: 28x28, 1 channel
    Conv1: 6 5x5 filters, padding 2 -> 6x28x28 output, Tanh activation
    MaxPool1: downscale by 2 -> 6x14x14 output
    Conv2: 16 5x5 filters -> 16x10x10 output, Tanh activation
    MaxPool2: downscale by 2 -> 16x5x5 output
    Conv3: 120 5x5 filters -> 120x1x1 output, Tanh activation
    Flatten: 120 layers
    FC1: 120 -> 84 layers, Tanh activation
    FC2: 84 -> 10 layers
    """
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh()
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        out = self.conv(x).view(-1, 120)
        return self.fc(out)


def train(model, device, train_loader, optimizer, loss, epoch):
    # Set the model to training mode
    model.train()
    # Store all the loss values for the graph
    tmp_loss = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if loss == "CE":
            # CrossEntropy Loss
            loss_fn = nn.CrossEntropyLoss()
        else:
            # MSE Loss
            # Prepare for one-hot labels
            y_one_hot = target.numpy()
            y_one_hot = (np.arange(10) == y_one_hot[:, None]).astype(np.float32)
            target = torch.from_numpy(y_one_hot)
            loss_fn = nn.MSELoss()

        # Compute the loss and perform backward propagation
        loss_ = loss_fn(output, target)
        loss_.backward()
        optimizer.step()

        # Print the progress every log interval
        if batch_idx % FLAGS.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_.item():.6f}")
            tmp_loss.append(loss_.item())

    return tmp_loss


def test(model, device, test_loader, is_training=False):
    # Set the model to testing mode
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Compute the loss after a forward pass
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if is_training:
        print(f"\nTraining set: Average loss: {test_loss:.4f}, Accuracy: "
              f"{correct}/{len(test_loader.dataset)} "
              f"({100. * correct / len(test_loader.dataset):.0f}%)\n")
    else:
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: "
              f"{correct}/{len(test_loader.dataset)} "
              f"({100. * correct / len(test_loader.dataset):.0f}%)\n")


def main():
    # Use either the CPU or GPU
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the MNIST dataset
    # x_train: 60Kx28x28, y_train: 60K, x_test: 10Kx28x28, y_test: 10K
    batch_size = FLAGS.train_batch_size
    test_batch_size = FLAGS.test_batch_size

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.data_dir, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)

    # Set the optimizer to update the weights and bias
    lr = FLAGS.lr
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # Training settings
    epochs = FLAGS.epochs
    loss = FLAGS.loss_fn

    if FLAGS.start_checkpoint:
        ckpt = torch.load(FLAGS.start_checkpoint)
        model.load_state_dict(ckpt)

    # Start training
    time0 = time()
    loss_values = []
    # Decay the learning rate 70% every epoch
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, epochs + 1):
        _ = train(model, device, train_loader, optimizer, loss, epoch)
        loss_values.extend(_)
        test(model, device, train_loader, is_training=True)
        test(model, device, test_loader)
        scheduler.step()

    if FLAGS.save_model:
        torch.save(model.state_dict(), FLAGS.save_checkpoint)

    time1 = time()
    # Plot the training loss
    plt.figure()
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    # Align the x values to the correct epoch number
    epoch_values = np.arange(1, len(loss_values) + 1, dtype=np.float)
    epoch_values *= FLAGS.train_batch_size * FLAGS.log_interval
    epoch_values /= len(train_loader.dataset)
    plt.xticks(np.arange(1, 11))
    plt.plot(epoch_values, loss_values)
    plt.savefig(FLAGS.loss_fig)
    print(f"Training and Testing total execution time is: {time1 - time0} seconds ")


if __name__ == "__main__":
    # Read all the command line arguments, or supply default values
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="downloads",
                        help="Where to download the MNIST data to")
    parser.add_argument("--train_batch_size", type=int, default=128, metavar="N",
                        help="input batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=10000, metavar="N",
                        help="input batch size for testing")
    parser.add_argument("--loss_fn", type=str, default="CE",
                        help="select one loss function from CE and MSE")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.10, metavar="LR",
                        help="learning rate ")
    parser.add_argument("--log_interval", type=int, default=100, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--start_checkpoint", type=str, default=None,  # or "mnist_cnn.pth",
                        help="If specified, restore this pretrained model before any training.")
    parser.add_argument("--save_model", action="store_true", default=True,
                        help="For Saving the current Model")
    parser.add_argument("--save_checkpoint", type=str, default="mnist_cnn.pth",
                        help="Save the trained model.")
    parser.add_argument("--loss_fig", type=str, default="downloads/loss_curve.png",
                        help="Where to save the plotted training loss curve.")
    FLAGS, _ = parser.parse_known_args()
    main()
