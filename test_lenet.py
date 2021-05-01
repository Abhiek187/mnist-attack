# Inputs: HxWxCxN, Filters: RxSxCxM, Output: ExFxMxN, P = padding, T = stride
# E = (H + 2P - R)/T + 1
# F = (W + 2P - S)/T + 1
# params = (RSC + 1)M
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Clamp the output to the range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # Set the model to testing mode
    model.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Feed one batch at a time
    for test_idx, (data, target) in enumerate(test_loader):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        if test_idx % 100 == 0:
            print(f"Eps = {epsilon}: Batch {test_idx}/{len(test_loader.dataset)} "
                  f"({100. * test_idx / len(test_loader):.0f}%)")

        # Get the gradient to prepare for the attack
        data.requires_grad = True
        # Do a forward pass
        output = model(data)
        # Get the index of the max log-probability
        init_pred = output.argmax(1, keepdim=True)[0]

        # If the initial prediction is wrong, no need to attack
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        test_loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Back propagation
        test_loss.backward()
        # Collect the gradient of the data
        data_grad = data.grad.data
        # Do an FGSM (fast gradient sign method) attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)
        # Check if the attack was successful
        final_pred = output.argmax(1, keepdim=True)[0]

        if final_pred.item() == target.item():
            correct += 1

        # Save the first 5 images for each epsilon
        if test_idx < 5:
            adv_examples.append((init_pred.item(), final_pred.item(),
                                 perturbed_data[0][0].cpu().detach()))

    accuracy = correct / len(test_loader.dataset)
    return accuracy, adv_examples


def main():
    # Use either the CPU or GPU
    no_cuda = False
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1, shuffle=True)

    # Load the pre-trained model
    model = LeNet().to(device)

    if FLAGS.start_checkpoint:
        ckpt = torch.load(FLAGS.start_checkpoint, map_location=device)
        model.load_state_dict(ckpt)

    epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1]
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 1.2, step=0.2))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()
    plt.savefig(FLAGS.acc_fig)
    plt.close(fig)

    # Visualize the effect of the attack
    cnt = 0
    fig = plt.figure(figsize=(8, 10))

    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])

            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)

            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")

    plt.tight_layout()
    plt.show()
    plt.savefig(FLAGS.adv_fig)
    plt.close(fig)


if __name__ == "__main__":
    # Read all the command line arguments, or supply default values
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="downloads",
                        help="Where to download the MNIST data to")
    parser.add_argument("--start_checkpoint", type=str, default="mnist_cnn.pth",
                        help="If specified, restore this pretrained model before any training.")
    parser.add_argument("--acc_fig", type=str, default="downloads/accuracy.png",
                        help="Where to save the plotted accuracy curve.")
    parser.add_argument("--adv_fig", type=str, default="downloads/attack.png",
                        help="Where to save the plotted attack figure.")
    FLAGS, _ = parser.parse_known_args()
    main()
