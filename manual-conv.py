from download_mnist import load
import numpy as np
from time import time

# Training settings
BATCH_SIZE = 128  # mini-batch SGD batch size
CLASSES = 10  # number of possible output values
EPOCHS = 10  # iterations during training
FILTER1 = 32  # filters in the first CONV layer
FILTER2 = 64  # filters in the second CONV layer
FILTER_SIZE = 3  # filter size for the convolution layers
H1 = 9216  # neurons in the first FC layer
H2 = 128  # neurons in the second FC layer


# A class to store all the weights and biases in each layer
class Weights:
    def __init__(self, rng):
        self.conv1 = 0.01 * rng.standard_normal((FILTER1, 1, FILTER_SIZE, FILTER_SIZE))
        self.conv1b = np.zeros(FILTER1)  # a bias value per filter
        self.conv2 = 0.01 * rng.standard_normal((FILTER2, FILTER1, FILTER_SIZE, FILTER_SIZE))
        self.conv2b = np.zeros(FILTER2)
        self.fc1 = 0.01 * rng.standard_normal((H1, H2))
        self.fc1b = np.zeros((1, H2))
        self.fc2 = 0.01 * rng.standard_normal((H2, CLASSES))
        self.fc2b = np.zeros((1, CLASSES))


def conv(x, kernel, bias):
    # Perform convolution using the provided filters
    # Inputs: HxWxCxN, Filters: RxSxCxM, Output: ExFxMxN, P = padding, T = stride
    # E = (H + 2P - R)/T + 1
    # F = (W + 2P - S)/T + 1
    # params = (RSC + 1)M
    n, _, h, w = x.shape
    m, _, r, s = kernel.shape
    # Calculate the size of the output
    e = h - r + 1
    f = w - s + 1
    y = np.zeros((n, m, e, f))

    # Loop across n inputs
    for b in range(n):
        # Loop across m filters
        for fi in range(m):
            # Slide the filter across the input
            for i in range(e):
                for j in range(f):
                    # Do element-wise multiplication and sum all the elements across all channels
                    x_ij = x[b, :, i:(i + r), j:(j + s)]
                    y[b, fi, i, j] = np.sum(x_ij * kernel[fi]) + bias[fi]

    return y


def max_pool(x, scale):
    # Downscale the input
    n, c, h, w = x.shape
    # Calculate the size of the output
    e = h // scale
    f = w // scale
    y = np.zeros((n, c, e, f))

    # Slide the filter across the input with a stride of the scale
    for i in range(e):
        for j in range(f):
            # Find the maximum value across scale^2 elements
            x_ij = x[:, :, i * scale:(i * scale + scale), j * scale:(j * scale + scale)]
            y[:, :, i, j] = np.amax(x_ij, axis=(2, 3))

    return y


def train(x, y):
    # Zero-center the input
    x -= np.mean(x, axis=0)  # take the mean of each of the k pixels
    m = x.shape[0]

    # Randomly initialize the weights and biases
    rng = np.random.default_rng()  # PCG generator
    weights = Weights(rng)
    alpha = 0.1  # learning rate

    # Send the entire training data every epoch
    for e in range(1, EPOCHS + 1):
        # Shuffle the input every epoch to feed unique batches to the nn
        p = rng.permutation(m)  # synchronize the random indices between the data and labels
        x = x[p]
        y = y[p]

        for b in range(m // BATCH_SIZE):
            # Filter the appropriate batches from the training input and output
            xb = x[BATCH_SIZE * b: BATCH_SIZE * (b + 1)]  # BATCH_SIZE x 1 x 28 x 28
            yb = y[BATCH_SIZE * b: BATCH_SIZE * (b + 1)]  # BATCH_SIZE

            # Forward propagation
            # ReLU: max(0, x)
            # Convolution layers
            relu1 = np.maximum(0, conv(xb, weights.conv1, weights.conv1b))  # 32 x 26 x 26
            print("Conv1 done")
            relu2 = np.maximum(0, conv(relu1, weights.conv2, weights.conv2b))  # 64 x 24 x 24
            print("Conv2 done")
            # Max pooling layer, followed by flattening
            pool = max_pool(relu2, 2)  # 64 x 12 x 12
            flat = pool.reshape(pool.shape[0], -1)  # BATCH_SIZE x 9216
            print("Pooling done")
            # Fully-connected layers
            relu3 = np.maximum(0, flat @ weights.fc1 + weights.fc1b)  # BATCH_SIZE x 128
            print("FC1 done")
            scores = relu3 @ weights.fc2 + weights.fc2b  # BATCH_SIZE x 10
            print("FC2 done")

            # Softmax: e^x / sum(e^xi)
            scores -= np.amax(scores, axis=1, keepdims=True)  # prevent overflow
            exps = np.exp(scores)
            # Sum all the exps across each row; keepdims = keep the array 2D
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            # Compute the cross-entropy loss
            # Get the probability amount for the correct output of each input
            correct_probs_log = -np.log(probs[range(BATCH_SIZE), yb])
            loss = np.sum(correct_probs_log) / BATCH_SIZE
            # Print every 10th batch
            if b % 10 == 0:
                print(f"Epoch {e}/{EPOCHS}, Batch {b + 1}/{m // BATCH_SIZE}: Loss {loss}")
                return weights  # TODO: remove once the backward propagation is working

            # Backward propagation
            dscores = probs  # BATCH_SIZE x 10
            dscores[range(BATCH_SIZE), yb] -= 1  # subtract 1 from the correct labels
            dscores /= BATCH_SIZE
            dw3 = relu2.T @ dscores + LAMBDA * w3  # 50 x 10
            db3 = np.sum(dscores, axis=0, keepdims=True)  # 10

            dh2 = dscores @ w3.T  # BATCH_SIZE x 50
            dh2[relu2 <= 0] = 0  # the derivative of max(0, x) for x <= 0 is 0
            dw2 = relu1.T @ dh2 + LAMBDA * w2  # 200 x 50
            db2 = np.sum(dh2, axis=0, keepdims=True)  # 50

            dh1 = dh2 @ w2.T  # BATCH_SIZE x 200
            dh1[relu1 <= 0] = 0
            dw1 = xb.T @ dh1 + LAMBDA * w1  # k x 200
            db1 = np.sum(dh1, axis=0, keepdims=True)  # 200

            # Update the weights and biases
            weights.conv1 -= alpha * dw1
            weights.conv1b -= alpha * db1
            weights.conv2 -= alpha * dw2
            weights.conv2b -= alpha * db2
            weights.fc1 -= alpha * dw3
            weights.fc1b -= alpha * db3
            weights.fc2 -= alpha * dw4
            weights.fc2b -= alpha * db4

        # Decay the learning rate every epoch
        alpha *= 0.7

        # Compute the training accuracy after each epoch
        test(x, y, weights, is_training=True)
        # relu3 = np.maximum(0, x @ weights.fc1 + weights.fc1b)  # m x 128
        # scores = relu3 @ weights.fc2 + weights.fc2b  # m x 10
        # y_pred = np.argmax(scores, axis=1)  # get the output with the highest score
        # accuracy = np.mean(y_pred == y)  # the ratio of correct outputs
        # print(f"Epoch {e}: Training accuracy {(100 * accuracy):.0f}%")

    return weights


def test(x, y, weights, is_training=False):
    # Calculate the accuracy of the testing data
    x -= np.mean(x, axis=0)  # need to zero-center the input, like with training
    # Convolution layers
    relu1 = np.maximum(0, conv(x, weights.conv1, weights.conv1b))  # 32 x 26 x 26
    print("Conv1 done")
    relu2 = np.maximum(0, conv(relu1, weights.conv2, weights.conv2b))  # 64 x 24 x 24
    print("Conv2 done")
    # Max pooling layer, followed by flattening
    pool = max_pool(relu2, 2)  # 64 x 12 x 12
    flat = pool.reshape(pool.shape[0], -1)  # m x 9216
    print("Pooling done")
    # Fully-connected layers
    relu3 = np.maximum(0, flat @ weights.fc1 + weights.fc1b)  # m x 128
    print("FC1 done")
    scores = relu3 @ weights.fc2 + weights.fc2b  # m x 10
    print("FC2 done")
    y_pred = np.argmax(scores, axis=1)
    accuracy = np.mean(y_pred == y)

    # Indicate whether this is the training or testing accuracy
    if is_training:
        print(f"Training accuracy {(100 * accuracy):.0f}%")
    else:
        print(f"Testing accuracy {(100 * accuracy):.0f}%")


def main():
    # x_train: 60K x 784, y_train: 60K, x_test: 10K x 784, y_test: 10K
    x_train, y_train, x_test, y_test = load()
    # Un-flatten the images and add a dimension for the number of channels
    x_train = x_train.reshape(-1, 1, 28, 28).astype(float)
    x_test = x_test.reshape(-1, 1, 28, 28).astype(float)

    start = time()
    weights = train(x_train, y_train)
    print(f"Training took {time() - start} s")
    test(x_test, y_test, weights)


if __name__ == "__main__":
    main()
