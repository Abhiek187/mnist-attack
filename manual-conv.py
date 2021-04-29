from download_mnist import load
import numpy as np
from time import time

ALPHA = 0.01  # learning rate
BATCH_SIZE = 128  # mini-batch SGD batch size
CLASSES = 10  # number of possible output values
EPOCHS = 10  # iterations during training
H1 = 200  # neurons in the 1st hidden layer
H2 = 50  # neurons in the 2nd hidden layer
LAMBDA = 0.001  # regularization constant


def train(x, y):
    # Zero-center the input
    x -= np.mean(x, axis=0)  # take the mean of each of the k pixels
    m, k = x.shape

    # Randomly initialize the weights and biases
    rng = np.random.default_rng()
    w1 = 0.01 * rng.standard_normal((k, H1))
    b1 = np.zeros((1, H1))
    w2 = 0.01 * rng.standard_normal((H1, H2))
    b2 = np.zeros((1, H2))
    w3 = 0.01 * rng.standard_normal((H2, CLASSES))
    b3 = np.zeros((1, CLASSES))

    # Send the entire training data every epoch
    for e in range(1, EPOCHS + 1):
        # Shuffle the input every epoch to feed unique batches to the nn
        p = rng.permutation(m)  # synchronize the random indices between the data and labels
        x = x[p]
        y = y[p]

        for b in range(m // BATCH_SIZE):
            # Filter the appropriate batches from the training input and output
            xb = x[BATCH_SIZE * b: BATCH_SIZE * (b + 1)]  # BATCH_SIZE x k
            yb = y[BATCH_SIZE * b: BATCH_SIZE * (b + 1)]  # BATCH_SIZE

            # Forward propagation
            # ReLU: max(0, x)
            relu1 = np.maximum(0, xb @ w1 + b1)  # BATCH_SIZE x 200
            relu2 = np.maximum(0, relu1 @ w2 + b2)  # BATCH_SIZE x 50
            scores = relu2 @ w3 + b3  # BATCH_SIZE x 10

            # Softmax: e^x / sum(e^xi)
            scores -= np.amax(scores, axis=1, keepdims=True)  # prevent overflow
            exps = np.exp(scores)
            # Sum all the exps across each row; keepdims = keep the array 2D
            probs = exps / np.sum(exps, axis=1, keepdims=True)

            # Compute the cross-entropy loss
            # Get the probability amount for the correct output of each input
            correct_probs_log = -np.log(probs[range(BATCH_SIZE), yb])
            loss_data = np.sum(correct_probs_log) / BATCH_SIZE
            # L2 loss (ridge regression)
            loss_reg = 0.5 * LAMBDA * (np.sum(w1 ** 2) + np.sum(w2 ** 2) + np.sum(w3 ** 2))
            loss = loss_data + loss_reg

            # Print every 10th batch
            if b % 10 == 0:
                print(f"Epoch {e}/{EPOCHS}, Batch {b + 1}/{m // BATCH_SIZE}: Loss {loss}")

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
            w1 -= ALPHA * dw1
            b1 -= ALPHA * db1
            w2 -= ALPHA * dw2
            b2 -= ALPHA * db2
            w3 -= ALPHA * dw3
            b3 -= ALPHA * db3

        # Compute the training accuracy after each epoch
        relu1 = np.maximum(0, x @ w1 + b1)  # m x 200
        relu2 = np.maximum(0, relu1 @ w2 + b2)  # m x 50
        scores = relu2 @ w3 + b3  # m x 10
        y_pred = np.argmax(scores, axis=1)  # get the output with the highest score
        accuracy = np.mean(y_pred == y)  # the ratio of correct outputs
        print(f"Epoch {e}: Training accuracy {(100 * accuracy):.0f}%")

    return w1, b1, w2, b2, w3, b3


def test(x, y, w1, b1, w2, b2, w3, b3):
    # Calculate the accuracy of the testing data
    x -= np.mean(x, axis=0)  # need to zero-center the input, like with training
    relu1 = np.maximum(0, x @ w1 + b1)  # m x 200
    relu2 = np.maximum(0, relu1 @ w2 + b2)  # m x 50
    scores = relu2 @ w3 + b3  # m x 10
    y_pred = np.argmax(scores, axis=1)
    accuracy = np.mean(y_pred == y)
    print(f"Testing accuracy {(100 * accuracy):.0f}%")


def main():
    # x_train: 60K x 784, y_train: 60K, x_test: 10K x 784, y_test: 10K
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)

    start = time()
    w1, b1, w2, b2, w3, b3 = train(x_train, y_train)
    print(f"Training took {time() - start} s")
    test(x_test, y_test, w1, b1, w2, b2, w3, b3)


if __name__ == "__main__":
    main()
