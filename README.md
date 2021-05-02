# mnist-attack
This program showcases the security vulnerabilities of neural networks. It starts by generating a LeNet-5 model for the MNIST dataset using PyTorch. Then it will inject an FGSM (fast gradient sign method) attack on the trained model. In doing so, the accuracy worsens without showing a noticeable difference in the perturbed images.

## Instructions
1. Run `python3 gen_lenet.py` to save the LeNet model.
2. Run `python3 test_lenet.py` to launch the adversarial attack.

The results will appear in the `downloads` folder (ignored by Git).
