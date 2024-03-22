# PyTorch MNIST Classifier

This repository contains a PyTorch implementation of a neural network designed to recognize handwritten digits from the MNIST dataset. I had a lot of fun doing this, it's a great starting point for anyone learning about machine learning and neural networks.

## Features

- PyTorch neural network training and evaluation.
- Data preprocessing and augmentation techniques.
- Techniques to prevent overfitting.
- Prediction script for evaluating new images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them.


### Installing

A step-by-step series of examples that tell you how to get a development environment running.

1. Clone the repository:
`git clone https://github.com/your-username/PyTorch-MNIST-Classifier.git`

2. Install the dependencies
`pip install torch torchvision torchaudio`

3. Training the Model
To train the model, execute the training script: `python3 mnist_classifier.py`
this will run 15 epochs training the neural network using the data within the MNIST_data folder. 
At the end of the execution, you will be able to see the console message  with the accuracy score achieved by the model on the The trained models are saved in the "mnist_model.pth"

4. Evaluating New Images
You can use the prediction script by executing this file:
`python3 predict_image.py` 
This will load the trained model if it has been trained in step 3. You can always delete the mnist_model.pth file to force retraining.
It will also show you accuracy of the network.
Then it will show you a list of jpg files and the prediction of the numbers right beside it.

You can create your own files and put them in the testData folder, ensure your custom images are placed in the testData folder, adhering to the 28px by 28px size with a black background and white digits for optimal performance.

I used some chatgpt to help me implement the steps, I have only had fundamental training on theory very little implementation but wanted to give this a try, because as I move along learning, I will understand this project more and more, which is helpful for me to track progress.

If you would like to reach out to me my email is: juantor16@gmail.com







