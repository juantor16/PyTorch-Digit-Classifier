import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data
from PIL import Image
import os
import glob

# Define the Net neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Model Trainer Class
class ModelTrainer:
    def __init__(self, model, trainloader, optimizer, loss_function, epochs=15):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.trainloader)}")
        torch.save(self.model.state_dict(), 'mnist_model.pth')
        print("Model saved as mnist_model.pth")

# Model Tester Class
class ModelTester:
    def __init__(self, model, testloader):
        self.model = model
        self.testloader = testloader

    def test_accuracy(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10,000 test images: {100 * correct / total}%')

# PredictImage Class for new data
class PredictImage:
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def predict(self, image_path):
        # Single image prediction
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def predict_multiple(self, folder_path):
        # Predict multiple images from a folder
        image_files = sorted(glob.glob(os.path.join(folder_path, 'testData/img_*.jpg')))
        predictions = {}
        for image_path in image_files:
            prediction = self.predict(image_path)
            predictions[image_path] = prediction
        return predictions

            

# Main code: Load data, train, evaluate, and predict
def main():
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('MNIST_data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST('MNIST_data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Initialize Model
    model = Net()

    # Check if a trained model exists
    model_path = 'mnist_model.pth'
    if os.path.isfile(model_path):
        print("Loading trained model...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Training model...")
        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
        loss_function = nn.CrossEntropyLoss()
        trainer = ModelTrainer(model, trainloader, optimizer, loss_function, epochs=15)
        trainer.train()
        torch.save(model.state_dict(), model_path)
        print("Model trained and saved.")

   # Evaluate model accuracy
    tester = ModelTester(model, testloader)
    tester.test_accuracy()

    # Predicting new data
    predictor = PredictImage(model)
    # Path to the folder containing your images
    folder_path = '.'  # Adjust this to the path where your images are located
    predictions = predictor.predict_multiple(folder_path)

    # Print predictions in order
    for image_path, predicted_digit in predictions.items():
        print(f'{image_path}: Predicted Digit - {predicted_digit}')

if __name__ == '__main__':
    main()
