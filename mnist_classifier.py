import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data

# Neural Network Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer (28x28 pixels input, 128 outputs)
        self.fc2 = nn.Linear(128, 64)     # Hidden layer (128 inputs, 64 outputs)
        self.fc3 = nn.Linear(64, 10)      # Output layer (64 inputs, 10 outputs - one per digit)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input tensor
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

    def load_model(self, path='mnist_model.pth'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

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

# Main Code
if __name__ == '__main__':
    # Data loading and transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST('MNIST_data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = datasets.MNIST('MNIST_data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Model, Optimizer, and Loss Function
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    # Training and Testing
    trainer = ModelTrainer(model, trainloader, optimizer, loss_function, epochs=15)
    trainer.train()

    tester = ModelTester(model, testloader)
    tester.load_model('mnist_model.pth')  # Make sure to have the model saved or comment this line if running for the first time
    tester.test_accuracy()
