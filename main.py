import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the CIFAR-10 network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train(net, dataloader, criterion, optimizer):
    net.train()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples
    return running_loss, accuracy

# Function to simulate federated learning with delayed gradients
def simulate_federated_learning(num_clients, delay):
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Divide data among clients
    trainloaders = torch.utils.data.random_split(trainset, num_clients)
    testloaders = torch.utils.data.random_split(testset, num_clients)

    # Create the network and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Lists to store accuracy losses
    accuracy_losses = []

    for epoch in range(25):
        for client in range(num_clients):
            # Train the model on each client's data
            trainloader = torch.utils.data.DataLoader(trainloaders[client], batch_size=16, shuffle=True)
            running_loss, accuracy = train(net, trainloader, criterion, optimizer)

            # Simulate delayed gradients
            if client == delay - 1:
                delayed_running_loss = running_loss
                delayed_accuracy = accuracy

        # Record accuracy loss for each iteration
        accuracy_loss = delayed_accuracy - accuracy
        accuracy_losses.append(accuracy_loss)

    return accuracy_losses

# Simulate federated learning with different delays
num_clients = 5
delays = [1, 2, 3, 4, 5]

accuracy_losses_all_delays = []
for delay in delays:
    accuracy_losses = simulate_federated_learning(num_clients, delay)
    accuracy_losses_all_delays.append(accuracy_losses)

# Plot the accuracy loss for each delay
plt.figure(figsize=(10, 6))
for i, delay in enumerate(delays):
    plt.plot(range(1, 26), accuracy_losses_all_delays[i], label=f'Delay={delay}')
plt.xlabel('Iteration')
plt.ylabel('Accuracy Loss')
plt.title('Accuracy Loss in Federated Learning with Delayed Gradients')
plt.legend()
plt.show()

