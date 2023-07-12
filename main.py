import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    trainsets = torch.utils.data.random_split(trainset, [int(len(trainset)/num_clients)] * num_clients)
    trainloaders = [torch.utils.data.DataLoader(trainsets[i], batch_size=16, shuffle=True) for i in range(num_clients)]
    # Have one test data for central model, no need to have local test data
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    # Create the network and optimizer
    # Central model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Create a copy of the model for each client
    clients = [Net().to(device) for _ in range(num_clients)]
    for client_model in clients:
        client_model.load_state_dict(net.state_dict())

    num_epoch = 10
    # TODO: Currently it is state_dict(), change it into gradients afterwards
    gradients = [[] for _ in range(num_epoch)]

    # Lists to store accuracy losses
    accuracy_losses = []

    for epoch in range(num_epoch):
        for client, client_model in enumerate(clients):
            # Train the model on each client's data
            trainloader = trainloaders[client]

            # Set the model to training mode
            client_model.train()

            # Visualize the training step
            with tqdm(total=len(trainloader), ncols=80) as progress_bar:
                # Train the client model on the local data
                for inputs, labels in trainloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Update tqdm bar
                    progress_bar.update(1)

        # Append gradients in to gradient list
        delay_num = [0 for _ in range(num_clients)]
        delay_num[0] = delay
        for i, d in enumerate(delay_num):
            target_slot_num = epoch + d
            # Dispose gradients we get if it surpasses the length of our list
            if target_slot_num < len(gradients):
                gradients[target_slot_num].append(clients[i].state_dict().copy())

        # Aggregate the client models' weights
        global_state_dict = net.state_dict()
        for param_name in global_state_dict:
            global_state_dict[param_name] = sum(state_dict[param_name] for state_dict in gradients[epoch]) / len(gradients[epoch])

        # Update the central model
        net.load_state_dict(global_state_dict, strict=False)
        net.to(device)

        # Evaluate the model on the test set and calculate accuracy loss
        net.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total

        # Record accuracy loss for each iteration
        accuracy_losses.append(1 - accuracy)

    return accuracy_losses

# Simulate federated learning with different delays
num_clients = 5
delays = [1, 5]

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