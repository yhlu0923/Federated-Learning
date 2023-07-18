import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

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
    
class Clients:
    def __init__(self, num_clients, base_model):
        self.num_clients = num_clients
        self.clients = [copy.deepcopy(base_model) for _ in range(num_clients)]
        self.optimizers = [optim.SGD(client.parameters(), lr=0.001, momentum=0.9) for client in self.clients]
        self.trainloaders = None

    def get_client(self, client_index):
        return self.clients[client_index]

    def get_optimizer(self, client_index):
        return self.optimizers[client_index]

    def get_trainloader(self, client_index):
        return self.trainloaders[client_index]

    def set_trainloaders(self, trainloaders):
        assert num_clients == len(trainloaders)
        self.trainloaders = trainloaders

    # def set_client(self, client_index, state_dict):
    #     self.clients[client_index].load_state_dict(state_dict)

def get_data_loaders(num_clients, batch_size):
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Divide data among clients
    split_sizes = [len(trainset) // num_clients] * num_clients
    split_sizes[-1] = len(trainset) - sum(split_sizes[:-1])
    trainsets = torch.utils.data.random_split(trainset, split_sizes)

    # Each trainloader represent a dataset for each client
    trainloaders = [torch.utils.data.DataLoader(trainsets[i], batch_size=batch_size, shuffle=True) for i in range(num_clients)]
    # Only one testing set for central model is needed
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloaders, testloader

def evaluate_model(net, testloader, device):
    net.to(device)
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
    return accuracy

def simulate_federated_learning(num_clients, delay, num_epoch, batch_size):
    trainloaders, testloader = get_data_loaders(num_clients, batch_size)

    # Create the network and optimizer
    # Central model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_central_model = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Create a model and an optimizer for each client
    clients = Clients(num_clients, net)
    clients.set_trainloaders(trainloaders)

    # Record the gradient difference here
    gradients = [[] for _ in range(num_epoch)]

    # Lists to store accuracies
    accuracies = []

    for epoch in range(num_epoch):

        # Distribute central model gradients to each remote model
        for idx in range(num_clients):
            clients.get_client(idx).load_state_dict(net.state_dict())

        # Start training for each clients
        for idx in range(num_clients):
            # Train the model on each client's data
            trainloader = clients.get_trainloader(idx)
            client_model = clients.get_client(idx).to(device)
            optimizer = clients.get_optimizer(idx)

            initial_state_dict = copy.deepcopy(client_model.state_dict())

            # Set the model to training mode
            client_model.train()

            # Visualize the training step
            with tqdm(total=len(trainloader), ncols=80, desc=f"Epoch {epoch+1}/{num_epoch}", postfix=f"Client {idx+1}/{num_clients}") as progress_bar:
                # Train the client model on the local data
                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    outputs = client_model(inputs)      # TRAINING MUST BE DONE ON CLIENTS
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # print('Grad:', client_model.conv1.weight.grad)
                    optimizer.step()

                    # Update tqdm bar
                    progress_bar.update(1)
            final_state_dict = copy.deepcopy(client_model.state_dict()) #.copy()
            # print(final_state_dict)
            # Iterate over the model's parameters and calculate the difference in gradients
            gradient_difference = {}
            for name, param in final_state_dict.items():
                # print(name)
                # print(param.requires_grad)
                # if param.requires_grad:
                #     print('inside')
                gradient_difference[name] = param - initial_state_dict[name]
            # Print or use the gradient difference as needed
            # Compare the state dictionaries
            same_state = True
            for name, param in initial_state_dict.items():
                # if param.requires_grad:
                if not torch.equal(param, final_state_dict[name]):
                    same_state = False
                    break
            if same_state:
                print("Same")
            else:
                print("Different")

        # # Calculate the difference in gradients (weights)
        # for idx in range(num_clients):
        #     initial_state_dict = net.state_dict()
        #     final_state_dict = clients.get_client(idx).state_dict()

        #     # Iterate over the model's parameters and calculate the difference in gradients
        #     gradient_difference = {}
        #     for name, param in final_state_dict.items():
        #         if param.requires_grad:
        #             gradient_difference[name] = param - initial_state_dict[name]

        #     # Print or use the gradient difference as needed
        #     print(gradient_difference)

        # Append gradients in to gradient list
        # Experimetal stage: Make only the last client has delay
        delay_num = [0 for _ in range(num_clients)]
        delay_num[0] = delay
        for i, d in enumerate(delay_num):
            target_slot_num = epoch + d
            # If the epoch plus delay is out of training stage
            if target_slot_num < len(gradients):
                client_model = clients.get_client(i)
                for param_name, param in client_model.named_parameters():
                    if param.grad is not None:
                        gradients[target_slot_num].append(param.grad.clone())

        # Aggregate the gradients of client models
        global_state_dict = net.state_dict()
        for param_name, param in global_state_dict.items():
            if param.requires_grad:
                gradients_sum = sum(gradients[epoch][i][param_name] for i in range(len(gradients[epoch])))
                param.grad = gradients_sum / num_clients

        # Update the central model with aggregated gradients
        optimizer.step()

        # Update the central model
        net.load_state_dict(global_state_dict, strict=False)

        # Evaluate the model on the test set and calculate accuracy
        accuracy = evaluate_model(net, testloader, device)
        # Record accuracy for each iteration
        accuracies.append(accuracy)

    return accuracies

# Simulate federated learning with different delays
num_epoch = 25
batch_size = 256
num_clients = 5
delays = [1, 2, 3, 4, 5]

accuracy_losses_all_delays = []
for delay in delays:
    accuracy_losses = simulate_federated_learning(num_clients, delay, num_epoch, batch_size)
    accuracy_losses_all_delays.append(accuracy_losses)

# Plot the accuracy loss for each delay
plt.figure(figsize=(10, 6))
for i, delay in enumerate(delays):
    plt.plot(range(0, num_epoch), accuracy_losses_all_delays[i], label=f'Delay={delay}')
plt.xlabel('Iteration')
plt.ylabel('Accuracy Loss')
plt.title('Accuracy Loss in Federated Learning with Delayed Gradients')
plt.legend()
plt.show()