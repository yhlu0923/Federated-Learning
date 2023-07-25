import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import argparse
from datetime import datetime
import math
import torchvision.models as models

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate federated learning with different delays')
    parser.add_argument('--pic_name', type=str, default='pic_train', help='Name of the picture')
    parser.add_argument('--model_name', type=str, default='resnet', choices=['resnet', 'small_net'], help='Name of the model')
    parser.add_argument('--opt_name', type=str, default='sgd', choices=['adam', 'sgd'], help='Name of the optimizer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--num_epoch', type=int, default=25, help='Number of epochs')
    parser.add_argument('--delays', type=int, nargs='+', default=[1, 5], help='List of delays')
    args = parser.parse_args()
    return args


def get_resnet():
    resnet = models.resnet18(pretrained=False)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

    return resnet


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
    def __init__(self, num_clients, base_model, opt_name):
        learning_rate = 0.001
        momentum = 0.9
        weight_decay = 1e-4

        self.num_clients = num_clients
        self.clients = [copy.deepcopy(base_model) for _ in range(num_clients)]
        self.optimizers = [
            optim.SGD(client.parameters(), lr=learning_rate, momentum=momentum) 
            if opt_name == "sgd" 
            else optim.Adam(client.parameters(), lr=learning_rate, weight_decay=weight_decay) 
            for client in self.clients
        ]
        self.trainloaders = None

    def get_client(self, client_index):
        return self.clients[client_index]

    def get_optimizer(self, client_index):
        return self.optimizers[client_index]

    def get_trainloader(self, client_index):
        return self.trainloaders[client_index]

    def set_trainloaders(self, trainloaders):
        assert self.num_clients == len(trainloaders)
        self.trainloaders = trainloaders


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


def simulate_federated_learning(model_name, opt_name, num_clients, delay, num_epoch, batch_size):
    trainloaders, testloader = get_data_loaders(num_clients, batch_size)

    # Create the network and optimizer
    # Central model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using GPU during training")

    # Decide model network
    net = None
    if model_name == "resnet":
        net = get_resnet().to(device)
        print("Using model: resnet")
    else:
        net = Net().to(device)
        print("Using model: Small network")
    criterion = nn.CrossEntropyLoss()

    # Create a model and an optimizer for each client
    clients = Clients(num_clients, net, opt_name)
    clients.set_trainloaders(trainloaders)

    # Record the gradient difference
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

            # Record the initial state dict to calculate the difference
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
            final_state_dict = copy.deepcopy(client_model.state_dict())
            # Iterate over the model's parameters and calculate the difference in gradients

            # Store all information regarding gradient
            gradient_info = {}
            # Calculate the difference in gradients (weights)
            gradient_difference = {}
            for name, param in final_state_dict.items():
                gradient_difference[name] = param - initial_state_dict[name]

            gradient_info['gradient_difference'] = gradient_difference

            # Append gradients in to gradient list
            # Experimetal stage: Make only the last client has delay
            delay_num = [0 for _ in range(num_clients)]
            for idx in range(math.ceil(num_clients // 2)):
                delay_num[idx] = delay

            gradient_info['epoch'] = epoch
            gradient_info['delay'] = delay_num[idx]

            target_slot_num = epoch + delay_num[idx]
            # If the epoch plus delay is inside of training stage
            if target_slot_num < len(gradients):
                gradients[target_slot_num].append(gradient_info)

        # Aggregate the gradients of client models
        # Get the list of gradient difference for this epoch
        list_gradient_info = gradients[epoch]
        # Get the base model
        global_state_dict = net.state_dict()
        # Add each gradient_difference to the base model
        # TODO: Play with delay. e.g. the delay gradients will have lower effect

        def reduce_factor(cur_epoch, gradient_info):
            factor = 1
            # if there are delay: the factor should be 1/delay
            if cur_epoch != gradient_info['epoch']:
                factor = factor / abs(cur_epoch - gradient_info['epoch'])
            # if the factor is out of reasonable range, factor = 1
            if factor > 1 or factor < 0:
                factor = 1
            return factor

        for gradient_info in list_gradient_info:
            gradient_difference = gradient_info['gradient_difference']
            for name, param in global_state_dict.items():
                global_state_dict[name] = param + (gradient_difference[name] / num_clients) * reduce_factor(epoch, gradient_info)

        # Update the central model
        net.load_state_dict(global_state_dict, strict=False)

        # Evaluate the model on the test set and calculate accuracy
        accuracy = evaluate_model(net, testloader, device)
        print(f'Accuracy: {accuracy}')
        # Record accuracy for each iteration
        accuracies.append(accuracy)

    return accuracies


def plot_and_save_img(args, accuracy_all_delays):
    # Plot the accuracy for each delay
    plt.figure(figsize=(10, 6))
    for i, delay in enumerate(args.delays):
        plt.plot(range(0, args.num_epoch), accuracy_all_delays[i], label=f'Delay={delay}')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy in Federated Learning with Delayed Gradients')
    plt.legend()

    # Add information from args to the plot
    args_info = f'Number of epochs: {args.num_epoch}\n' \
                f'Model: {args.model_name}\n' \
                f'Optimizer: {args.opt_name}\n' \
                f'Number of clients: {args.num_clients}\n' \
                f'Batch size: {args.batch_size}\n' \
                f'Delays: {args.delays}'
    plt.text(0.7, 0.1, args_info, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    pic_name = f"{args.pic_name}-num_client_{args.num_clients}-delay_{args.delays}-{current_time}.png"
    plt.savefig(pic_name)  # Saving the plot as an image
    plt.show()


def main():
    args = parse_args()

    # Simulate federated learning with different delays
    accuracy_all_delays = []
    for delay in args.delays:
        accuracies = simulate_federated_learning(
            args.model_name,
            args.opt_name,
            args.num_clients, 
            delay, 
            args.num_epoch, 
            args.batch_size)
        accuracy_all_delays.append(accuracies)

    plot_and_save_img(args, accuracy_all_delays)


if __name__ == '__main__':
    main()