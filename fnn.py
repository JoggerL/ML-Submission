import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum().item()
    return rights, len(labels)

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
acc0 = []
acc1 = []
acc2 = []
plot_train_acc = [[] for _ in range(num_epochs)]
plot_test_acc = [[] for _ in range(num_epochs)]
plot_loss = []


total_step = len(train_loader)
for epoch in range(num_epochs):
    train_rights = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(outputs, labels)
        train_rights.append(right)
        if (i + 1) % 10 == 0:
            test_rights = []
            for (images, labels) in test_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                right = accuracy(outputs, labels)
                test_rights.append(right)

            train_r = (sum([tup[0] for tup in train_rights]),
                       sum([tup[1] for tup in train_rights]))

            test_r = (sum([tup[0] for tup in test_rights]),
                      sum([tup[1] for tup in test_rights]))

            plot_train_acc[epoch].append(100. * train_r[0] / train_r[1])
            plot_test_acc[epoch].append(100. * test_r[0] / test_r[1])
            plot_loss.append(loss.item())

            print(
                "Current epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining "
                "Set accuracy: {:.2f}%\t Test Set accuracy: {:.2f}%".format(
                    epoch, i * len(images), len(train_loader.dataset),
                           100. * i / len(train_loader),
                    loss.item(),
                           100. * train_r[0] / train_r[1],
                           100. * test_r[0] / test_r[1],
                ))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


for epoch in range(num_epochs):
    plt.figure()
    plt.plot(np.arange(0, len(plot_train_acc[epoch])*10, 10), plot_train_acc[epoch])
    plt.plot(np.arange(0, len(plot_test_acc[epoch])*10, 10), plot_test_acc[epoch])
    plt.grid(axis='both', color='0.95')
    plt.xlabel('Number of Batch (*100)')
    plt.ylabel('Accuracy')
    plt.title(f'Epoch {epoch+1} Accuracy')
    plt.legend(["Training Set", "Test Set"])
    plt.savefig(f"{epoch+1}.png", format='png')

plt.figure()
plt.plot(np.arange(0, len(plot_loss)), plot_loss)
plt.grid(axis='both', color='0.95')
plt.ylabel('Loss')
plt.title('Changing of Loss')
plt.savefig("6.png", format='png')
