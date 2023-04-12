import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("QtAgg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 28
num_classes = 10
num_epochs = 3
batch_size = 100

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum().item()
    return rights, len(labels)


net = CNN()
net.to(device)
criterion = nn.CrossEntropyLoss()

plot_train_acc0 = []
plot_train_acc1 = []
plot_train_acc2 = []
plot_val_acc0 = []
plot_val_acc1 = []
plot_val_acc2 = []
plot_loss = []

for epoch in range(num_epochs):
    train_rights = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        net.train()
        output = net(data)
        loss = criterion(output, target)
        net.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in net.parameters():
                param -= 0.001 * param.grad

        right = accuracy(output, target)
        train_rights.append(right)

        if batch_idx % 10 == 0:
            net.eval()
            val_rights = []
            for (data, target) in test_loader:
                data, target = data.to(device), target.to(device)
                output = net(data)
                right = accuracy(output, target)
                val_rights.append(right)

            train_r = (sum([tup[0] for tup in train_rights]),
                       sum([tup[1] for tup in train_rights]))

            val_r = (sum([tup[0] for tup in val_rights]),
                     sum([tup[1] for tup in val_rights]))

            print(
                "Current epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTraining "
                "Set accuracy: {:.2f}%\t Test Set accuracy: {:.2f}%".format(
                    epoch, batch_idx * batch_size, len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data,
                           100. * train_r[0] / train_r[1],
                           100. * val_r[0] / val_r[1],
                ))
            if epoch == 0:
                plot_train_acc0.append(100. * train_r[0] / train_r[1])
                plot_val_acc0.append(100. * val_r[0] / val_r[1])
            elif epoch == 1:
                plot_train_acc1.append(100. * train_r[0] / train_r[1])
                plot_val_acc1.append(100. * val_r[0] / val_r[1])
            elif epoch == 2:
                plot_train_acc2.append(100. * train_r[0] / train_r[1])
                plot_val_acc2.append(100. * val_r[0] / val_r[1])
            plot_loss.append(loss.data.cpu().numpy())

plt.figure()
plt.plot(np.arange(0, len(plot_train_acc0)*10, 10), plot_train_acc0)
plt.plot(np.arange(0, len(plot_val_acc0)*10, 10), plot_val_acc0)
plt.grid(axis='both', color='0.95')
plt.xlabel('Number of Batch (*100)')
plt.ylabel('Accuracy')
plt.title('Epoch 1 Accuracy')
plt.legend(["Training Set","Test Set"])
plt.savefig("1"+".png", format='png')

plt.figure()
plt.plot(np.arange(0, len(plot_train_acc1)*10, 10), plot_train_acc1)
plt.plot(np.arange(0, len(plot_val_acc1)*10, 10), plot_val_acc1)
plt.grid(axis='both', color='0.95')
plt.xlabel('Number of Batch (*100)')
plt.ylabel('Accuracy')
plt.title('Epoch 2 Accuracy')
plt.legend(["Training Set","Test Set"])
plt.savefig("2"+".png", format='png')

plt.figure()
plt.plot(np.arange(0, len(plot_train_acc2)*10, 10), plot_train_acc2)
plt.plot(np.arange(0, len(plot_val_acc2)*10, 10), plot_val_acc2)
plt.grid(axis='both', color='0.95')
plt.xlabel('Number of Batch (*100)')
plt.ylabel('Accuracy')
plt.title('Epoch 3 Accuracy')
plt.legend(["Training Set","Test Set"])
plt.savefig("3"+".png", format='png')

plt.figure()
plt.plot(np.arange(0,  len(plot_loss)), plot_loss)
plt.grid(axis='both', color='0.95')
plt.ylabel('Loss')
plt.title('Changing of Loss')
plt.savefig("4"+".png", format='png')