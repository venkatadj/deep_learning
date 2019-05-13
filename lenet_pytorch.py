import torch.nn as nn
from collections import OrderedDict
from torchvision import datasets
from matplotlib import pyplot as plt
import torch.optim as optim
import torch
import numpy as np
import skimage as sk
np.random.seed(1337)

class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('0', nn.Conv2d(1, 8, kernel_size=(5,5), stride=(1, 1),padding=(2,2))),
            ('1',nn.ReLU()),
            ('2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            ('3', nn.Conv2d(8, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))),
            ('4', nn.ReLU()),
            ('5', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('0', nn.Linear(in_features=400, out_features=120, bias=True)),
            ('1', nn.ReLU()),
            ('2', nn.Linear(in_features=120, out_features=84, bias=True)),
            ('3', nn.ReLU()),
            ('4', nn.Linear(in_features=84, out_features=10, bias=True)),
            ('5', nn.Softmax())
        ]))

    def forward(self, img):
        output = self.features(img)
        output = output.view(-1, 400)
        output = self.classifier(output)
        return output

def get_data_mnist(self):
    self.mnist_trainset = datasets.MNIST(root='~', train=True, download=True, transform=None)
    self.train_data=self.mnist_trainset.train_data
    self.train_data=self.train_data.float()
    self.train_data = self.train_data/255
    self.train_data =self.train_data.reshape(self.train_data.shape[0],1,1,28,28)
    self.train_labels=self.mnist_trainset.train_labels
    self.mnist_testset = datasets.MNIST(root='~', train=False, download=True, transform=None)
    self.test_data=self.mnist_testset.test_data
    self.test_data = self.test_data.float()
    self.test_data = self.test_data / 255
    self.test_data = self.test_data.reshape(self.test_data.shape[0], 1, 1, 28, 28)
    self.test_labels = self.mnist_testset.test_labels

def train_model(self):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.net.parameters(), lr=0.001)
    self.net.train()
    for epoch in range(self.no_of_epoch):
        running_loss = 0.0
        for i in range(len(self.train_data)):
            labels=self.train_labels[i]
            if epoch>1:
                data=sk.util.random_noise(self.train_data[i]) #data augmentation to improve accuracy
            else:
                data = self.train_data[i]
            #data=data.unsqueeze(0)
            optimizer.zero_grad()
            outputs = self.net(data)
            loss=criterion(outputs,labels.unsqueeze(0))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    torch.save(self.net.state_dict(),'lenet.pth')
    torch.save(self.net,'lenet_all.pth')
    print('Finished Training')

def test_model(self):
    self.net=lenet()
    self.net.load_state_dict(torch.load('lenet.pth'))
    correct=0
    total=len(self.test_data)
    self.net.eval()
    for i in range(len(self.test_data)):
        labels = self.test_labels[i]
        data = self.test_data[i]
        outputs=self.net(data)
        value,predicted_index=torch.max(outputs.data[0],0)
        if predicted_index==labels:
            correct=correct+1
        if i<5:
            plt.title("label=%s" % int(predicted_index))
            plt.imshow(data.reshape(28, 28), cmap='gray')
            plt.show()
    print("Accuracy for %s images=%s" % (total,((float(correct)/float(total))*100)))


class lenet_model:
    def __init__(self):
        self.mnist_trainset=[]
        self.no_of_epoch=2
        self.batch_size=200
        self.no_of_classes = 10
        self.net=lenet()
    get_data_mnist=get_data_mnist
    train_model=train_model
    test_model=test_model

def main():
    mdl=lenet_model()
    mdl.get_data_mnist()
    mdl.train_model()
    mdl.test_model()

if __name__=='__main__':
    main()
