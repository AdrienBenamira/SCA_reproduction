
import warnings
from torchvision import transforms, utils
from src.DataLoader import AscadDataLoader_train, AscadDataLoader_test
from src.Preprocessing import ToTensor
from src.net import Net
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings('ignore',category=FutureWarning)
from src.config import Config

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser


config = Config()

#TODO: seed the pipeline for reproductibility

#TODO: incorporate the prepocessing of the Github
compose = transforms.Compose([  ToTensor() ])
#TODO:

#LOAD trainset
trainset = AscadDataLoader_train(config, transform=compose)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)
print(len(trainset))
print(len(trainloader))

testset = AscadDataLoader_test(config, transform=compose)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)



#TODO: Change the model for the one of the paper
net = Net()


#TODO: propose multiple loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=config.train.lr, momentum=config.train.momentum)



# TODO: plot in tensorboard the curves loss and accuracy for train and val
for epoch in range(config.train.epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data["trace"].float(), data["sensitive"].float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

