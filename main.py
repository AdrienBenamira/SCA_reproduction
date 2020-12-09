
import warnings
from torchvision import transforms, utils
from src.DataLoader import AscadDataLoader_train, AscadDataLoader_test

from src.net import Net
import torch
import torch.nn as nn
import torch.optim as optim

from src.Preprocessing import Horizontal_Scaling_0_1, ToTensor, Horizontal_Scaling_m1_1

warnings.filterwarnings('ignore',category=FutureWarning)
from src.config import Config

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser


config = Config()

#TODO: seed the pipeline for reproductibility

#TODO: incorporate the prepocessing of the Github
compose = transforms.Compose([  ToTensor() ])
Horizontal_scale_0_1 = transforms.Compose([  ToTensor(), Horizontal_Scaling_0_1() ])
Horizontal_scale_m1_1 = transforms.Compose([  ToTensor(), Horizontal_Scaling_m1_1() ])
#LOAD trainset
trainset = -1

if config.dataloader.scaling == "None":
    trainset = AscadDataLoader_train(config, transform=compose)

elif config.dataloader.scaling == "horizontal_scale_0_1":
    trainset = AscadDataLoader_train(config, transform=Horizontal_scale_0_1)

elif config.dataloader.scaling == "horizontal_scale_m1_1":
    trainset = AscadDataLoader_train(config, transform=Horizontal_scale_m1_1)

elif config.dataloader.scaling == "feature_scaling_0_1":
    trainset = AscadDataLoader_train(config, transform=compose)
    trainset.feature_min_max_scaling(0,1)

elif config.dataloader.scaling == "feature_scaling_m1_1":
    trainset = AscadDataLoader_train(config, transform=compose)
    trainset.feature_min_max_scaling(-1,1)

elif config.dataloader.scaling == "feature_standardization":
    trainset = AscadDataLoader_train(config, transform=compose)
    trainset.feature_standardization()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)

if config.dataloader.scaling == "horizontal_scale_0_1":
    testset = AscadDataLoader_test(config, transform=Horizontal_scale_0_1)
elif config.dataloader.scaling == "horizontal_scale_m1_1":
    testset = AscadDataLoader_test(config, transform=Horizontal_scale_m1_1)
else:
    scaler = trainset.get_feature_scaler()
    testset= AscadDataLoader_test(config, transform=compose, feature_scaler=scaler)
    testset.feature_scaling()

testloader = torch.utils.data.DataLoader(testset, batch_size=config.dataloader.batch_size,
                                         shuffle=config.dataloader.shuffle,
                                        num_workers=config.dataloader.num_workers)
# print("Trainset:")
# for i in range(len(trainset)):
#     print(trainset[i])

print("Testset:")
for i in range(len(testset)):
    print("testset " + str(i))
    print(testset[i])



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
        for trace in inputs:
            outputs = net(trace)
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

