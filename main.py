
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms, utils
from src.DataLoader import AscadDataLoader_train, AscadDataLoader_test, AscadDataLoader_validation
from src.net import Net
from src.Preprocessing import Horizontal_Scaling_0_1, ToTensor, Horizontal_Scaling_m1_1
from src.config import Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore',category=FutureWarning)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# initiate the parser


config = Config()

#Seed the pipeline for reproductibility
seed = config.general.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Incorporate the prepocessing of the Github
compose = transforms.Compose([  ToTensor() ])
Horizontal_scale_0_1 = transforms.Compose([  ToTensor(), Horizontal_Scaling_0_1() ])
Horizontal_scale_m1_1 = transforms.Compose([  ToTensor(), Horizontal_Scaling_m1_1() ])
#LOAD trainset
trainset = -1
valset = -1
if config.dataloader.scaling == "None":
    trainset = AscadDataLoader_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    valset = AscadDataLoader_validation(config, X_validation, Y_validation)

elif config.dataloader.scaling == "horizontal_scale_0_1":
    trainset = AscadDataLoader_train(config, transform=Horizontal_scale_0_1)
    X_validation, Y_validation = trainset.train_validation_split()
    valset = AscadDataLoader_validation(config, X_validation, Y_validation)

elif config.dataloader.scaling == "horizontal_scale_m1_1":
    trainset = AscadDataLoader_train(config, transform=Horizontal_scale_m1_1)
    X_validation, Y_validation = trainset.train_validation_split()
    valset = AscadDataLoader_validation(config, X_validation, Y_validation)

elif config.dataloader.scaling == "feature_scaling_0_1":
    trainset = AscadDataLoader_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    trainset.feature_min_max_scaling(0,1)
    valset = AscadDataLoader_validation(config, X_validation, Y_validation, feature_scaler=trainset.get_feature_scaler())

elif config.dataloader.scaling == "feature_scaling_m1_1":
    trainset = AscadDataLoader_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    trainset.feature_min_max_scaling(-1,1)
    valset = AscadDataLoader_validation(config, X_validation, Y_validation,feature_scaler=trainset.get_feature_scaler())

elif config.dataloader.scaling == "feature_standardization":
    trainset = AscadDataLoader_train(config, transform=compose)
    X_validation, Y_validation = trainset.train_validation_split()
    trainset.feature_standardization()
    valset = AscadDataLoader_validation(config, X_validation, Y_validation, feature_scaler=trainset.get_feature_scaler())

valset.feature_scaling()





#trainset.to_categorical(num_classes=256)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=config.dataloader.batch_size,
                                          shuffle=config.dataloader.shuffle,
                                          num_workers=config.dataloader.num_workers)

dataloader = {"train": trainloader, "val": valloader}

testset = -1
if config.dataloader.scaling == "horizontal_scale_0_1":
    testset = AscadDataLoader_test(config, transform=Horizontal_scale_0_1)
elif config.dataloader.scaling == "horizontal_scale_m1_1":
    testset = AscadDataLoader_test(config, transform=Horizontal_scale_m1_1)
else:
    scaler = trainset.get_feature_scaler()
    testset= AscadDataLoader_test(config, transform=compose, feature_scaler=scaler)
    testset.feature_scaling()

#testset.to_categorical(num_classes=256)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_dataloader.batch_size,
                                         shuffle=config.test_dataloader.shuffle,
                                        num_workers=config.test_dataloader.num_workers)
# print("Trainset:")
# for i in range(len(trainset)):
# #     print(trainset[i])
#      print(trainset[i]["sensitive"])

# print("Testset:")
# for i in range(len(testset)):
#     print("testset " + str(i))
#     print(testset[i]["sensitive"])

#
writer = SummaryWriter("runs/noConv1D_ascad_desync_50_3")
#TODO: Change the model for the one of the paper
net = Net()

#TODO: propose NLLloss (Categorical Cross Entropy), Adam optimizer and Cyclic Learning Rate
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=float(config.train.lr))


scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = float(config.train.lr), epochs = 50, steps_per_epoch=len(trainloader))
# TODO: plot in tensorboard the curves loss and accuracy for train and val
for epoch in range(config.train.epochs):  # loop over the dataset multiple times
    print('Epoch {}/{}'.format(epoch+1, config.train.epochs))
    print('-' * 10)
    for phase in ["train", "val"]:
        if phase == "train":
            net.train()
        else:
            net.eval()

        running_loss = 0.0

        for i, data in enumerate(dataloader[phase], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["trace"].float(), data["sensitive"].float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(phase == "train"):
                outputs = net(inputs)
                labels = labels.view(int(config.dataloader.batch_size)).long() ##This is because NLLLoss only take in this form.
                outputs = torch.log(outputs)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # print statistics
            running_loss += loss.item()

        ## Update the learning rate.
        if phase == "train":
           scheduler.step()

        epoch_loss = running_loss / len(dataloader[phase])
        if phase == "train":
            writer.add_scalar('training loss', epoch_loss, epoch * len(dataloader["train"]))
        elif phase == "val":
            writer.add_scalar('val loss', epoch_loss, epoch * len(dataloader["val"]))

        print('{} Epoch Loss: {:.4f}'.format(phase, epoch_loss))



print('Finished Training')




#Saving trained model and loading model.
PATH = './model/noConv1D_ascad_desync_50_3.pth'
torch.save(net.state_dict(), PATH)
net = Net()
net.load_state_dict(torch.load(PATH))


#Evaluating the model based on the entropy metric
dataiter = iter(testloader)
sample = dataiter.next()
attack_traces, target_labels   = sample["trace"].float(), sample["sensitive"].float()
predictions = net(attack_traces)
predictions = torch.log(torch.add(predictions,1e-40))




## Rank of the keys
nattack =100
ntraces = 400
interval = 1
ranks = np.zeros((nattack , int(ntraces/interval)))

for i in tqdm(range(nattack)):
    ranks[i] = testset.rank(predictions, ntraces, interval)
ranklist = np.mean(ranks, axis=0)
print(ranklist)
#Plotting the graph of number of traces over rank of the key for the real key.
fig, ax = plt.subplots(figsize=(15, 7))
x = [x for x in range(0,ntraces, interval)]
ax.plot(x, np.mean(ranks, axis=0), 'b')
ax.set(xlabel='Number of traces', ylabel='Mean rank')
plt.show()



writer.close()
#
