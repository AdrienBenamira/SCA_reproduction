
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms, utils
from src.DataLoader import AscadDataLoader_train, AscadDataLoader_test
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
testset = -1
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

#trainset.to_categorical(num_classes=256)

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


writer = SummaryWriter("runs/noConv1D_ascad_desync_50_2")
# #TODO: Change the model for the one of the paper
# net = Net()
#
# #TODO: propose multiple loss and optimizer
# criterion = nn.NLLLoss()
# optimizer = optim.Adam(net.parameters(), lr=float(config.train.lr))
#
# # TODO: plot in tensorboard the curves loss and accuracy for train and val
# for epoch in range(config.train.epochs):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data["trace"].float(), data["sensitive"].float()
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         labels = labels.view(int(config.dataloader.batch_size)).long() ##This is because NLLLoss only take in this form.
#         outputs = torch.log(outputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         # print statistics
#         running_loss += loss.item()
#
#         if i % 1000 == 999:    # print every 1000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 1000))
#             writer.add_scalar('training loss',
#                               running_loss / 1000,
#                               epoch * len(trainloader) + i)
#
#             running_loss = 0.0
#
# print('Finished Training')
#



#Saving trained model and loading model.
PATH = './model/noConv1D_ascad_desync_50_1.pth'
#torch.save(net.state_dict(), PATH)
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
ntraces = 300
interval = 1
ranks = np.zeros((nattack , int(ntraces/interval)))

for i in tqdm(range(nattack)):
    ranks[i] = testset.rank(predictions, ntraces, interval)

#Plotting the graph of number of traces over rank of the key for the real key.
fig, ax = plt.subplots(figsize=(15, 7))
x = [x for x in range(0,ntraces, interval)]
ax.plot(x, np.mean(ranks, axis=0), 'b')
ax.set(xlabel='Number of traces', ylabel='Mean rank')
plt.show()

writer.close()

