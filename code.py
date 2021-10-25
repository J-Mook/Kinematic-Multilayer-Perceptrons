import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D

max_joint = 1
train_cut=12000

def main():
    

    dataset = CustomDataset()
    testset = Customtestset()

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    nb_epochs = 30

    criterion = nn.MSELoss()

    loss_graph = [] # 그래프 그릴 목적인 loss.

    n=len(dataloader)

    for epoch in range(nb_epochs+1):
        running_loss = 0
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            # H(x) 계산
            prediction = model(x_train)

            # cost 계산
            loss = criterion(prediction, y_train)

            # cost로 H(x) 계산
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('Epoch {:4d}/{} Batch {}/{} loss: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader),running_loss/n))
        loss_graph.append(running_loss/n)
    
    plt.figure()
    plt.plot(np.arange(len(loss_graph)), loss_graph)
    plt.xlabel('epochs')
    plt.ylabel('loss ratio')

    
    test_resx = []
    test_resy = []
    test_res=[]
    for data in testset.x_data:
        res = torch.detach(model(torch.FloatTensor(data))).numpy()
        
        test_res.append(res)
        # test_resx.append(res[0])
        # test_resy.append(res[1])

    correct_x = []
    correct_y = []
    correct = []
    for data in testset.y_data:
        correct.append(data)
        # correct_x.append(data[0])
        # correct_y.append(data[1])

    error_graphx = []
    error_graphy = []
    error_graphz = []
    error_hue = []
    for idx in range(len(test_res)):
        # x1, y1, x2, y2 = test_resx[idx], test_resy[idx], correct_x[idx], correct_y[idx]
        # error_graphx.append(x2-x1)
        # error_graphy.append(y2-y1)
        x1, y1, z1, x2, y2, z2 = test_res[idx][0], test_res[idx][1], test_res[idx][2], correct[idx][0], correct[idx][1], correct[idx][2]
        error_graphx.append((x2-x1)*1000)
        error_graphy.append((y2-y1)*1000)
        error_graphz.append((z2-z1)*1000)
        error_hue.append(math.dist([x1, y1, z1], [x2, y2, z2]))
    print("average error : ", np.average(error_hue)/math.sqrt(3*max_joint*max_joint) * 100 , "%")
    print(np.average(error_hue))
    print(np.average(np.abs(error_graphx)))
    print(np.average(np.abs(error_graphy)))
    print(np.average(np.abs(error_graphz)))
    # plt.figure()
    # plt.axhline(0, color='black')
    # plt.axvline(0, color='black')
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    # plt.axis([-0.5, 0.5, -0.5, 0.5])
    # plt.scatter(error_graphx,error_graphy,s=0.1,label="error", alpha=1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)
    ax.set_xlabel('x error(mm)')
    ax.set_ylabel('y error(mm)')
    ax.set_zlabel('z error(mm)')
    ax.scatter(error_graphx, error_graphy, error_graphz, marker='.', s=5, alpha = 0.8, label = "test result")
    ax.scatter(0,0,0,s=20,label="correct",c="red")
    plt.legend()
    plt.show()

    
class CustomDataset(Dataset): 
    def __init__(self):

        # df = pd.read_csv(r'machanism.csv', delimiter=',')
        df = pd.read_csv(r'robot_inverse_kinematics_dataset.csv', delimiter=',')
        self.x_data = df.iloc[:train_cut, :-3].values
        self.y_data = df.iloc[:train_cut, -3:].values
        self.x_data = self.x_data
        self.y_data = self.y_data

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class Customtestset(Dataset): 
    def __init__(self):

        # df = pd.read_csv(r'machanism_test.csv', delimiter=',')
        df = pd.read_csv(r'robot_inverse_kinematics_dataset.csv', delimiter=',')
        self.x_data = df.iloc[train_cut:, :-3].values
        self.y_data = df.iloc[train_cut:, -3:].values
        self.x_data = self.x_data
        self.y_data = self.y_data

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fcf = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fcf(x)

        return x

main()