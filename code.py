import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
import time

max_joint = 3

def main():
    

    dataset = CustomDataset()
    testset = Customtestset()

    # print(dataset.x_data)
    # print(dataset.y_data)

    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) 
    nb_epochs = 50

    criterion = nn.MSELoss()

    loss_graph = [] # 그래프 그릴 목적인 loss.

    n=len(dataloader)

    # for epoch in tqdm.tqdm(range(nb_epochs)):
    for epoch in range(nb_epochs):
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
            # loss_graph.append(loss.item()) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader),running_loss/n))
        loss_graph.append(running_loss/n) # 한 epoch에 모든 배치들에 대한 평균 loss 리스트에 담고,
    
    plt.figure()
    plt.plot(np.linspace(0,1,len(loss_graph)), loss_graph)
    # plt.show()


    test_resx = []
    test_resy = []
    for data in testset.x_data:
        res = torch.detach(model(torch.FloatTensor(data))).numpy()
        test_resx.append(res[0] * max_joint)
        test_resy.append(res[1] * max_joint)

    correct_x = []
    correct_y = []
    for data in testset.y_data:
        # H(x) 계산
        correct_x.append(data[0] * max_joint)
        correct_y.append(data[1] * max_joint)

    error_graphx = []
    error_graphy = []
    error_hue = []
    for idx in range(len(test_resx)):
        x1, y1, x2, y2 = test_resx[idx], test_resy[idx], correct_x[idx], correct_y[idx]
        error_graphx.append(x2-x1)
        error_graphy.append(y2-y1)
        error_hue.append(math.dist([x1,y1],[x2,y2]))
    
    print("average error : ", np.average(error_hue)/math.sqrt(2*max_joint*max_joint) * 100 , "%")
    plt.figure()
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.axis([-0.5, 0.5, -0.5, 0.5])
    # plt.figure(figsize=(10,6))
    plt.scatter(error_graphx,error_graphy,s=0.1,label="error", alpha=1)
    plt.scatter(0,0,s=10,label="correct",c="red")
    plt.legend()
    plt.show()

    
    # new_ang =  [1.5 / (2*math.pi), 1.5 / (2*math.pi)] # -0.459627647	0.569307497
    # new_var =  torch.FloatTensor([new_ang]) # -0.459627647	0.569307497
    
    # start = time.time()
    # pred_y = model(new_var) * max_joint
    # print("예측값 :", pred_y) 
    # print("경과 시간 :", start - time.time()) 

    # start = time.time()
    # recx = math.cos(new_ang[0]*(2*math.pi)) + math.cos(new_ang[0]*(2*math.pi) + new_ang[1]*(2*math.pi))
    # recy = math.sin(new_ang[0]*(2*math.pi)) + math.sin(new_ang[0]*(2*math.pi) + new_ang[1]*(2*math.pi))
    # print("예측값 :", [recx,]) 
    # print("경과 시간 :", start - time.time()) 


# Dataset 상속
class CustomDataset(Dataset): 
    def __init__(self):

        df = pd.read_csv(r'machanism.csv', delimiter=',')
        # print(df.shape)
        self.x_data = df.iloc[:, :-2].values
        self.y_data = df.iloc[:, -2:].values
        print(self.y_data)
        self.x_data = self.x_data / (2*math.pi)
        self.y_data = self.y_data / max_joint

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class Customtestset(Dataset): 
    def __init__(self):

        df = pd.read_csv(r'machanism_test.csv', delimiter=',')
        # print(df.shape)
        self.x_data = df.iloc[:, :-2].values
        self.y_data = df.iloc[:, -2:].values
        self.x_data = self.x_data / (2*math.pi)
        self.y_data = self.y_data / max_joint

    # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)
        # self.fc6 = nn.Linear(200, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


main()