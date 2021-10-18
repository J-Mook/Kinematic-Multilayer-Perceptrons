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

def main():

    dataset = CustomDataset()

    print(dataset.x_data)
    print(dataset.y_data)

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) 
    nb_epochs = 100

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
    # plt.figure(figsize=(10,6))
    plt.plot(np.linspace(0,1,len(loss_graph)), loss_graph)
  
    plt.show()

    new_var =  torch.FloatTensor([[1.5 / (2*math.pi), 1.5 / (2*math.pi)]]) # -0.459627647	0.569307497
    pred_y = model(new_var) * 2
    print("훈련 후 입력이 250, 180, 1.5, 1.5일 때의 예측값 :", pred_y) 

# Dataset 상속
class CustomDataset(Dataset): 
    def __init__(self):

        df = pd.read_csv(r'machanism.csv', delimiter=',')
        print(df.shape)
        self.x_data = df.iloc[:, 2:-2].values
        self.y_data = df.iloc[:, -2:].values
        print(self.x_data)
        self.x_data = self.x_data / (2*math.pi)
        self.y_data = self.y_data / 2
        print(self.x_data)



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
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        # self.fc4 = nn.Linear(64, 2)
        # self.fc5 = nn.Linear(8, 1)
        # self.fc6 = nn.Linear(200, 2)


    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc4(x))
        # x = F.tanh(self.fc5(x))
        # x = F.sigmoid(self.fc6(x))

        return x


main()