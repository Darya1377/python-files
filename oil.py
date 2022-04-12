import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv('production.csv')

gas = df.groupby('API')['Gas'].apply(lambda df_: df_.reset_index(drop=True))
water = df.groupby('API')['Water'].apply(lambda df_: df_.reset_index(drop=True))
liquid = df.groupby('API')['Liquid'].apply(lambda df_: df_.reset_index(drop=True))

df_prod = gas.unstack()
df_prod2 = water.unstack()
df_prod3 = liquid.unstack()
#df = pd.merge(df_prod, df_prod2, how='outer')
#df2 = pd.merge(df_prod3, df, how='outer')
#print(df2)
data = df_prod.values
data = data / data.max()
data = data[:, :, np.newaxis]

data2 = df_prod2.values
data2 = data2 / data2.max()
data2 = data2[:, :, np.newaxis]

data3 = df_prod3.values
data3 = data3 / data3.max()
data3 = data3[:, :, np.newaxis]


data_tr_gas = data[:40]
data_tr_water = data2[:40]
data_tr_liquid = data3[:40]

data_tst_gas = data[40:]
data_tst_water = data2[40:]
data_tst_liquid = data3[40:]
arr3 = np.concatenate([data_tst_gas, data_tst_water], axis=2)
data_tst = np.concatenate([arr3, data_tst_liquid], axis=2)

x_data_liquid = [data_tr_liquid[:, i:i+12] for i in range(11)]
y_data_liquid = [data_tr_liquid[:, i+1:i+13] for i in range(11)]

x_data_water = [data_tr_water[:, i:i+12] for i in range(11)]
y_data_water = [data_tr_water[:, i+1:i+13] for i in range(11)]

x_data_gas = [data_tr_gas[:, i:i+12] for i in range(11)]
y_data_gas = [data_tr_gas[:, i+1:i+13] for i in range(11)]

x_data_liquid = np.concatenate(x_data_liquid, axis=0)
y_data_liquid = np.concatenate(y_data_liquid, axis=0)

x_data_water = np.concatenate(x_data_water, axis=0)
y_data_water = np.concatenate(y_data_water, axis=0)

x_data_gas = np.concatenate(x_data_gas, axis=0)
y_data_gas = np.concatenate(y_data_gas, axis=0)

arr = np.concatenate([x_data_liquid, x_data_water], axis=2)
x_data = np.concatenate([arr, x_data_gas], axis=2)

arr2 = np.concatenate([y_data_liquid, y_data_water], axis=2)
y_data = np.concatenate([arr2, y_data_gas], axis=2)



tensor_x = torch.Tensor(x_data) # transform to torch tensor
tensor_y = torch.Tensor(y_data)

oil_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
oil_dataloader = DataLoader(oil_dataset, batch_size=16) # create your dataloader

for x_t, y_t in oil_dataloader:
    break


class OilModel(nn.Module):
    def __init__(self, timesteps=12, units=32):
        super().__init__()
        self.lstm1 = nn.LSTM(3, units, 2, batch_first=True)
        self.dense = nn.Linear(units, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        h, _ = self.lstm1(x)
        outs = []
        for i in range(h.shape[0]):
            outs.append(self.relu(self.dense(h[i])))
        out = torch.stack(outs, dim=0)
        return out


model = OilModel()
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()
NUM_EPOCHS = 20

for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    num = 0
    for x_t, y_t in oil_dataloader:
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = model(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        num += 1

    print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss / num:.3f}')

print('Finished Training')

x_tst = data_tst[:, :12]
predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))

for i in range(12):
    x = np.concatenate((x_tst[:, i:], predicts), axis=1)
    x_t = torch.from_numpy(x).float()
    pred = model(x_t).detach().numpy()
    last_pred = pred[:, -1:]  # Нас интересует только последний месяц
    predicts = np.concatenate((predicts, last_pred), axis=1)


plt.figure(figsize=(10, 6))
for iapi in range(4):
    plt.subplot(2, 2, iapi+1)
    plt.plot(np.arange(x_tst.shape[1]), x_tst[iapi, :, 0], label='Actual')
    plt.plot(np.arange(predicts.shape[1])+x_tst.shape[1], predicts[iapi, :, 0], label='Prediction')
    plt.legend()
plt.show()





