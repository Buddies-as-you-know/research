import numpy as np
import math
from sklearn.manifold import TSNE
from matplotlib import pyplot
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
#import tqdm
from torch.autograd import Variable
def preprocessing(floatArray):
    window = np.hamming(125)
    w = np.empty(125)
    d2 = np.empty((floatArray.shape[0],27*16))
    for i in range(floatArray.shape[0]-125):
        for j in range(15):
            w = np.abs(np.fft.fftn(floatArray[i:i+125,j]*window))
            d2[i,27*j:27*j+26] = np.log10(1 + w[4:30])
    return d2
def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg)
def labeler(array): # ラベル定義
  label = np.empty(125*7)
  for i in range(125*7):
    if i <= 125:
      label[i] = 0 #0 is normal
    elif i <= 125*2:
      label[i] = 1 #1is forward
    elif i <= 125*3:
      label[i] = 2 # 2 is righet
    elif i <= 125*4:
      label[i] = 3
    elif i <= 125*5:
      label[i] = 4
    elif i <= 125*6:
      label[i] = 5
    else:
      label[i] = 6
  return label
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60,kernel_size=(1, 15),stride=(1,3))
        self.conv2 = nn.Conv2d(60, 60, kernel_size=(1, 4),stride=(1,2))
        self.conv3 = nn.Conv2d(60, 60, kernel_size=(30,1),stride=(1,3))
        self.conv4 = nn.Conv2d(60, 90, kernel_size=(1, 3),stride=(1,1))
        self.conv5 = nn.Conv2d(90, 120, kernel_size=(1, 1),stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2),stride=(1,2))
        #lf.soft = nn.Softmax(dim = 1)
        self.fc=nn.Linear(2520,7)

    def forward(self, x):
      x = self.conv1(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.conv2(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.conv3(x)
      #print(x.shape)
      x = self.conv4(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.conv5(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = x.view(7,2520)
      x = self.fc(x)
      #print(x.shape)
      #x = self.soft(x)
      return x

if __name__ == "__main__":
    data = np.loadtxt("C:\\Users\\owner\\Desktop\\EEGdata\\OpenBCI-RAW-2022-04-19_15-41-40.txt",dtype='str',delimiter=",",skiprows=5)
    b = data[:,1:17]
    floatArray = b.astype(float)
    d2 = np.empty((floatArray.shape[0],27*16))
    d2 = preprocessing(floatArray)
    ts = d2[0:125*7*50,:]
    ts=torch.Tensor(ts)
    ts = ts.view(125*7,50,432)
    label = labeler(ts)
    train_data, test_data, train_label, test_label = train_test_split(ts, label, test_size=0.2,shuffle = True)
    train_x = torch.Tensor(train_data)
    test_x = torch.Tensor(test_data)
    #train_y = torch.LongTensor(train_label)  # torch.int64のデータ型に
    #test_y = torch.LongTensor(test_label)
    train_y = torch.Tensor(train_label)
    test_y = torch.Tensor(test_label)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True)
    val_loader = DataLoader(test_dataset,batch_size=7,shuffle=False)
    net = Net()

# loss関数の定義
    criterion = nn.CrossEntropyLoss()
    using_cuda = torch.cuda.is_available()
    accuracies = []
# 最適化関数の定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):
      running_loss = 0.0
      for i, (inputs, labels) in enumerate(train_loader, 0):
        # zero the parameter gradients
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.view(7,1,50,432)
        #print(inputs.shape)
        optimizer.zero_grad()
        # forward + backward + optimiz
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[{:d}, {:5d}] loss: {:.12f}'
                    .format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    print('Finished Training')

    # 次回から学習しなくていいように、学習済みモデルのパラメータを"net_00.prmとして保存"
    params = net.state_dict()
    torch.save(params, "net_00.prm", pickle_protocol=4)

    print('Finished Training')

    ### 学習済みモデルのテスト ###
    test_total_acc = 0
    net.eval()
#net_path = 'model.pth'
#net.load_state_dict(torch.load(net_path))
    pred_list = []
    true_list = []
with torch.no_grad():
    for n,(data,label) in enumerate(val_loader):
        data = data.to(device)
        label = label.to(device)
        data = data.view(7,1,50,432)
        output = net(data)
        test_total_acc += cal_acc(label.long(),output)
        pred = torch.argmax(output , dim =1)
        pred_list += pred.detach().cpu().numpy().tolist()
        true_list += label.detach().cpu().numpy().tolist() 
print(f"test acc:{test_total_acc/len(test_dataset)*100}")
datas = np.loadtxt("C:\\Users\\owner\\Desktop\\EEGdata\\OpenBCI-RAW-2022-04-19_14-51-56.txt",dtype='str',delimiter=",",skiprows=5)
bs = datas[:,1:17]
floatArrays = bs.astype(float)
p = np.empty((floatArray.shape[0],27*16))
p = preprocessing(floatArrays)
tss = p[:10500,:]
tss=torch.Tensor(tss)
tss = tss.view(210,50,432)
labels = np.empty(210)
for i in range(210):
    if i <= 100:
      labels[i] = 0 #0 is normal
    else :
      labels[i] = 1 #1is forwar
pred_list = []
true_list = []
test_total_acc = 0
pp_x = torch.Tensor(tss)
pp_y = torch.Tensor(labels)
pp_dataset = TensorDataset(pp_x, pp_y)
pp_loader = DataLoader(pp_dataset, batch_size=7, shuffle=False)
with torch.no_grad():
    for n,(data,label) in enumerate(pp_loader):
        data = data.to(device)
        label = label.to(device)
        data = data.view(7,1,50,432)
        output = net(data)
        test_total_acc += cal_acc(label.long(),output)
        pred = torch.argmax(output , dim =1)
        print(pred_list)
        pred_list += pred.long().detach().cpu().numpy().tolist()
        true_list += label.long().detach().cpu().numpy().tolist() 
        print(true_list)
print(f"test acc:{test_total_acc/len(test_dataset)*100}")