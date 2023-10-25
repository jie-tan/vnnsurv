import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
import vnn.utils_biovnn as ub
from torch.utils.data import TensorDataset, DataLoader
from utils import VNNSurv3,evaluate3
from utils import DeepCox_LossFunc

# 导入特征重要性排序
sorted_index = np.load('./sorted_index.npy')
num_feats = 30
sub_idx = sorted_index[:num_feats]

temp = pd.read_excel(io='./Supplementary2.xlsx',sheet_name=1,index_col=0)
info = pd.DataFrame(temp.columns)
info = info.iloc[:,0].str.split('_',expand=True)
genelist = info.iloc[:,0].values[:-3]
genelist = genelist[sub_idx]

###############################################
# construct biovnn
###############################################
result_dir = './vnn/'

genelist = list(genelist)
biovnn_pre = ub.BioVNN_pre(genelist, result_dir)
biovnn_pre.perform() # 需要的数据保存为pkl

###############################################
# load features
###############################################
features = pd.read_excel(io='./Supplementary2.xlsx',sheet_name=1,index_col=0).values
features = features[:, np.append(sub_idx,[117,118,119])]

info2 = pd.read_excel(io='./Supplementary2.xlsx',sheet_name=0)
ostime = info2.loc[:,'OS_time'].values
status = info2.loc[:,'OS_status'].values
ostime[status==0] = -ostime[status==0]   #status=0为删失，存活或失访

epochs = 10
batchsize = 1024 #batchsize小泛化能力更强，128看起来效果最好
learning_rate = 5e-3
learning_rate_decay = 1e-5
scheduler_gamma = 0.5
outputdim_vnn = 128
dropout_vnn = 0.1 #30个特征时改为0.2
fc_dim = int(outputdim_vnn/2)
dropout_fc = 0.3

gpu = 0
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
print(device)


test_save_set = []  # 保存参数和测试集结果
test_save_set.append([" batchsize: " + str(batchsize) + " learning rate: " + str(learning_rate) +
                      " decay: " + str(learning_rate_decay) +" gamma: " + str(scheduler_gamma) +
                      " outputdim_vnn: " + str(outputdim_vnn) + " dropout_vnn: "+ str(dropout_vnn) +
                      " fc_dim: " + str(fc_dim) + " dropout_fc: " + str(dropout_fc)])


train_data = features
train_time = ostime

train_data = torch.from_numpy(train_data).type(torch.float32)
train_time = torch.from_numpy(train_time).type(torch.float32)
train_dataset = TensorDataset(train_data, train_time)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)


# num_feature = 117
num_feature = 30
model = VNNSurv3(num_feature, result_dir,outputdim_vnn, dropout_vnn, fc_dim, dropout_fc)
loss_func = DeepCox_LossFunc()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 1000], gamma=scheduler_gamma)

model = model.to(device)
train_Cindex_list, train_curve = [], []
test_Cindex_list, test_curve = [], []

for i in range(epochs):
    # train
    model.train()
    # risks = torch.zeros([train_samples, 1], dtype=torch.float)
    total_loss = 0
    risks = []
    times = []
    for id, data in enumerate(train_dataloader):
        model.train()
        input, time = data
        input, time = input.to(device), time.to(device)
        risk = model(input)
        risks.extend(list(risk.detach().cpu().numpy()))
        times.extend(time.cpu().numpy())
        loss = loss_func(risk, time)
        # print(loss)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

train_Cindex, train_risks = evaluate3(model, train_dataloader, device)

# torch.save(model,'./model/vnnsurv.pt')
torch.save(model,'./model/vnnsurv-top30.pt')

