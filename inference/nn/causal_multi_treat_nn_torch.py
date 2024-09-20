import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.system('echo $CUDA_VISIBLE_DEVICES')
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from Encoding import load_feature
 
class My_Multi_Treat_base_Net_torch(nn.Module):
    def __init__(self,batch_size, num_layers=3, num_class=2):
        super(My_Multi_Treat_base_Net_torch,self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_class = num_class
        self.Linearbase = nn.Sequential(nn.Linear(batch_size, 200),nn.Dropout(0.7))
        if self.num_class == 2:
            self.output = nn.Sequential(nn.Linear(batch_size, 150),nn.Dropout(0.8),nn.Linear(150, 1))
        else:
            self.output = nn.Sequential(nn.Linear(batch_size, 150),nn.Dropout(0.8),nn.Linear(150, num_class))


    def forward(self, x, t, control_name=0):

        treat = np.array(t)
        torch_train = torch.cat([torch.tensor(x[feature].values),torch.from_numpy(treat.reshape(-1,1))],dim=1)

        # torch_t = torch.where(t==0, torch.zeros_like(t),
        #              torch.where(t==1, torch.ones_like(t),
        #              torch.where(t==2, torch.ones_like(t)*2,
        #              torch.where(t==3, torch.ones_like(t)*3,torch.ones_like(t)*4
        #             ))))

        xc = torch_train[(torch_train[:,-1]==0)]
        x1 = torch_train[(torch_train[:,-1]==1)]
        x2 = torch_train[(torch_train[:,-1]==2)]
        x3 = torch_train[(torch_train[:,-1]==3)]
        x4 = torch_train[(torch_train[:,-1]==4)]

        for i in self.num_layers:
            xc = F.relu(self.Linearbase(xc))
            x1 = F.relu(self.Linearbase(x1))
            x2 = F.relu(self.Linearbase(x2))
            x3 = F.relu(self.Linearbase(x3))
            x4 = F.relu(self.Linearbase(x4))
        
        output_c = F.sigmoid(self.output(xc))
        output1 = F.sigmoid(self.output(x1))
        output2 = F.sigmoid(self.output(x2))
        output3 = F.sigmoid(self.output(x3))
        output4 = F.sigmoid(self.output(x4))
        return [output_c, output1, output2, output3, output4]

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss,self).__init__()
    def forward(self, pred, true, if_binary=True):
        n = len(pred)
        diff=0
        for i in n:
            pos_i = torch.eq(true[i], 1).float()
            neg_i= torch.eq(true[i], 0).float()
            num_pos_i = torch.sum(pos_i)
            num_neg_i = torch.sum(neg_i)
            num_total_i = num_pos_i + num_neg_i
            alpha_pos_i = num_neg_i / num_total_i
            alpha_neg_i = num_pos_i / num_total_i
            weights_i = alpha_pos_i * pos_i + alpha_neg_i * neg_i
            if if_binary:
                diff_i =  F.binary_cross_entropy_with_logits(pred[i], true[i], weights_i, reduction = 'sum')
                diff += diff_i
        return diff

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

class My_Multi_Treat_Model:
    def train(model, train_loader, myloss, optimizer, epoch):
        model.train()
        for batch_idx, (train_data,train_y, train_treat) in enumerate(train_loader):
            x = Variable(train_data,requires_grad=True) #Varibale 默认时不要求梯度的，如果要求梯度，需要说明
            t = Variable(train_treat,requires_grad=True)
            y = Variable(train_y,requires_grad=True)

            optimizer.zero_grad()
            output = model(x,t)
            loss = myloss(output, y)
            loss.backward()
            optimizer.step()
            if batch_idx%100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                    epoch, batch_idx*len(train_data), len(train_loader.dataset),
                    100.*batch_idx/len(train_loader), loss.data.cpu().numpy()[0]))




 

 
