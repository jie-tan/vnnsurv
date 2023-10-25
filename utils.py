import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from vnn.VNN_cell import VNN_cell
import pickle
from lifelines.utils import concordance_index as ci

class VNNSurv3(nn.Module):
    def __init__(self, num_feature,result_dir,outputdim_vnn, dropout_vnn,fc_dim,dropout_fc):
        super(VNNSurv3, self).__init__()
        torch.manual_seed(1234)
        torch.set_printoptions(profile="full")
        # self.args = args
        self.num_feature = num_feature
        self.cell_dim = outputdim_vnn
        self.dropout_vnn = dropout_vnn
        self.fc_dim = fc_dim
        self.dropout_ratio = dropout_fc
        self.result_dir = result_dir
        # self.run_mode = args.biovnn_run_mode
        self.neuron_ratio = 1
        self.use_average_neuron_n = False
        self.only_combine_child_gene_group = False

        with open(self.result_dir+"/BioVNN_pre.pkl","rb") as f:
            self.biovnn_dict=pickle.load(f)
        self.VNN = VNN_cell(omic_dim=1,
                            input_dim=self.num_feature,  # gene num
                             output_dim=self.cell_dim,
                             biovnn_dict=self.biovnn_dict,
                             only_combine_child_gene_group=self.only_combine_child_gene_group,
                             neuron_ratio=self.neuron_ratio,
                             use_average_neuron_n=self.use_average_neuron_n,
                             dropout_p=self.dropout_vnn)
        self.regression = nn.Sequential(
            nn.Linear(self.cell_dim+3, self.fc_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(self.fc_dim, 1))

    def forward(self, x):
        sup = x[:,-3:]
        x = x[:,:-3]
        x = x.unsqueeze(1)
        x = x.permute(2, 0, 1)
        x = [x[i] for i in range(len(x))]
        VNN_out = self.VNN(x)
        x = torch.concat((VNN_out, sup),1)
        risk = self.regression(x)

        return risk


class DeepCox_LossFunc(torch.nn.Module):
    def __init__(self):
        super(DeepCox_LossFunc, self).__init__()
    def forward(self,y_predict,t):
        t = torch.tensor(t)
        y_pred_list = y_predict.view(-1)
        y_pred_exp = torch.exp(y_pred_list)
        t_list = t.view(-1)
        t_E = torch.gt(t_list,0)
        y_pred_cumsum = torch.cumsum(y_pred_exp, dim=0)
        y_pred_cumsum_log = torch.log(y_pred_cumsum)
        loss1 = -torch.sum(y_pred_list.mul(t_E))
        loss2 = torch.sum(y_pred_cumsum_log.mul(t_E))
        loss = (loss1 + loss2)/torch.sum(t_E)
        return loss


def concordance_index(y_true, y_pred):
    """
    Compute the concordance-index value.

    Parameters
    ----------
    y_true : np.array
        Observed time. Negtive values are considered right censored.
    y_pred : np.array
        Predicted value.

    Returns
    -------
    float
        Concordance index.
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int32)
    ci_value = ci(t, y_pred, e)
    return ci_value

def evaluate3(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for id, data in enumerate(dataloader):
            input, time = data
            input, time = input.to(device), time.to(device)
            risks = model(input)
        risks_save = risks.detach()
        if time.device.type=='cuda':
            time = time.cpu()
        cindex = concordance_index(time.numpy(), -risks_save.cpu().numpy())
    return cindex, risks_save