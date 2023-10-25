import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse

parser = argparse.ArgumentParser(
    description='VNNSurv: an interpretable survival model for DLBCL')
parser.add_argument('-i', dest='input', nargs=1, required=True,
                    help='the path of input file')
parser.add_argument('-m', dest='model', nargs=1, required=True,
                    help='the name of the model to use')
parser.add_argument('-o', dest='output', nargs=1, default=[os.getcwd()],
                    help='output directory. The default is the current path')
parser.add_argument('-g', dest='gpu', type=int, nargs=1, default=0,
                    help='the index of gpu device')
args = parser.parse_args()

##############################################################################################################
# Load data
def load_data():
    print("Loading input...")
    input = os.path.abspath(os.path.expanduser(args.input[0]))
    # input = '/home/tanshaoshao/Work/prognosis model/GraphVNN/github/example/input.xlsx'
    if os.path.isfile(input):
        Features = pd.read_excel(io=input,sheet_name=0,index_col=0).values
    else:
        sys.exit('Input error: no such directory or file ')
    return Features

# Load model
def load_model():
    print("Loading model...")
    m = args.model[0]
    # m = 'vnnsurv'
    if m+'.pt' in os.listdir('./model/'):
        model = torch.load('./model/'+m+'.pt')
    else:
        sys.exit('Model error: no such model')
    return model

# predict
def pred(model, dataloader, device):
    model.eval()
    risks_save = []
    with torch.no_grad():
        for id, data in enumerate(dataloader):
            input = data[0]
            input = input.to(device)
            risks = model(input)
        risks_save.extend(list(risks.detach().cpu().numpy()))
    return risks_save

if __name__ == '__main__':
    Features = load_data()
    model = load_model()
    output_folder = args.output[0]
    gpu = args.gpu[0]
    # output_folder = 'example/'
    # gpu = 0
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    data = torch.from_numpy(Features).type(torch.float32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

    risks_save = pred(model, dataloader, device)

    np.savetxt(output_folder + 'output_risks.txt', risks_save)