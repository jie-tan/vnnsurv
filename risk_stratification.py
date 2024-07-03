import os
import sys
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description='GPI: genetic-based prognostic index')
parser.add_argument('-i', dest='input', nargs=1, required=True,
                    help='the path of input file')
parser.add_argument('-o', dest='output', nargs=1, default=[os.getcwd()],
                    help='output directory. The default is the current path')
args = parser.parse_args()

def load_data():
    print("Loading input...")
    input = os.path.abspath(os.path.expanduser(args.input[0]))
    # input = '/home/tanshaoshao/Work/prognosis model/GraphVNN/web/example/input_example_top30_gpi.xlsx'
    if os.path.isfile(input):
        features = pd.read_excel(io=input,sheet_name=0,index_col=0).values
    else:
        sys.exit('Input error: no such directory or file ')
    return features

if __name__ == '__main__':
    features = load_data()
    weight = np.load('/home/tanshaoshao/Work/prognosis model/GraphVNN/application928/result/weight_30.npy')
    score = np.sum(features[:,:30] * weight, axis=1)

    label = np.array(['Intermediate']*len(features))
    label[np.where(score>0.025)] = 'High'
    label[np.where(score<-0.015)] = 'Low'

    output_folder = args.output[0]
    np.savetxt(output_folder + '/output_risk_stratification.txt', label, fmt='%s')
    print('Finished')
    # np.savetxt('/home/tanshaoshao/Work/prognosis model/GraphVNN/web/example/output_risk_stratification.txt',label,fmt='%s')