# READ ME
VNNsurv is an interpretable survival model for diffuse large B-cell lymphoma with biologically informed visible neural network. 

## How to use


### Dependencies

The codes of VNNSurv are implemented on Python 3.10. To use it, Python 3.10 together with the packages in `requirements.txt` is required.


We recommend using [Anaconda](https://www.anaconda.com/) to install python and use `conda` or `pip` to install all dependencies. The dependency packages required to run VNNSurv can be installed with this command:

	pip install -r requirements.txt

	
If your machine has a GPU, please configure the corresponding CUDA, CUDNN. Then you can check your CUDA version by `nvidia-smi`, and install pytorch and torchvision [following guidelines on the official website](https://pytorch.org/).



### Data preparation

The input information for VNNSurv consists of structured genetic alteration profiles and three basic clinical variables. Users can refer to the input tables in the example folder for construction. The processing code can be found in `preprocess.py` used for TCGA cohort, and users can modify it according to the characteristics of their own data.


### Usage

	predict.py [-h] -i INPUT -m SELECT_MODEL [-o OUTPUT] [-g GPU]


#### Options

	-h, --help       show this help message and exit
	-i INPUT         the path of input file
	-m SELECT_MODEL  the name of the model to use
	-o OUTPUT        output directory. The default is the current path

#### Example


	python predict.py -i ./example/input.xlsx -o ./example/ -m vnnsurv
	


## Citation
Tan J, Xie JC, Huang JR, Deng WZ, Yang YD*. Interpretable prognosis prediction and subtype identification for diffuse large B-cell lymphoma with biologically informed visible neural network.

## Contact
If you find any bugs or encounter any problems while using VNNSurv, please feel free to contact <jie_tan@pku.edu.cn>.


