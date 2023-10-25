import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.optimizer import Optimizer
from torch_geometric.nn import GATConv, max_pool
import logging
from scipy.optimize import fsolve

from time import time
import multiprocessing
cores = multiprocessing.cpu_count() // 2

# VNN_Cell
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class Ranger(Optimizer):
    """Ranger deep learning optimizer - RAdam + Lookahead combined.
    https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

    Ranger has now been used to capture 12 records on the FastAI leaderboard.

    This version = 9.3.19

    Credits:
    RAdam -->  https://github.com/LiyuanLucasLiu/RAdam
    Lookahead --> rewritten by lessw2020, but big thanks to Github @LonePatient and @RWightman for ideas from their code.
    Lookahead paper --> MZhang,G Hinton  https://arxiv.org/abs/1907.08610

    summary of changes:
    full code integration with all updates at param level instead of group, moves slow weights into state dict (from generic weights),
    supports group learning rates (thanks @SHolderbach), fixes sporadic load from saved model issues.
    changes 8/31/19 - fix references to *self*.N_sma_threshold;
    changed eps to 1e-5 as better default than 1e-8.
    """

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5,
                 weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold,
                        eps=eps, weight_decay=weight_decay)
        super(Ranger, self).__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # now we can get to work...
        # removed as we now use step from RAdam...no need for duplicate step counting
        # for group in self.param_groups:
        #    group["step_counter"] = 0
        # print("group step counter init")

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # self.first_run_check=0

        # lookahead weights
        # 9/2/19 - lookahead param tensors have been moved to state storage.
        # This should resolve issues with load/save where weights were left in GPU memory from first load, slowing down future runs.

        # self.slow_weights = [[p.clone().detach() for p in group['params']]
        #                     for group in self.param_groups]

        # don't use grad for lookahead weights
        # for w in it.chain(*self.slow_weights):
        #    w.requires_grad = False

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        if closure is not None:
            loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    # print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1

                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']  # get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  # (fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  # copy interpolated weights to RAdam param tensor

        return loss


class VNN_cell(nn.Module): # BioVNN
    def __init__(self,
                 omic_dim,
                 input_dim, 
                 output_dim, 
                 biovnn_dict, 
                 run_mode="ref",
                 act_func='Mish',
                 use_sigmoid_output=True, # True
                 dropout_p=0.5, # 0
              #   layer_names=None, # community_hierarchy_dicts_all
                 only_combine_child_gene_group=True, #  True
                 neuron_min=10, 
                 neuron_ratio=0.2,
                 use_classification=True, # True
                 child_map_fully=None,  # None
                #  group_level_dict=None,  # self.community_level_dict
                 use_average_neuron_n=True,
                 for_lr_finder=False) -> object: #  False
        
        super(VNN_cell, self).__init__()  # Inherited from the parent class nn.Module
        self.omic_dim = omic_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_names =biovnn_dict["community_hierarchy_dicts_all"]
        self.group_level_dict = biovnn_dict['community_level_dict']
        self.level_group_dict= biovnn_dict["level_community_dict"]
        
        self.use_sigmoid_output = use_sigmoid_output
        self.dropout_p = dropout_p
        self.only_combine_child_gene_group = only_combine_child_gene_group
        print("only_combine_child_gene_group: ",self.only_combine_child_gene_group)
        self.use_classification = use_classification
        self.child_map_fully = child_map_fully
        self.gene_group_idx = self.layer_names['gene_group_idx'] # {'R-HSA-1059683': 10171, 'R-HSA-109581': 10172,
        self.idx_name = self.layer_names['idx_name']

        self.level_neuron_ct = dict()
        self.com_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        if self.dropout_p > 0:
            self.dropout_layers = nn.ModuleDict()
        self._set_layer_names()
        self.build_order = []
        if run_mode=="ref":
            self.child_map = biovnn_dict["mask"] # mask=child_map
        elif run_mode=="random":
            print("using child map random")
            self.child_map=biovnn_dict["mask_random"]
        self.neuron_min = neuron_min
        self.neuron_ratio = neuron_ratio
        self._set_layers()
        if act_func.lower() == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func.lower() == 'mish':
            self.act_func = Mish()
        elif act_func.lower() == 'swish' or act_func.lower() == 'silu':
            self.act_func = Swish()
        self.sigmoid = nn.Sigmoid()
        self.output = [None] * len(self.build_order)
        if self.only_combine_child_gene_group:
            logging.info("{} gene groups do not combine gene features".format(len(self.only_combine_gene_group_dict)))
        self.for_lr_finder = for_lr_finder
        
        self.norm = nn.LayerNorm(541, eps=1e-6)

    def _set_layers(self):
        neuron_n_dict = self._build_layers()
        if self.child_map_fully is not None:
            logging.info("Non-fully connected:")
        self.report_parameter_n()
        logging.debug(self.build_order)

    def _set_layer_names(self):
        for g in self.gene_group_idx.keys(): 
            self.com_layers[g] = None
            self.bn_layers['bn_{}'.format(g)] = None
            self.output_layers['output_{}'.format(g)] = None
            if self.dropout_p > 0:
                self.dropout_layers['drop_{}'.format(g)] = None

    def _build_layers(self, neuron_n_dict=None):
        
        neuron_to_build = list(range(len(self.child_map)))
        # print(len(neuron_to_build)) # 1865
        self.only_combine_gene_group_dict = {}
        if neuron_n_dict is None:
            neuron_n_dict = dict()
        # print("self.idx_name",self.idx_name)
        while len(neuron_to_build) > 0:
            for i in neuron_to_build:
                j = i + self.input_dim
                children = self.child_map[i] 
                child_feat = [z for z in children if z < self.input_dim] # gene_idx
                child_com = [self.idx_name[z] for z in children if z >= self.input_dim] # pathways_names
                child_none = [self.com_layers[z] for z in child_com if self.com_layers[z] is None] # 从下往上搭建网络
                if len(child_none) > 0:
                    logging.debug("Pass Gene group {} with {} children".format(j, len(children)))
                    continue
                neuron_name = self.idx_name[j] 
                # Only combine child gene groups without combine gene features if there is one child gene group
                if self.only_combine_child_gene_group and len(child_com) > 0:
                    # print("True")
                    children_n = len(child_com) 
                    if i == len(self.child_map) - 1: 
                        # children_n += 512 
                        children_n = self.output_dim
                    child_feat = []
                    self.only_combine_gene_group_dict[neuron_name] = 1
                else:
                    # print("False")
                    children_n = len(children)
                    if i == len(self.child_map) - 1: 
                        children_n = self.output_dim
                    # if children_n == 0:
                    #     print(j)
                    #     children_n = 1
                logging.debug("Building gene group {} with {} children".format(j, len(children)))
                if i not in neuron_n_dict:
                    neuron_n = np.max([self.neuron_min, int(children_n * self.neuron_ratio)])
                    # if i == len(self.child_map) - 1: 
                    #     neuron_n = 512
                    # else:
                    #     neuron_n = 10
                    # neuron_n=1
                    neuron_n_dict[i] = neuron_n
                else:
                    neuron_n = neuron_n_dict[i]
                level = self.group_level_dict[neuron_name]
                if level not in self.level_neuron_ct.keys():
                    self.level_neuron_ct[level] = neuron_n
                else:
                    self.level_neuron_ct[level] += neuron_n
                total_in = int(len(child_feat)*self.omic_dim + np.sum([self.com_layers[z].out_features for z in child_com]))
                self.com_layers[neuron_name] = nn.Linear(total_in, neuron_n) # gene个数+孩子pathway所在神经元的输出维度 ==>  child的个数
                self.bn_layers['bn_{}'.format(neuron_name)] = nn.BatchNorm1d(neuron_n)
                if self.dropout_p > 0:
                    self.dropout_layers['drop_{}'.format(neuron_name)] = nn.Dropout(self.dropout_p)
                # self.output_layers['output_{}'.format(neuron_name)] = nn.Linear(neuron_n, self.output_dim)
                neuron_to_build.remove(i)
                self.build_order.append(i)
        print("Successfully build layers!")
        return neuron_n_dict

    def report_parameter_n(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("Total {} parameters and {} are trainable".format(total_params, trainable_params))
        return trainable_params

    def forward(self, features):  # Forward pass: stacking each layer together
        if self.for_lr_finder:
            features = [features[:, i].reshape(features.shape[0], -1) for i in range(features.shape[1])] # 没有shape哦，list[list] 10171(gene)[2000(batch)]
        # print("features_len",len(features)) #10171 # feature gene + delete gene
        # print("features[0]_len",len(features[0])) # 2000(batch)
        # print("output_len",len(self.output))# 1425
        # features： gene数*batch*1 | self.output: pathway（gene group）数
        features = features + self.output  # #features: （gene数+pathway数）*batch*1 | 其中gene数+pathway数=所有神经元

        pred = [None] * len(self.build_order) 
        states = [None] * len(self.build_order) 
        
        for i in self.build_order:
            j = i + self.input_dim
            neuron_name = self.idx_name[j]
            
            com_layer = self.com_layers[neuron_name]
            bn_layer = self.bn_layers['bn_{}'.format(neuron_name)]
            
            children = self.child_map[i]
            if neuron_name in self.only_combine_gene_group_dict:
                children = [z for z in children if z >= self.input_dim]

            input_list = [features[z] for z in children]
            input_mat = torch.cat(input_list, axis=1)
            # print(neuron_name, input_mat.shape)
            features[j] = com_layer(input_mat)
            ## BN after activation
            state = self.act_func(features[j])
            states[i] = state
            features[j] = bn_layer(state)
            # features[j] = state

            if self.dropout_p > 0:
                drop_layer = self.dropout_layers['drop_{}'.format(neuron_name)]
                features[j] = drop_layer(features[j])
            # output_layer = self.output_layers['output_{}'.format(neuron_name)]
            # if self.use_sigmoid_output:
            #     pred[i] = self.sigmoid(output_layer(features[j]))
            # else:
            #     pred[i] = output_layer(features[j])
        # for i in range(10785, len(features)):
        #     print(neuron_name, features[i].shape[1])
        # ret = torch.cat(features[10785:], dim=1)
        
        # print(len(features))
        # out = self.norm(features[-1])
        out = features[-1]
        # out = pred[-1]
        return out


class VNN_cell2(nn.Module):  # BioVNN
    def __init__(self,
                 omic_dim,
                 input_dim,
                 output_dim,
                 biovnn_dict,
                 run_mode="ref",
                 act_func='Mish',
                 use_sigmoid_output=True,  # True
                 dropout_p=0.5,  # 0
                 #   layer_names=None, # community_hierarchy_dicts_all
                 only_combine_child_gene_group=True,  # True
                 neuron_min=10,
                 neuron_ratio=0.2,
                 use_classification=True,  # True
                 child_map_fully=None,  # None
                 #  group_level_dict=None,  # self.community_level_dict
                 use_average_neuron_n=True,
                 for_lr_finder=False) -> object:  # False

        super(VNN_cell2, self).__init__()  # Inherited from the parent class nn.Module
        self.omic_dim = omic_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer_names = biovnn_dict["community_hierarchy_dicts_all"]
        self.group_level_dict = biovnn_dict['community_level_dict']
        self.level_group_dict = biovnn_dict["level_community_dict"]

        self.use_sigmoid_output = use_sigmoid_output
        self.dropout_p = dropout_p
        self.only_combine_child_gene_group = only_combine_child_gene_group
        print("only_combine_child_gene_group: ", self.only_combine_child_gene_group)
        self.use_classification = use_classification
        self.child_map_fully = child_map_fully
        self.gene_group_idx = self.layer_names['gene_group_idx']  # {'R-HSA-1059683': 10171, 'R-HSA-109581': 10172,
        self.idx_name = self.layer_names['idx_name']

        self.level_neuron_ct = dict()
        self.com_layers = nn.ModuleDict()
        # self.bn_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        if self.dropout_p > 0:
            self.dropout_layers = nn.ModuleDict()
        self._set_layer_names()
        self.build_order = []
        if run_mode == "ref":
            self.child_map = biovnn_dict["mask"]  # mask=child_map
        elif run_mode == "random":
            print("using child map random")
            self.child_map = biovnn_dict["mask_random"]
        self.neuron_min = neuron_min
        self.neuron_ratio = neuron_ratio
        self._set_layers()
        if act_func.lower() == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func.lower() == 'mish':
            self.act_func = Mish()
        elif act_func.lower() == 'swish' or act_func.lower() == 'silu':
            self.act_func = Swish()
        self.sigmoid = nn.Sigmoid()
        self.output = [None] * len(self.build_order)
        if self.only_combine_child_gene_group:
            logging.info("{} gene groups do not combine gene features".format(len(self.only_combine_gene_group_dict)))
        self.for_lr_finder = for_lr_finder

        self.norm = nn.LayerNorm(541, eps=1e-6)

    def _set_layers(self):
        neuron_n_dict = self._build_layers()
        if self.child_map_fully is not None:
            logging.info("Non-fully connected:")
        self.report_parameter_n()
        logging.debug(self.build_order)

    def _set_layer_names(self):
        for g in self.gene_group_idx.keys():
            self.com_layers[g] = None
            # self.bn_layers['bn_{}'.format(g)] = None
            self.output_layers['output_{}'.format(g)] = None
            if self.dropout_p > 0:
                self.dropout_layers['drop_{}'.format(g)] = None

    def _build_layers(self, neuron_n_dict=None):

        neuron_to_build = list(range(len(self.child_map)))
        # print(len(neuron_to_build)) # 1865
        self.only_combine_gene_group_dict = {}
        if neuron_n_dict is None:
            neuron_n_dict = dict()
        # print("self.idx_name",self.idx_name)
        while len(neuron_to_build) > 0:
            for i in neuron_to_build:
                j = i + self.input_dim
                children = self.child_map[i]
                child_feat = [z for z in children if z < self.input_dim]  # gene_idx
                child_com = [self.idx_name[z] for z in children if z >= self.input_dim]  # pathways_names
                child_none = [self.com_layers[z] for z in child_com if self.com_layers[z] is None]  # 从下往上搭建网络
                if len(child_none) > 0:
                    logging.debug("Pass Gene group {} with {} children".format(j, len(children)))
                    continue
                neuron_name = self.idx_name[j]
                # Only combine child gene groups without combine gene features if there is one child gene group
                if self.only_combine_child_gene_group and len(child_com) > 0:
                    # print("True")
                    children_n = len(child_com)
                    if i == len(self.child_map) - 1:
                        # children_n += 512
                        children_n = self.output_dim
                    child_feat = []
                    self.only_combine_gene_group_dict[neuron_name] = 1
                else:
                    # print("False")
                    children_n = len(children)
                    if i == len(self.child_map) - 1:
                        children_n = self.output_dim
                    # if children_n == 0:
                    #     print(j)
                    #     children_n = 1
                logging.debug("Building gene group {} with {} children".format(j, len(children)))
                if i not in neuron_n_dict:
                    neuron_n = np.max([self.neuron_min, int(children_n * self.neuron_ratio)])
                    # if i == len(self.child_map) - 1:
                    #     neuron_n = 512
                    # else:
                    #     neuron_n = 10
                    # neuron_n=1
                    neuron_n_dict[i] = neuron_n
                else:
                    neuron_n = neuron_n_dict[i]
                level = self.group_level_dict[neuron_name]
                if level not in self.level_neuron_ct.keys():
                    self.level_neuron_ct[level] = neuron_n
                else:
                    self.level_neuron_ct[level] += neuron_n
                total_in = int(
                    len(child_feat) * self.omic_dim + np.sum([self.com_layers[z].out_features for z in child_com]))
                self.com_layers[neuron_name] = nn.Linear(total_in, neuron_n)  # gene个数+孩子pathway所在神经元的输出维度 ==>  child的个数
                # self.bn_layers['bn_{}'.format(neuron_name)] = nn.BatchNorm1d(neuron_n)
                if self.dropout_p > 0:
                    self.dropout_layers['drop_{}'.format(neuron_name)] = nn.Dropout(self.dropout_p)
                # self.output_layers['output_{}'.format(neuron_name)] = nn.Linear(neuron_n, self.output_dim)
                neuron_to_build.remove(i)
                self.build_order.append(i)
        print("Successfully build layers!")
        return neuron_n_dict

    def report_parameter_n(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("Total {} parameters and {} are trainable".format(total_params, trainable_params))
        return trainable_params

    def forward(self, features):  # Forward pass: stacking each layer together
        if self.for_lr_finder:
            features = [features[:, i].reshape(features.shape[0], -1) for i in
                        range(features.shape[1])]  # 没有shape哦，list[list] 10171(gene)[2000(batch)]
        # print("features_len",len(features)) #10171 # feature gene + delete gene
        # print("features[0]_len",len(features[0])) # 2000(batch)
        # print("output_len",len(self.output))# 1425
        # features： gene数*batch*1 | self.output: pathway（gene group）数
        features = features + self.output  # #features: （gene数+pathway数）*batch*1 | 其中gene数+pathway数=所有神经元

        pred = [None] * len(self.build_order)
        states = [None] * len(self.build_order)

        for i in self.build_order:
            j = i + self.input_dim
            neuron_name = self.idx_name[j]

            com_layer = self.com_layers[neuron_name]
            # bn_layer = self.bn_layers['bn_{}'.format(neuron_name)]

            children = self.child_map[i]
            if neuron_name in self.only_combine_gene_group_dict:
                children = [z for z in children if z >= self.input_dim]

            input_list = [features[z] for z in children]
            input_mat = torch.cat(input_list, axis=1)
            # print(neuron_name, input_mat.shape)
            features[j] = com_layer(input_mat)
            ## BN after activation
            state = self.act_func(features[j])
            states[i] = state
            # features[j] = bn_layer(state)
            features[j] = state

            if self.dropout_p > 0:
                drop_layer = self.dropout_layers['drop_{}'.format(neuron_name)]
                features[j] = drop_layer(features[j])
            # output_layer = self.output_layers['output_{}'.format(neuron_name)]
            # if self.use_sigmoid_output:
            #     pred[i] = self.sigmoid(output_layer(features[j]))
            # else:
            #     pred[i] = output_layer(features[j])
        # for i in range(10785, len(features)):
        #     print(neuron_name, features[i].shape[1])
        # ret = torch.cat(features[10785:], dim=1)

        # print(len(features))
        # out = self.norm(features[-1])
        out = features[-1]
        # out = pred[-1]
        return out

class FullyNet(VNN_cell):
    def __init__(self,
                 input_dim,
                 output_dim,
                 drug_dim,
                 biovnn_dict,
                 run_mode="full",
                 
                 act_func='Mish',
                 use_sigmoid_output=True, # True
                 dropout_p=0.1,# or 0 
                 gene_feat=3,
              #   layer_names=None, # community_hierarchy_dicts_all
                 only_combine_child_gene_group=True, #  True
                 neuron_min=10,
                 neuron_ratio=0.2,
                 use_classification=True, # True
                 child_map_fully=None,
                #  group_level_dict=None,  # self.community_level_dict
                 use_average_neuron_n=True, # False为保留BioVNN的结构
                 for_lr_finder=False): #  False

        super(FullyNet, self).__init__(input_dim,output_dim,drug_dim, biovnn_dict,run_mode, act_func,use_sigmoid_output,
                                       dropout_p,gene_feat,only_combine_child_gene_group,neuron_min,neuron_ratio,
                                       use_classification, child_map_fully, use_average_neuron_n,for_lr_finder)  # Inherited from the parent class nn.Module

        self.use_average_neuron_n = use_average_neuron_n
        #self.child_map_fully=biovnn_dict['mask_ones']

        parameter_n = self.report_parameter_n()
        self._build_layers_fully(parameter_n)
        logging.info("Fully connected network:")
        self.report_parameter_n()

    def solve_neuron_n(self):
        def func(i):
            x = i[0]
            # return [input_dim * x + (layer_n-1) * (x ** 2 + x) - parameter_n]
            return [10171 * x + 12 * (x ** 2 + x) - 714077]

        r = fsolve(func, [0])
        return int(r[0])

    def _build_layers_fully(self, parameter_n=39974, layer_n=13):
        # Reset layers
        self.com_layers = None
        self.fully_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        if self.dropout_p > 0:
            self.dropout_layers = nn.ModuleDict()
        self.build_order = []
        # Total available neuron number
        total_n = parameter_n // self.input_dim
        if total_n / float(layer_n) < 1:  # only need one hidden layer
            self.build_order.append(0)
            self.fully_layers['fully_0'] = nn.Linear(self.input_dim, total_n)
            self.bn_layers['bn_0'] = nn.BatchNorm1d(total_n)
            if self.dropout_p > 0:
                self.dropout_layers['drop_0'] = nn.Dropout(self.dropout_p)
            self.output_layers['output_0'] = nn.Linear(total_n, self.output_dim)
            print("The fully connected network has 1 layers")
        else:
            neuron_per_layer = self.solve_neuron_n()
            if self.use_average_neuron_n:
                logging.info(
                    "The fully connected network has {} neurons per layer for total {} layers".format(neuron_per_layer,
                                                                                             layer_n))
            print("The fully connected network has {} layers".format(layer_n))
            # neuron_per_layer = total_n // layer_n
            for i in range(layer_n):
                self.build_order.append(i)

                if self.use_average_neuron_n==True:
                    # print("Use average_neuron_n!")
                    if i == 0:
                        # self.fully_layers['fully_' + str(i)] = nn.Linear(self.input_dim*self.gene_feat, neuron_per_layer)
                        self.fully_layers['fully_' + str(i)] = nn.Linear(self.input_dim, neuron_per_layer)
                        self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(neuron_per_layer)
                    elif i ==layer_n-1:
                        self.fully_layers['fully_' + str(i)] = nn.Linear(neuron_per_layer, neuron_per_layer)
                        self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(neuron_per_layer)
                    else:
                        self.fully_layers['fully_' + str(i)] = nn.Linear(neuron_per_layer, neuron_per_layer)
                        self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(neuron_per_layer)


                    self.output_layers['output_' + str(i)] = nn.Linear(neuron_per_layer, self.output_dim)
                else:
                    print("Don't use average_neuron_n!")
                    if i == 0:
                        in_n = self.input_dim#*self.gene_feat
                        out_n = self.level_neuron_ct[i + 1]
                    else:
                        in_n = self.level_neuron_ct[i]
                        out_n = self.level_neuron_ct[i + 1]
                    self.fully_layers['fully_' + str(i)] = nn.Linear(in_n, out_n)
                    logging.info("The fully connected network layer {} has {} neurons".format(i, out_n))
                    self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(out_n)
                    self.output_layers['output_' + str(i)] = nn.Linear(out_n, self.output_dim)

                if self.dropout_p > 0:
                    self.dropout_layers['drop_' + str(i)] = nn.Dropout(self.dropout_p)

        self.output = [None] * len(self.build_order)
    def drug_gene_att(self, features, drug):
        gene_feat = torch.stack(features, axis=0)
        drug_feat = drug

        Q = self.dropout_d(self.drug_query(drug_feat))
        K = self.gene_key(gene_feat)
        V = self.gene_value(gene_feat)
        # V = gene_feat

        attention_scores = torch.sum(Q*K, dim=-1).unsqueeze(-1) # [L, B, 1]
        attention_scores = attention_scores.sigmoid()

        gene_feat = V*attention_scores

        return gene_feat, attention_scores
    
    def forward(self, features):  # Forward pass: stacking each layer together
        # gene_feat,att=self.drug_gene_att(features,drug)
        gene_feat=torch.stack(features,axis=0)
        #print(len(features))
#        print(len(features[0]))
 #       print(len(features[0][0]))
        features = features + self.output
        
        pred = [None] * len(self.build_order)
        states = [None] * len(self.build_order)
        
        for i in self.build_order:
 #           print(i)
            # j = i + self.input_dim
            neuron_name = i
            fully_layer = self.fully_layers['fully_{}'.format(neuron_name)]
            bn_layer = self.bn_layers['bn_{}'.format(neuron_name)]
            if i == 0:
               input_mat = gene_feat
               input_mat = torch.flatten(input_mat.transpose(1,0), start_dim=1)

            else:
                input_mat = states[i-1]
            state= fully_layer(input_mat)
            ## BN after activation
            state = self.act_func(state)
            state= bn_layer(state)
            
            if self.dropout_p > 0:
                drop_layer = self.dropout_layers['drop_{}'.format(neuron_name)]
                states[i] = drop_layer(state)
        # print(features[-1].shape)
        #     output_layer = self.output_layers['output_{}'.format(neuron_name)]
        #     if self.use_sigmoid_output:
        #         pred[i] = self.sigmoid(output_layer(features[j]))
        #     else:
        #         pred[i] = output_layer(features[j])

        # if self.for_lr_finder:
        #     return pred[-1][:, 1]
        # return pred, states

        return states[-1]
        # print(features[-1].shape)
        #     output_layer = self.output_layers['output_{}'.format(neuron_name)]
        #     if self.use_sigmoid_output:
        #         pred[i] = self.sigmoid(output_layer(features[j]))
        #     else:
        #         pred[i] = output_layer(features[j])

        # if self.for_lr_finder:
        #     return pred[-1][:, 1]
        # return pred, states

        
