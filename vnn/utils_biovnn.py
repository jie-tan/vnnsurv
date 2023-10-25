import os
import re
import pandas as pd
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict as ddict
import matplotlib.pyplot as plt

import sys
sys.path.append("../biovnn")


path = "./"
    
class BioVNN_pre():
    def __init__(self, gene_name_list,result_dir):
            
        self.community_hierarchy_path=os.path.join("./vnn/ReactomePathwaysRelation.txt")
        self.community_path = os.path.join("./vnn/ReactomePathways.gmt")
        self.result_dir = result_dir
        # with open("./biovnn/params/biovnn.yml", 'r') as stream:
        #     self.params = yaml.safe_load(stream)
        self.seed = 0
        self.run_mode=['ref'] # or "full"
        self.use_all_feature_for_fully_net = False #
        self.use_all_feature_for_random_group=False
        self.gene_name_list=gene_name_list # gene_name_list

        
    def prepare_data(self):
        # 1. 获取
        #   self.community_genes = set()
        #   self.community_dict = {}   Pathways_name: pathway
        #   self.gene_community_dict = ddict(list)   gene :Pathways_name
        #   self.community_size_dict = {}
        self.load_communities() 

        # 2. 获取     
        #   self.cell_line_id_mapping  ModelID:COSMICID   
        # self.load_known_genes()  
        
        self.align_data()

    def perform(self):
        # mask=child_map_all
        self.prepare_data()
        self.save_data()
        # self.run_exp()

    def load_communities(self):
        lines = open('{}'.format(self.community_path)).readlines()
        if 'pathway' in self.community_path.lower(): ###
            ind_key = 1
            ind_gene = 3
        elif self.community_path.lower().endswith('.gmt'):
            ind_key = 1
            ind_gene = 3
        else:
            ind_key = 0
            ind_gene = 1

        self.community_genes = set()
        self.community_dict = {}
        self.gene_community_dict = ddict(list)
        self.community_size_dict = {}

        # community=pathways
        for line in lines:
            line = line.strip().split('\t')
            self.community_dict[line[ind_key]] = line[ind_gene:]
            self.community_size_dict[line[ind_key]] = len(line[ind_gene:])
            self.community_genes |= set(line[ind_gene:])
            for g in line[ind_gene:]:
                self.gene_community_dict[g].append(line[ind_key])

    def load_known_genes(self):
            dir=path+"rawdata/ccle/Model.csv"
            
            self.cell_line_metadata = pd.read_csv(dir)
            self.cell_line_metadata = self.cell_line_metadata.set_index('ModelID')
            self.cell_line_id_mapping = self.cell_line_metadata['COSMICID'].to_dict()
            self.cell_line_id_mapping = ddict(lambda: None, self.cell_line_id_mapping)
            # cell_line_cosmic_mapping= cell_line_metadata['COSMICID'].to_dict()
            # cell_line_cosmic_mapping= ddict(lambda: None, cell_line_cosmic_mapping)
            # self.cell_line_id_pri_dis = ddict(lambda: None, self.cell_line_id_pri_dis)
            # self.cell_line_id_sub_dis = ddict(lambda: None, self.cell_line_id_sub_dis)
            
            # print(self.cell_line_id_mapping) # ModelID:CCLEName
            # print(self.cell_line_id_pri_dis) # CCLEName:(none:lineage)
            # print(self.cell_line_id_sub_dis) # lineage_subtype

    def align_data(self): 
        # 这里有一步select_feature_genes,需要在预处理部分进行
        # self.select_feature_genes()
        self.filter_community() # 可选
        # self.run_create_filter() # self.community_filter好像没用到
        self.build_hierarchy()

    def filter_community(self,
                        community_affected_size_min=5,community_affected_size_max=999999,
                        require_label_gene_in_gene_group=True):
        com_to_drop = []
        modeled_com_genes = set()
        modeled_genes = set()

        modeled_genes |= set(self.gene_name_list)
        for com, members in self.community_dict.items(): # com:pathname,member:pathway(from gene)
            self.community_dict[com] = sorted(list(set(modeled_genes).intersection(members)))
            if len(self.community_dict[com]) < community_affected_size_min:
                com_to_drop.append(com)
            elif len(self.community_dict[com]) > community_affected_size_max:
                com_to_drop.append(com)
            elif len(set(members) & set(self.gene_name_list)) < 1:
                com_to_drop.append(com)
            else:
                modeled_com_genes |= set(self.community_dict[com])
        for com in com_to_drop:
            self.community_dict.pop(com, None)

    def run_create_filter(self):
        self.feature_genes = set()
        self.genes_in_label_idx = {}
        self.idx_genes_in_label = {}
        # 感觉不需要
        # self.community_filter = self.create_filter(self.gene_community_dict, self.community_dict,
        #                                              self.community_size_dict, random=False)
   
    def create_filter(self, gene_community_dict, community_dict, community_size_dict, random=False):
        community_filter = ddict(set)
        if not random:
            self.genes_in_label_idx = {}
            self.idx_genes_in_label = {}
        i = 0
        for g in self.gene_name_list:
            coms = gene_community_dict[g]  # 获取包含基因g的pathways
            coms = list(set(coms) & (community_dict.keys()))
            com_size = [community_size_dict[x] for x in coms]
            community_filter[g] |= set([g])

            for s, com in sorted(zip(com_size, coms)): 
                genes = set(community_dict[com]) # 获取pathway com 所包含的gene
                # Choose top n genes so that not too many features were used per gene group
                if 0:
                    1
                else:
                    self.genes_in_label_idx[g] = i # {gene:idx}
                    self.idx_genes_in_label[i] = g # {idx:gene}
                    i += 1

        if not random:
            print(
                "The dependency of total {} genes will be predicted".format(len(self.genes_in_label_idx.keys())))
        return community_filter

    def build_hierarchy(self):
        leaf_communities, df = self.load_leaf_communities() # 叶子的pathway_name 和 全部relations
        print("shape:",df.shape)
        # df
        # 12369  R-HSA-991365   R-HSA-997272
        #  0     root    R-HSA-74160
        child = leaf_communities
        level = 1        # The layer having only gene children
        self.community_level_dict = dict()
        self.level_community_dict = dict()
        count_dict = ddict(int) # 该pathway形成的神经元个数（？
        for x in child:
            self.community_level_dict[x] = level
            count_dict[x] += 1
        self.level_community_dict[level] = child
        # logging.info("Layer {} has {} gene groups".format(level, len(child)))
        # while level==1:
        while 1:
            df_level = df.loc[df[1].isin(child)] # 找到含有child的relations
            if df_level.shape[0] == 0:
                break
            level += 1
            parent = sorted(list(set(df_level[0])))
            
            for parent_group in parent:
                # 这个在不断更新，会保留更高的level,全部和只算两层的结果会不一样
                self.community_level_dict[parent_group] = level  
                count_dict[parent_group] += 1
            self.level_community_dict[level] = parent
            child = parent
        # print(self.community_level_dict)
        # Make the layer number of each community unique
        self.level_community_dict = ddict(list) #
        for g, level in  self.community_level_dict.items():
            self.level_community_dict[level].append(g)
        for level, groups in sorted(self.level_community_dict.items()):
            print("Layer {} has {} gene groups".format(level, len(groups)))

        gene_groups_all = sorted(list(self.community_dict.keys())) +['root']
        # gene_groups_all = sorted(list(self.community_level_dict.items()), key=lambda x:x[1])
        # gene_groups_all = [i[0] for i in gene_groups_all]
        # gene_groups_all = sorted(list(self.community_level_dict.keys()))
        # from collections import Counter
        # dict1=Counter(gene_groups_all)
        # print(gene_groups_all)
        # print(dict1)
        if "root" in self.community_level_dict.keys():
            print(
            "Total {} layers of {} gene groups in the hierarchy including the root".format(level, len(gene_groups_all)))
        else:
            print(
                "Total {} layers of {} gene groups in the hierarchy not including the root".format(level, len(gene_groups_all)))

        # feature_genes_all = sorted(list(self.gene_name_list))
        feature_genes_all = []
        # feature_n = []
        # np.random.RandomState(self.params['seeds'][0])
        np.random.RandomState(self.seed)
        # for data_type in self.data_types:
        #     feat_n = len(self.__dict__[data_type].columns)
        #     self.feature_n.append(feat_n)
        #     # Randomly reselect features for each feature matrix
            # if 'full' in self.run_mode and self.use_all_feature_for_fully_net:
            #     # feat_pool = sorted(list(self.__dict__[data_type + '_all'].columns))
            #     # feature_genes_all += feat_pool
            #     # cell_idx = self.__dict__[data_type].index
            #     # self.__dict__[data_type] = self.__dict__[data_type + '_all'].loc[cell_idx, feat_pool]
            #     # logging.info(
            #     #     "Use all {} genes from {} as features to form fully connected networks".format(feat_n, data_type))
            # elif 'ref' not in self.run_mode and self.use_all_feature_for_random_group:
            #     feat_pool = list(self.__dict__[data_type + '_all'].columns)
            #     # Require gene labels in the features
            #     pre_select = set(feat_pool) & set(self.genes_in_label)
            #     feat_pool = sorted(list(set(feat_pool) - set(self.genes_in_label)))
            #     random_feat = sorted(list(np.random.choice(feat_pool, feat_n - len(pre_select), replace=False)))
            #     feature_genes_all += random_feat + list(pre_select)
            #     feature_genes_all = sorted(feature_genes_all)
            #     cell_idx = self.__dict__[data_type].index
            #     self.__dict__[data_type] = self.__dict__[data_type + '_all'].loc[cell_idx, random_feat]
            #     logging.info(
            #         "Randomly select {} genes including {} gene of prediction from {} as features to form random gene groups".format(
            #             feat_n, len(self.genes_in_label), data_type))
            # else:
            #     feature_genes_all += sorted(list(self.__dict__[data_type].columns))
        
    
        feature_genes_all+=self.gene_name_list # 看一下师兄的
        
        # del_genes_all = sorted(list(genes_in_label_idx))
        # feature_n.append(len(del_genes_all))
        # genes_in_label = del_genes_all
        # save_label_genes(genes_in_label)
        # self.y = self.dependency[self.genes_in_label]
        # self.y_binary = ((self.y >= 0.5) + 0).astype(int)
        # The order of indexed genes and gen groups:
        entity_all = feature_genes_all + gene_groups_all
        self.idx_name = {i: k for i, k in enumerate(entity_all)}
        name_idx = ddict(list)
        for k, v in self.idx_name.items():
            name_idx[v].append(k)
        # if len(self.data_types) > 1:
        #     self.mut_genes_idx = {}
        #     self.rna_genes_idx = {}
        #     for k, v in name_idx.items():
        #         for idx in v:
        #             if idx < self.feature_n[0]:
        #                 self.mut_genes_idx[k] = idx
        #             elif self.feature_n[0] <= idx < self.feature_n[0] + self.feature_n[1]:
        #                 self.rna_genes_idx[k] = idx
        self.feature_genes_idx = {x: min(name_idx[x]) for x in feature_genes_all}
        self.gene_group_idx = {x: name_idx[x][0] for x in gene_groups_all}
        self.community_hierarchy_dicts_all = { 'idx_name':self.idx_name,
                                            'feature_genes_idx': self.feature_genes_idx,
                                            'gene_group_idx': self.gene_group_idx}
        
        self.child_map_all = [] # ref
        self.child_map_all_random = []
        self.child_map_all_ones = [] # full

        gene_pool = sorted(list(set(feature_genes_all) ))
        self.community_filter_random = ddict(list)
        
        self.community_dict_random = {}
        random_hierarchy = pd.DataFrame()

        self.gene_community_dict_random = ddict(list)
        self.community_size_dict_random = {}
        
        # prng = np.random.RandomState(self.params['seeds'][0])
        prng = np.random.RandomState(self.seed)
        print("Building gene group hierarchy")

        # if run_mode 不是random
        partially_shuffled_membership = None
        partially_shuffled_relation = None
        idx_gene_group = None
        idx_gene_pool = None
        '''
        draw MultiDigraph
        '''
        # G = nx.MultiDiGraph()
        # plt.figure(figsize=(400, 400))
        plt.figure(3,figsize=(48,36))
        G = nx.Graph()
        pos={}
        edges=[]
        idxs=set()
        cross_level_num={}
        
        min_group_idx = min(self.gene_group_idx.values())
        print("len",len(self.gene_group_idx))
        for group, idx in sorted(self.gene_group_idx.items()): # # 保存pathway的idx {'R-HSA-1059683': 10171, 'R-HSA-109581': 10172,
            if group in self.community_dict:
                genes = self.community_dict[group]
                gene_idx = self.genes_to_feat_del_idx(genes)
                if partially_shuffled_membership is not None:
                    genes_random_idx = partially_shuffled_membership[idx - min_group_idx].nonzero()[0]
                    genes_random = sorted([idx_gene_pool[x] for x in genes_random_idx])
                else:
                    genes_random = sorted(list(prng.choice(gene_pool, len(genes), replace=False)))
                self.community_dict_random[group] = genes_random # random模式,随机在基因池里choice一条pathway
                for g in genes_random:
                    self.gene_community_dict_random[g].append(group)
                self.community_size_dict_random[group] = len(genes_random)

                feat_genes = set(genes_random) & set(self.feature_genes_idx.keys())
                # if len(data_types) > 1:
                #     feat_gene_idx = []
                #     for g in feat_genes:
                #         if g in self.mut_genes_idx:
                #             feat_gene_idx.append(self.mut_genes_idx[g])
                #         if g in self.rna_genes_idx:
                #             feat_gene_idx.append(self.rna_genes_idx[g])
                # else:
                feat_gene_idx = [self.feature_genes_idx[x] for x in feat_genes]
                gene_idx_random = feat_gene_idx
            else:
                gene_idx = []
                gene_idx_random = []

            child = sorted(df.loc[df[0] == group, 1].tolist())
            p_level=self.community_level_dict[group]
            # label=1
            # for c in child:
            #     c_level=self.community_level_dict[c]
            #     if c_level==p_level:
            #         print("The pathway {} has the same level {} with child {}!".format(group,c_level,c))
            #         label=0
            # if label==1:
            #     print("No child in the same level {} with parent {}".format(p_level,group))
            
            
            self.re2=0
            if self.re2:
                # 如果只算两层relations
                child&=self.community_level_dict.keys()
            else:
                pass
            
            child_idx = sorted([self.gene_group_idx[x] for x in child if x in self.gene_group_idx])
            idx_level=self.community_level_dict[group]
            for c in child:
                if c in self.gene_group_idx:
                    c_level=self.community_level_dict[c]
                    cross_level=abs(c_level-idx_level)
                    if cross_level not in cross_level_num:
                        cross_level_num[cross_level]=1
                    else:
                        cross_level_num[cross_level]+=1
                    if cross_level>4:
                        edges.append((self.gene_group_idx[c],idx))
                        idxs.add(idx)
                        idxs.add(self.gene_group_idx[c])
            
            # if len(child_idx)<1:
            #     print(self.community_level_dict[group])
            # neighbor[idx]=child_idx
            # for idxx in neighbor[idx]:
            #     if idxx in neighbor.keys():
            #         if idx in neighbor[idxx]:
            #             edges.extend([(idx,idxx),(idxx,idx)])
                
            self.child_map_all.append(sorted(gene_idx + child_idx))
            if len(self.child_map_all[-1]) == 0:
                print("Gene group {} does not have children".format(group))
            # Build random group hierarchy

            # if partially_shuffled_relation is not None:
            #     child_idx_random = partially_shuffled_relation[idx - min_group_idx, :].nonzero()[0]
            #     child_idx_random = [x + min_group_idx for x in child_idx_random]
            #     child_random = sorted([idx_gene_group[x] for x in child_idx_random])
            # else:
            child_idx_random = []
            child_random = []
            for c in child:
                child_level = self.community_level_dict[c]
                random_child = prng.choice(self.level_community_dict[child_level], 1, replace=False)[0] # 从child c 所在的level里random一个
                child_random.append(random_child)
                random_c_idx = self.gene_group_idx[random_child]
                child_idx_random.append(random_c_idx)

            for rc in sorted(child_random):
                random_hierarchy = pd.concat([random_hierarchy, pd.DataFrame([group, rc]).T], axis=0)
            self.child_map_all_random.append(sorted(gene_idx_random + child_idx_random))
            try:
                assert len(gene_idx) == len(gene_idx_random), "Random gene number does not match"
            except AssertionError:
                pass

            # Children for fully connected neural networks
            if group in leaf_communities:
                gene_idx_ones = list(self.feature_genes_idx.values())
            else:
                gene_idx_ones = []
            parent_level = self.community_level_dict[group]
            child_level = parent_level - 1
            if child_level in self.level_community_dict:
                child_ones = self.level_community_dict[child_level] # child_level 这层的所有神经元
            else:
                child_ones = []
            child_idx_ones = [self.gene_group_idx[x] for x in child_ones if x in self.gene_group_idx]
            self.child_map_all_ones.append(sorted(gene_idx_ones + child_idx_ones))
        
        
        print("edges len:",len(edges))
        G.add_edges_from(edges)
        G.add_nodes_from(list(idxs))
        
        # for group,id in self.gene_group_idx.items():
        #     l=self.community_level_dict[group]
        #     y=l*5
        #     pathways=self.level_community_dict[l]
        #     x=pathways.index(group)
        #     x*=20
        #     pos[id]=(x,y)
        draw={}
        
        for idxx in idxs:
            group=self.idx_name[idxx]
            l=self.community_level_dict[group]
            if l not in draw.keys():
                draw[l]=[idxx]
            else:
                draw[l].append(idxx)
        
        for idxx in idxs:
            group=self.idx_name[idxx]
            l=self.community_level_dict[group]
            x=draw[l].index(idxx)
            x-=len(draw[l])/2
            pos[idxx]=(x*5,l*10)
                
        nx.draw_networkx(G,pos,node_size=300,width=5,with_labels=None, node_color='#2878b5',edge_color='#82B0D2')
        plt.savefig("path2.png")
        
        total=0
        lists=sorted(list(cross_level_num.keys()))
        for i in lists:
            num=cross_level_num[i]
            total+=num
            print("The num of relations crossing {} levels is {}".format(i,num))
        print("Total {} relations".format(total))
        
        self.save_communities(self.community_dict_random)
        # Save random hierarchy as file
        random_hierarchy.to_csv(os.path.join(self.result_dir, 'random_group_hierarchy.tsv'),
                                index=None, sep='\t', header=None)
        # self.community_filter_random = self.create_filter(self.gene_community_dict_random, self.community_dict_random,
        #                                                     self.community_size_dict_random, random=True) # 不误

        # community_filter_map = []
        # community_filter_map_random = []
        # feature_n = len(feature_genes_all)
        # for g in del_genes_all:
        #     feat_genes = set(community_filter[g])
        #     if len([]) > 1:
        #         1
        #     else:
        #         feat_gene_idx = sorted([feature_genes_idx[x] for x in feat_genes if x in self.feature_genes_idx])
        #     feat_genes_array = np.zeros(feature_n)
        #     feat_genes_array[feat_gene_idx] = 1
        #     community_filter_map.append(feat_genes_array)
        #     feat_genes_random = set(community_filter_random[g])
        #     if len([]) > 1:
        #         # feat_genes_random_idx = []
        #         # for g in feat_genes:
        #         #     if g in self.mut_genes_idx:
        #         #         feat_genes_random_idx.append(self.mut_genes_idx[g])
        #         #     if g in self.rna_genes_idx:
        #         #         feat_genes_random_idx.append(self.rna_genes_idx[g])
        #         # feat_genes_random_idx = sorted(feat_genes_random_idx)
        #         1
        #     else:
        #         feat_genes_random_idx = sorted(
        #             [feature_genes_idx[x] for x in feat_genes_random if x in feature_genes_idx])
        #     feat_genes_array = np.zeros(feature_n)
        #     feat_genes_array[feat_genes_random_idx] = 1
        #     community_filter_map_random.append(feat_genes_array)

    def load_leaf_communities(self):
        f = self.community_hierarchy_path
        # The first column 0 is the parent and the second column 1 is the child
        df = pd.read_csv(f, sep='\t', header=None)
        if 'Reactome' in f:
            df = df.loc[df[0].str.contains('HSA')]  # Get human-only pathways
        # Make root as the parent of those gene groups without parents
        df_root = pd.DataFrame(columns=df.columns)
        for x in set(df[0]) - set(df[1]):
            if x in self.community_dict or 'GO:' in x:
                df_root = pd.concat([df_root, pd.DataFrame(['root', x]).T])
        # Remove those relationship of groups not in the analysis
        df = df.loc[df[1].isin(self.community_dict.keys()) & df[0].isin(self.community_dict.keys())]
        df = pd.concat([df, df_root])
        leaf_communities = sorted(list((set(df[1]) - set(df[0])) & set(self.community_dict.keys())))

        return leaf_communities, df

    def genes_to_feat_del_idx(self,genes):
        feat_genes = set(genes) & set(self.feature_genes_idx.keys())
        # if len(data_types) > 1:
        #     feat_gene_idx = []
        #     for g in feat_genes:
        #         if g in mut_genes_idx:
        #             feat_gene_idx.append(mut_genes_idx[g])
        #         if g in rna_genes_idx:
        #             feat_gene_idx.append(rna_genes_idx[g])
        # else:
        feat_gene_idx = [self.feature_genes_idx[x] for x in feat_genes]
        # if self.use_deletion_vector:
        #     del_gene_idx = [self.del_genes_idx[x] for x in del_genes]
        # else:
        # del_gene_idx = []
        gene_idx = feat_gene_idx # del_gene_idx
        return gene_idx

    def save_communities(self,community_dict,d=None):
        """Save community genes to file."""
        if d is None:
            fout = open(os.path.join(self.result_dir, 'community_list.tsv'), 'w')
            d = community_dict
            s = ''
        else:
            fout = open(os.path.join(self.result_dir, 'community_random_list.tsv'), 'w')
            s = '_random'
        for k, v in d.items():
            fout.write('{}\n'.format('\t'.join([k + s] + v)))
        fout.close()

    def load_selected_gene_list(self):
        file_path="../data/DGIdb_genes_w_interactions.txt"
        self.select_genes_in_label = pd.read_csv(file_path, header=None)[0].tolist()

    def save_data(self):
        with open(os.path.join(self.result_dir,"BioVNN_pre.pkl"),"wb") as f:
            d={
                "community_hierarchy_dicts_all":self.community_hierarchy_dicts_all,
                "community_level_dict":self.community_level_dict,
                "level_community_dict":self.level_community_dict,
                "mask":self.child_map_all,
                "mask_ones":self.child_map_all_ones # 用于fully connected
                # "params":self.params
            }
            pickle.dump(d,f)
        print("BioVNN_pre.pkl Successfully save!")
        # with open(os.path.join(self.result_dir,"BioVNN_pre.pkl"),"wb") as f:
        #     dict1={
        #         "community_hierarchy_dicts_all":self.community_hierarchy_dicts_all,
        #         "community_level_dict":self.community_level_dict,
        #         "level_community_dict":self.level_community_dict,
        #         "mask":self.child_map_all,
        #         # "params":self.params
        #     }
        #     pickle.dump(dict1,f)


