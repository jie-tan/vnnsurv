import os
import numpy as np
import pandas as pd

info = pd.read_excel(io='./TCGA-DLBC/application928/bloodbld2019003535-suppl2.xlsx',sheet_name=0)
info = info[info['Included in statistical analysis']=='Yes'][['Gene','Feature']]
gene_cnv = info[info['Feature']!='Mutation']
gene_mut = info[info['Feature']=='Mutation'][['Gene']]

## 修改后需要重新跑结果, features和risks保存为xx2
# gene_mut['Gene'] = gene_mut['Gene'].str.replace('_noncan', '')
gene_mut['Gene'] = gene_mut['Gene'].str.replace('_265', '')
# gene_mut['Gene'] = gene_mut['Gene'].str.replace('_646', '')

clinical = pd.read_csv('./TCGA-DLBC/DLBC_clinical.csv')
clinical = clinical.iloc[:48,:]
id_clinical = clinical['submitter_id'].values

# 处理生存时间
status = clinical['vital_status'].values
ostime = np.zeros(len(clinical))
for i in range(len(ostime)):
    if status[i]=='Alive':
        ostime[i] = -clinical['days_to_last_follow_up'].values[i]
    else:
        ostime[i] = clinical['days_to_death'].values[i]
# np.save('./TCGA/DLBC/ostime.npy', ostime)
status2 = np.ones(48)
status2[status=='Alive'] = 0
# np.save('./TCGA/DLBC/status.npy', status2)
status = status2.copy()

# 获得三个特征
age = clinical['age_at_index'].values
age = (age>=60).astype(int)
treatment = clinical['treatments_pharmaceutical_treatment_or_therapy'].values
treatment[treatment=='not reported'] = 'no'
m = {'yes': 1, 'no': 0}
treatment = np.array(list(map(m.get, treatment)))
denovo = np.ones(len(clinical))   # 应该对应recurrence，但没有信息，统一标记为新发
Feature_sup = np.transpose(np.vstack((age,treatment,denovo)))

snp = pd.read_csv('./TCGA-DLBC/DLBC_SNP.csv')
Feature_mut = []
for i in range(len(id_clinical)):
    idd = id_clinical[i]
    snp_id = snp[snp['Tumor_Sample_Barcode'].str.contains(idd)][['Hugo_Symbol']]
    snp_id.rename(columns={'Hugo_Symbol': 'Gene'},inplace=True)
    snp_id.drop_duplicates(inplace=True)
    snp_id['Value'] = 1
    temp = pd.merge(gene_mut,snp_id,how='left',on='Gene')
    feature_mut = temp['Value'].values
    feature_mut = np.nan_to_num(feature_mut)  # nan to 0
    Feature_mut.append(feature_mut)
Feature_mut = np.array(Feature_mut)
sum(Feature_mut[:,80-12])

info_cnv = pd.read_csv("./TCGA-DLBC/cnv/gdc_sample_sheet.2023-04-19.tsv",delimiter='\t')
cnv_path = "./TCGA-DLBC/cnv/gdc_download_20230419_021518.667875/"
Feature_cnv = []
for i in range(len(id_clinical)):
    idd = id_clinical[i]
    file = info_cnv[info_cnv['Case ID'].str.contains(idd)]['File ID'].values[0]
    dir = [f for f in os.listdir(cnv_path+file) if '.tsv' in f][0]
    cnv_id = pd.read_csv(cnv_path+file+'/'+dir, delimiter='\t')
    cnv_id.dropna(inplace=True)
    cnv_id.rename(columns={'gene_name': 'Gene'}, inplace=True)
    cnv_id['Feature'] = 'NO'
    cnv_id.loc[cnv_id['copy_number'] > 3,'Feature'] = 'Amplification'
    cnv_id.loc[cnv_id['copy_number'] < 1, 'Feature'] = 'Homozygous deletion'
    cnv_id['Value'] = 1
    temp = pd.merge(gene_cnv, cnv_id, how='left', on=['Gene','Feature'])
    feature_cnv = temp['Value'].values
    feature_cnv = np.nan_to_num(feature_cnv)  # nan to 0
    Feature_cnv.append(feature_cnv)
Feature_cnv = np.array(Feature_cnv)

Features = np.hstack((Feature_cnv,Feature_mut,Feature_sup))
