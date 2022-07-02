import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx
import bnlearn
import time

survey_data_loc = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/BN_rule/'
survey_data = pd.read_csv(survey_data_loc+'BN_data_start1_adj.csv')

edges = [('farm_size', 'imp_env'),
         ('info_use', 'imp_env'),
         ('info_use', 'max_fam'),
        ('peer_ec','max_fam'),
        ('benefit','SC_Will'),
        ('concern','SC_Will'),
        ('imp_env','SC_Will'),
        ('max_fam','SC_Will'),
        ('lql','SC_Will'),
        ('SC_Contract','SC_Will'),
        ('SC_Rev_mean','SC_Will'),
        ('SC_Rev_range','SC_Will'),
        ('SC_Will','SC_Ratio'),
        ('SC_Contract','SC_Ratio'),
        ('SC_Rev_mean','SC_Ratio'),
        ('SC_Rev_range','SC_Ratio')]
# Make the actual Bayesian DAG
model = bnlearn.make_DAG(edges)
model = bnlearn.parameter_learning.fit(model, survey_data)

bn_df = bnlearn.sampling(model,n=10**6)
temp=time.time()
bn_df['SC_Will'][(bn_df['farm_size']==1)&(bn_df['info_use']==2)&(bn_df['peer_ec']==2)&(bn_df['imp_env']==3)].mean()-1
time.time()-temp

