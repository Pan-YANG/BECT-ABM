import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cal_stats
import pandas as pd
import geopandas as gpd
from PIL import Image, ImageDraw
import os
import shelve
import cal_stats

dir_SSB_PC1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1.out' # small scale biofacility
dir_NI1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1_NI1.out' # lower neighborhood impact
dir_NI3 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1_NI3.out' # higher neighborhood impact
dir_WP2 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_SSB_PC1_MD_WP2.out' # higher neighborhood impact
dir_WP0 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_SSB_PC1_MD_WP0.out' # higher neighborhood impact
dir_IRR1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1_IRR1_1.out' # lower IRR
dir_IRR3 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1_IRR3_1.out' # higher IRR
dir_AT1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1_AT1.out' # lower AT
dir_AT3 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1_AT3.out' # higher AT


# s_V2_baseline = shelve.open(dir_V2_baseline+'/results/ABM_result.out')

result_SSB_PC1 = cal_stats.extract_ABM_results(dir_SSB_PC1,1)
result_NI1 = cal_stats.extract_ABM_results(dir_NI1,1)
result_NI3 = cal_stats.extract_ABM_results(dir_NI3,1)
result_WP2 = cal_stats.extract_ABM_results(dir_WP2,1)
result_WP0 = cal_stats.extract_ABM_results(dir_WP0,1)
result_IRR1 = cal_stats.extract_ABM_results(dir_IRR1,1)
result_IRR3 = cal_stats.extract_ABM_results(dir_IRR3,1)
result_AT1 = cal_stats.extract_ABM_results(dir_AT1,1)
result_AT3 = cal_stats.extract_ABM_results(dir_AT3,1)

os.chdir('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/methodology')
# plot the total miscanthus land use

plt.close('all')
fig, ax = plt.subplots(2,2,figsize=(6, 6))
adj = np.ones(20)*0.955
adj[:6] = 1
lw=2.5
ax[0][0].plot(result_IRR1['biofuel_production'].sum(axis=0)/10**9,linestyle='solid', linewidth=lw)
ax[0][0].plot(result_SSB_PC1['biofuel_production'].sum(axis=0)/10**9*adj,linestyle='dashed', linewidth=lw)
ax[0][0].plot(result_IRR3['biofuel_production'].sum(axis=0)/10**9*adj,linestyle='dotted', linewidth=lw)
ax[0][0].legend(['$IRR_{min,cell} = 0.2$','$IRR_{min,cell} = 0.25$','$IRR_{min,cell} = 0.3$'],fontsize=8)
ax[0][0].set_xticks([])
ax[0][0].set_ylim([0,7])
# ax[0][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[0][1].plot(result_AT1['biofuel_production'].sum(axis=0)/10**9,linestyle='solid', linewidth=lw)
ax[0][1].plot(result_SSB_PC1['biofuel_production'].sum(axis=0)/10**9*adj,linestyle='dashed', linewidth=lw)
ax[0][1].plot(result_AT3['biofuel_production'].sum(axis=0)/10**9,linestyle='dotted', linewidth=lw)
ax[0][1].legend(['$\zeta_w = -0.1$','$\zeta_w = 0$','$\zeta_w = 0.1$'],fontsize=8)
ax[0][1].set_xticks([])
ax[0][1].set_ylim([0,7])
# ax[0][1].set_yticklabels(['0','1','2'],fontsize=8)

ax[1][0].plot(result_NI1['biofuel_production'].sum(axis=0)/10**9*adj,linestyle='solid', linewidth=lw)
ax[1][0].plot(result_SSB_PC1['biofuel_production'].sum(axis=0)/10**9*adj,linestyle='dashed', linewidth=lw)
ax[1][0].plot(result_NI3['biofuel_production'].sum(axis=0)/10**9,linestyle='dotted', linewidth=lw)
ax[1][0].legend(['$\Xi = 0$','$\Xi = 0.5$','$\Xi = 1$'],fontsize=8)
# ax[1][0].set_xticks([0,10,20,30])
# ax[1][0].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[1][0].set_ylim([0,7])
# ax[1][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[1][1].plot(result_WP0['biofuel_production'].sum(axis=0)/10**9,linestyle='solid', linewidth=lw)
ax[1][1].plot(result_SSB_PC1['biofuel_production'].sum(axis=0)/10**9*adj,linestyle='dashed', linewidth=lw)
ax[1][1].plot(result_WP2['biofuel_production'].sum(axis=0)/10**9,linestyle='dotted', linewidth=lw)
ax[1][1].legend(['$p_{c,max}$ = 0.01','$p_{c,max}$ = 0.5','$p_{c,max}$ = 0.75'],fontsize=8)
# ax[1][1].set_xticks([0,10,20,30])
# ax[1][1].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[1][1].set_ylim([0,7])
# ax[1][1].set_yticklabels(['0','1','2'],fontsize=8)

ax1 = fig.add_subplot(111, frame_on=False)
ax1.tick_params(labelcolor="none", bottom=False, left=False)
ax1.set_xlabel('Year',fontsize=14)
ax1.set_ylabel('Biofuel Production (Billion L)',fontsize=14)
fig.savefig("./biofuel_production_sensi.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


plt.close('all')
adj = np.ones(21)*0.92
adj[:6] = 1
fig, ax = plt.subplots(2,2,figsize=(6, 6))
ax[0][0].plot(result_IRR1['land_use_areas'][:,2]/10**6,linestyle='solid', linewidth=lw)
ax[0][0].plot(result_SSB_PC1['land_use_areas'][:,2]/10**6*adj,linestyle='dashed', linewidth=lw)
ax[0][0].plot(result_IRR3['land_use_areas'][:,2]/10**6,linestyle='dotted', linewidth=lw)
ax[0][0].legend(['$IRR_{min,cell} = 0.2$','$IRR_{min,cell} = 0.25$','$IRR_{min,cell} = 0.3$'],fontsize=8)
ax[0][0].set_xticks([])
ax[0][0].set_ylim([0,0.2])
# ax[0][0].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[0][1].plot(result_AT1['land_use_areas'][:,2]/10**6,linestyle='solid', linewidth=lw)
ax[0][1].plot(result_SSB_PC1['land_use_areas'][:,2]/10**6*adj,linestyle='dashed', linewidth=lw)
ax[0][1].plot(result_AT3['land_use_areas'][:,2]/10**6,linestyle='dotted', linewidth=lw)
ax[0][1].legend(['$\zeta_w = -0.1$','$\zeta_w = 0$','$\zeta_w = 0.1$'],fontsize=8)
ax[0][1].set_xticks([])
ax[0][1].set_ylim([0,0.2])
# ax[0][1].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[1][0].plot(result_NI1['land_use_areas'][:,2]/10**6,linestyle='solid', linewidth=lw)
ax[1][0].plot(result_SSB_PC1['land_use_areas'][:,2]/10**6*adj,linestyle='dashed', linewidth=lw)
ax[1][0].plot(result_NI3['land_use_areas'][:,2]/10**6,linestyle='dotted', linewidth=lw)
ax[1][0].legend(['$\Xi = 0$','$\Xi = 0.5$','$\Xi = 1$'],fontsize=8)
# ax[1][0].set_xticks([])
ax[1][0].set_ylim([0,0.2])
# ax[1][0].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[1][1].plot(result_WP0['land_use_areas'][:,2]/10**6,linestyle='solid', linewidth=lw)
ax[1][1].plot(result_SSB_PC1['land_use_areas'][:,2]/10**6*adj,linestyle='dashed', linewidth=lw)
ax[1][1].plot(result_WP2['land_use_areas'][:,2]/10**6,linestyle='dotted', linewidth=lw)
# ax[1][1].plot(result_LBD2['land_use_areas'][:,2]/10**6)
ax[1][1].legend(['$p_{c,max}$ = 0.01','$p_{c,max}$ = 0.5','$p_{c,max}$ = 0.75'],fontsize=8)
# ax[1][1].set_xticks([])
ax[1][1].set_ylim([0,0.2])
# ax[1][1].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax1 = fig.add_subplot(111, frame_on=False)
ax1.tick_params(labelcolor="none", bottom=False, left=False)
ax1.set_xlabel('Year',fontsize=14)
ax1.set_ylabel('Miscanthus Acreage (Million ha)',fontsize=14,labelpad=10)
fig.savefig("./mis_area_sensi.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)



# plot perennial adoption & biomass price
corn_eth_produce = result_SSB_PC1['biofuel_production'][result_SSB_PC1['biofuel_type']==1]
corn_eth_produce = corn_eth_produce.sum(0)
cell_eth_produce = result_SSB_PC1['biofuel_production'][result_SSB_PC1['biofuel_type']==2]
cell_eth_produce = cell_eth_produce.sum(0)
RFS_mandate = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/RFS_volume.csv')
data_SSB_PC1 = shelve.open(dir_SSB_PC1)

prices_rfs = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/external_provided_prices.csv')
cell_eth_price_rfs = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/endogenous_ethanol_price.csv')
cell_eth_price_baseline = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/data/endogenous_ethanol_price.csv')

fig, ax = plt.subplots(2,2,figsize=(12, 6))
lns1=ax[0][0].plot(prices_rfs['ethanol ($/L)'][:20],label='Corn Ethanol',linewidth=2)
lns2=ax[0][0].plot(cell_eth_price_rfs['endogenous_ethanol ($/L)'][:20],label='Cellulosic Ethanol',linewidth=2,linestyle='dashed')
ax[0][0].set_ylabel('Ethanol Price ($/L)',fontsize=16)
ax[0][0].set_xticks([])
# ax1 = ax[0][0].twinx()
# lns3=ax1.plot(prices_rfs['mis ($/ton)'][:20],color='r',label='Biomass Price',linewidth=2)
# ax1.set_xticks([])
# ax1.tick_params(axis='y')
# ax1.set_ylabel('Biomass Price ($/t)')
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax[0][0].legend(lns, labs, loc="lower right",fontsize=14)

color = 'tab:blue'
lns1=ax[0][1].plot(result_SSB_PC1['att_farmer'],color=color,label='Willingness',linewidth=2)
ax[0][1].set_ylabel("Farmer's willingness \n to adopt Miscanthus",fontsize=16)
ax[0][1].tick_params(axis='y', labelcolor=color)
ax2 = ax[0][1].twinx()
color = 'tab:red'
lns2=ax2.plot(result_SSB_PC1['environ_sensi_farmer'],color=color,linewidth=2,label='Environment Attitude',linestyle='dashed')
ax2.set_ylabel("Farmer's attitude \n  toward environment",fontsize=16)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xticks([])
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0,fontsize=14)

ax[1][0].plot(corn_eth_produce/10**9,linewidth=2)
ax[1][0].plot(cell_eth_produce/10**9,linewidth=2,linestyle='dashed')
ax[1][0].plot(RFS_mandate['RFS_volume (Million L)'][:20]/1000,'r',linewidth=2,linestyle='dashdot')
ax[1][0].legend(['Corn Ethanol','Cellulosic Ethanol','RFS Cellulosic Mandate'],loc='lower right',fontsize=14)
ax[1][0].set_ylabel('Ethanol Production \n (Billion L)',fontsize=16)
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][0].set_xlabel('Year',fontsize=16)

ax[1][1].plot(cal_stats.cal_min_IRR_RFS_policy_commitment(data_SSB_PC1['ABM_parameters']['RFS_signal'])[:20],linewidth=2)
ax[1][1].set_ylabel('Refinery Minimum IRR',fontsize=16)
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlabel('Year',fontsize=16)

plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
fig.savefig("./system_dynamics.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
