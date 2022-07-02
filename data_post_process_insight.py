import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import cal_stats
import pandas as pd
import geopandas as gpd
from PIL import Image, ImageDraw
import os
import shelve
import cal_stats
import parameters as params

def cal_biofacility_feed_use(data):

    N_ref = data['ref_list_backup'].__len__()
    simu_horizon = params.simu_horizon
    ref_list = data['ref_list_backup']
    output = np.zeros((N_ref, simu_horizon))
    for j in range(N_ref):
        temp = 0
        for i in range(simu_horizon):
            if i < ref_list[j].Attributes['start_year']:
                output[j, i] = 0
                temp = i + 1
            elif (i - temp) < ref_list[j].States['byproduct_production'].__len__():
                if ref_list[j].Attributes['tech_type'] == 6:  # if it is about cofire
                    output[j, i] = ref_list[j].States['byproduct_production'][i - temp][3]/1.68
                elif ref_list[j].Attributes['tech_type'] == 7: # if it is about BCHP
                    output[j, i] = ref_list[j].States['byproduct_production'][i - temp][3]/0.96

    return output

def cal_mean_price(data):

    N_ref = data['ref_list_backup'].__len__()
    simu_horizon = params.simu_horizon
    ref_list = data['ref_list_backup']
    cum_price = np.zeros(simu_horizon)
    cum_feed = np.zeros(simu_horizon)
    total_feed = np.zeros(simu_horizon)
    for i in range(simu_horizon):
        for temp_farmer in data['farmer_agent_list']:
            for j in range(temp_farmer.Attributes['patch_ID'].size):
                if temp_farmer.States['land_use'][i][j] == 1:
                    total_feed[i] += temp_farmer.States['yield'][i][j] * 0.1 \
                                     * temp_farmer.Attributes['patch_areas'][j]
                elif temp_farmer.States['land_use'][i][j] == 3:
                    total_feed[i] += temp_farmer.States['yield'][i][j] * \
                                     temp_farmer.Attributes['patch_areas'][j]
        for j in range(N_ref):
            s_year = ref_list[j].Attributes['start_year']
            if (i >= s_year) & ((i+1 - s_year) < ref_list[j].States['byproduct_production'].__len__()):
                cum_price[i] += (ref_list[j].States['contracted_patch_price'][i+1-s_year] *
                                 ref_list[j].States['contracted_patch_amount'][i+1-s_year][:,3]).sum()
                cum_feed[i] += ref_list[j].States['contracted_patch_amount'][i+1-s_year][:,3].sum()
        cum_price[i] += (total_feed[i]-cum_feed[i]) * data['feed_prices_hist'][i][3]
        cum_feed[i] += total_feed[i]-cum_feed[i]
    return cum_price/cum_feed

dir_SQ = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_SQ.out' # status quo
dir_SSB_0 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_SSB_0.out' # small sc
dir_SSB_1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_SSB_1.out' # small sc
dir_PC1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_PC1.out' # no policy commitment
dir_SSB_PC1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1.out' # small scale biofacility
dir_PC1_SSB1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_PC1_SSB1.out' # small scale biofacility
dir_newtech = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_newtech.out' # new technology
dir_newtech_SSB = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_newtech_SSB.out' # new technology
dir_newtech_SSB1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_newtech_SSB1.out' # new technology
dir_LBD0 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_LBD0.out' # no learning by doing
dir_LBD0_SSB = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_LBD0_SSB.out' # no learning by doing
dir_LBD0_SSB1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_LBD0_SSB1.out' # no learning by doing
dir_LBD0_TMDL1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_TMDL1.out' # no learning by doing
dir_CRP1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_CRP1.out' # status quo
dir_WP2 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_WP2.out' # status quo
dir_CNT = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_IS_CNT.out' # status quo

result_SQ = cal_stats.extract_ABM_results(dir_SQ,1)
result_SSB_0 = cal_stats.extract_ABM_results(dir_SSB_0,1)
result_SSB_1 = cal_stats.extract_ABM_results(dir_SSB_1,1)
result_PC1 = cal_stats.extract_ABM_results(dir_PC1,1)
result_SSB_PC1 = cal_stats.extract_ABM_results(dir_SSB_PC1,1)
result_PC1_SSB1 = cal_stats.extract_ABM_results(dir_PC1_SSB1,1)
result_newtech = cal_stats.extract_ABM_results(dir_newtech,1)
result_newtech_SSB = cal_stats.extract_ABM_results(dir_newtech_SSB,1)
result_newtech_SSB1 = cal_stats.extract_ABM_results(dir_newtech_SSB1,1)
result_LBD0 = cal_stats.extract_ABM_results(dir_LBD0,1)
result_LBD0_SSB = cal_stats.extract_ABM_results(dir_LBD0_SSB,1)
result_LBD0_SSB1 = cal_stats.extract_ABM_results(dir_LBD0_SSB1,1)
result_TMDL1 = cal_stats.extract_ABM_results(dir_LBD0_TMDL1,1)
result_CRP1 = cal_stats.extract_ABM_results(dir_CRP1,1)
result_WP2 = cal_stats.extract_ABM_results(dir_WP2,1)
result_CNT = cal_stats.extract_ABM_results(dir_CNT,1)


data_SQ = shelve.open(dir_SQ)
data_SSB_0 = shelve.open(dir_SSB_0)
data_SSB_1 = shelve.open(dir_SSB_1)
data_PC1 = shelve.open(dir_PC1)
data_SSB_PC1 = shelve.open(dir_SSB_PC1)
data_PC1_SSB1 = shelve.open(dir_PC1_SSB1)
data_newtech = shelve.open(dir_newtech)
data_newtech_SSB = shelve.open(dir_newtech_SSB)
data_newtech_SSB1 = shelve.open(dir_newtech_SSB1)
data_LBD0 = shelve.open(dir_LBD0)
data_LBD0_SSB = shelve.open(dir_LBD0_SSB)
data_LBD0_SSB1 = shelve.open(dir_LBD0_SSB1)
data_TMDL1 = shelve.open(dir_LBD0_TMDL1)
data_CRP1 = shelve.open(dir_CRP1)
data_WP2 = shelve.open(dir_WP2)
data_CNT = shelve.open(dir_CNT)
os.chdir('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/methodology')



plt.close('all')
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_SSB_0['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_LBD0['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_LBD0_SSB['land_use_areas'][:,2]/10**6)
# ax[0].legend(['SQ','SSB','PC','PC_SSB'])
ax[0][0].set_ylim([0,0.12])
ax[0][0].set_yticks([0,0.03,0.06,0.09,0.12])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].set_ylabel('Miscanthus Acreage (Million ha)')
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_SSB_0['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_LBD0['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_LBD0_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([0,3])
ax[1][0].set_yticks([0,0.5,1,1.5,2,2.5,3])
ax[1][0].set_yticklabels(['0.00','0.50','1.00','1.50','2.00','2.50','3.00'])
# ax[1].legend(['SQ','SSB','PC','PC_SSB'])
ax[1][0].set_xlabel('Year')
ax[1][0].set_ylabel('Biofuel Production (Billion L)')
ax[1][0].grid(True)

ax[0][1].plot(cal_biofacility_feed_use(data_SQ).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_SSB_0).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_LBD0).sum(0)/10**6)
adj = np.ones(20)*511904
adj[:5]=0
ax[0][1].plot((cal_biofacility_feed_use(data_LBD0_SSB).sum(0))/10**6)
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[0][1].set_ylabel('Biofacility Biomass Consumption \n (Million ton)')
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['att_farmer'])
ax[1][1].plot(result_SSB_0['att_farmer'])
ax[1][1].plot(result_LBD0['att_farmer'])
ax[1][1].plot(result_LBD0_SSB['att_farmer'])
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[1][1].set_xlabel('Year')
ax[1][1].set_ylabel('Average Farmer Willingness')
ax[1][1].grid(True)
plt.legend(['SQ','SSB','PC','PC_SSB'],loc = 'lower left', bbox_to_anchor = (1.05, 0))
plt.tight_layout()
fig.savefig("./scaling_up_impact.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


plt.close('all')
ft_size=12
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:,2]/10**6)
# ax[0][0].plot(result_LBD0_SSB['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_SSB_0['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_newtech['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_newtech_SSB['land_use_areas'][:,2]/10**6)
ax[0][0].set_ylim([-0.01,0.15])
ax[0][0].set_yticks([0,0.03,0.06,0.09,0.12])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].set_ylabel('Miscanthus Acreage (Million ha)',fontsize = ft_size)
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9)
# ax[1][0].plot(result_SSB_0['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_LBD0_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_newtech['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_newtech_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([0,5])
ax[1][0].set_yticks([0,1,2,3,4])
ax[1][0].set_yticklabels(['0.00','1.00','2.00','3.00','4.00'])
ax[1][0].set_ylabel('Biofuel Production (Billion L)',fontsize = ft_size)
ax[1][0].set_xlabel('Year',fontsize = ft_size)
ax[1][0].grid(True)

ax[0][1].plot(cal_biofacility_feed_use(data_SQ).sum(0)/10**6)
# ax[0][1].plot(cal_biofacility_feed_use(data_SSB_0).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_LBD0_SSB).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech_SSB).sum(0)/10**6)
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[0][1].set_ylabel('Biofacility Biomass Consumption \n (Million ton)',fontsize = ft_size)
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['att_farmer'])
# ax[1][1].plot(result_SSB_0['att_farmer'])
ax[1][1].plot(result_LBD0_SSB['att_farmer'])
ax[1][1].plot(result_newtech['att_farmer'])
ax[1][1].plot(result_newtech_SSB['att_farmer'])
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[1][1].set_xlabel('Year',fontsize = ft_size)
ax[1][1].set_ylabel('Average Farmer Willingness',fontsize = ft_size)
ax[1][1].grid(True)
plt.legend(['Status Quo','\nPolicy Commitment',
            '\nPolicy Commitment \n+ Small Scale Facility',
            '\nNew Tech \n+ Policy Commitment \n+ Small Scale Facility'],loc = 'lower left', bbox_to_anchor = (1.05, 0))
plt.tight_layout()
fig.savefig("./new_tech_path.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)



plt.close('all')
ft_size=16
lw=2.5
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(7, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:20,2]/10**6,'k:', linewidth=lw)
ax[0][0].plot(result_LBD0_SSB['land_use_areas'][:20,2]/10**6,'b--', linewidth=lw)
# ax[0][0].plot(result_SSB_0['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_newtech['land_use_areas'][:20,2]/10**6,'g-.', linewidth=lw)
ax[0][0].plot(result_newtech_SSB['land_use_areas'][:20,2]/10**6,'r', linewidth=lw)
ax[0][0].set_ylim([-0.01,0.15])
ax[0][0].set_yticks([0,0.03,0.06,0.09,0.12])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].set_ylabel('Miscanthus Acreage \n(M ha)',fontsize = ft_size)
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9,'k:', linewidth=lw)
# ax[1][0].plot(result_SSB_0['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_LBD0_SSB['biofuel_production'].sum(axis=0)/10**9,'b--', linewidth=lw)
ax[1][0].plot(result_newtech['biofuel_production'].sum(axis=0)/10**9,'g-.', linewidth=lw)
ax[1][0].plot(result_newtech_SSB['biofuel_production'].sum(axis=0)/10**9,'r', linewidth=lw)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([0,5])
ax[1][0].set_yticks([0,1,2,3,4])
ax[1][0].set_yticklabels(['0.00','1.00','2.00','3.00','4.00'])
ax[1][0].set_ylabel('Biofuel Production \n(Billion L)',fontsize = ft_size)
ax[1][0].set_xlabel('Year',fontsize = ft_size)
ax[1][0].grid(True)

ax[0][1].plot(cal_biofacility_feed_use(data_SQ).sum(0)/10**6,'k:', linewidth=lw)
# ax[0][1].plot(cal_biofacility_feed_use(data_SSB_0).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_LBD0_SSB).sum(0)/10**6,'b--', linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech).sum(0)/10**6,'g-.', linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech_SSB).sum(0)/10**6,'r', linewidth=lw)
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[0][1].set_ylabel('Small-Scale Biofacility \nBiomass Consump.\n (M ton)',fontsize = ft_size)
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['att_farmer'][:20],'k:', linewidth=lw)
# ax[1][1].plot(result_SSB_0['att_farmer'])
ax[1][1].plot(result_LBD0_SSB['att_farmer'][:20],'b--', linewidth=lw)
ax[1][1].plot(result_newtech['att_farmer'][:20],'g-.', linewidth=lw)
ax[1][1].plot(result_newtech_SSB['att_farmer'][:20],'r', linewidth=lw)
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[1][1].set_xlabel('Year',fontsize = ft_size)
ax[1][1].set_ylabel('Ave. Farmer \nWillingness',fontsize = ft_size)
ax[1][1].grid(True)
fig.subplots_adjust(top=0.8, left=0.05, right=0.95, bottom=0.05)
fig.legend(['Status Quo',
            'Policy Commitment \n+ Small Scale Facility', 'Policy Commitment + New Tech',
            'Policy Commitment \n+ Small Scale Facility +New Tech'],loc = 'upper center', bbox_to_anchor = (0.53, 1.15),
           ncol=2,fontsize = 14,frameon=False)
plt.tight_layout()
fig.savefig("./highlight.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)




plt.close('all')
ft_size=12
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(8, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:,2]/10**6)
# ax[0][0].plot(result_LBD0_SSB['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_SSB_0['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_newtech['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_newtech_SSB['land_use_areas'][:,2]/10**6)
ax[0][0].set_ylim([-0.01,0.15])
ax[0][0].set_yticks([0,0.03,0.06,0.09,0.12])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].set_ylabel('Miscanthus Acreage (Million ha)',fontsize = ft_size)
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9)
# ax[1][0].plot(result_SSB_0['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_LBD0_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_newtech['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_newtech_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([0,5])
ax[1][0].set_yticks([0,1,2,3,4])
ax[1][0].set_yticklabels(['0.00','1.00','2.00','3.00','4.00'])
ax[1][0].set_ylabel('Biofuel Production (Billion L)',fontsize = ft_size)
ax[1][0].set_xlabel('Year',fontsize = ft_size)
ax[1][0].grid(True)

ax[0][1].plot(cal_biofacility_feed_use(data_SQ).sum(0)/10**6)
# ax[0][1].plot(cal_biofacility_feed_use(data_SSB_0).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_LBD0_SSB).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech).sum(0)/10**6)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech_SSB).sum(0)/10**6)
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[0][1].set_ylabel('Biofacility Biomass Consumption \n (Million ton)',fontsize = ft_size)
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['att_farmer'])
# ax[1][1].plot(result_SSB_0['att_farmer'])
ax[1][1].plot(result_LBD0_SSB['att_farmer'])
ax[1][1].plot(result_newtech['att_farmer'])
ax[1][1].plot(result_newtech_SSB['att_farmer'])
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[1][1].set_xlabel('Year',fontsize = ft_size)
ax[1][1].set_ylabel('Average Farmer Willingness',fontsize = ft_size)
ax[1][1].grid(True)
plt.legend(['Status Quo','\nPolicy Commitment',
            '\nPolicy Commitment \n+ Small Scale Facility',
            '\nNew Tech \n+ Policy Commitment \n+ Small Scale Facility'],loc = 'lower left', bbox_to_anchor = (1.05, 0))
plt.tight_layout()
fig.savefig("./new_tech_path.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)



plt.close('all')
ft_size=16
lw=2.5
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(7, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_CRP1['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_newtech['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_CNT['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_SSB_0['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_LBD0['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_WP2['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].set_ylim([-0.01,0.15])
ax[0][0].set_yticks([0,0.03,0.06,0.09,0.12])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].set_ylabel('Miscanthus Acreage \n(M ha)',fontsize = ft_size)
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[1][0].plot(result_CRP1['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[1][0].plot(result_newtech['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[1][0].plot(result_CNT['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[1][0].plot(result_SSB_0['biofuel_production'].sum(axis=0)/10**9,linewidth=lw)
ax[1][0].plot(result_LBD0['biofuel_production'].sum(axis=0)/10**9,linewidth=lw)
ax[1][0].plot(result_WP2['biofuel_production'].sum(axis=0)/10**9,linewidth=lw)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([0,5])
ax[1][0].set_yticks([0,1,2,3,4])
ax[1][0].set_yticklabels(['0.00','1.00','2.00','3.00','4.00'])
ax[1][0].set_ylabel('Biofuel Production \n(Billion L)',fontsize = ft_size)
ax[1][0].set_xlabel('Year',fontsize = ft_size)
ax[1][0].grid(True)

ax[0][1].plot(cal_biofacility_feed_use(data_SQ).sum(0)/10**6, linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_CRP1).sum(0)/10**6, linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_newtech).sum(0)/10**6, linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_CNT).sum(0)/10**6, linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_SSB_0).sum(0)/10**6,linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_LBD0).sum(0)/10**6,linewidth=lw)
ax[0][1].plot(cal_biofacility_feed_use(data_WP2).sum(0)/10**6,linewidth=lw)
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[0][1].set_ylabel('Biofacility Biomass \nConsump. (M ton)',fontsize = ft_size)
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['att_farmer'][:20], linewidth=lw)
ax[1][1].plot(result_CRP1['att_farmer'][:20], linewidth=lw)
ax[1][1].plot(result_newtech['att_farmer'][:20], linewidth=lw)
ax[1][1].plot(result_CNT['att_farmer'][:20], linewidth=lw)
ax[1][1].plot(result_SSB_0['att_farmer'][:20], linewidth=lw)
ax[1][1].plot(result_LBD0['att_farmer'][:20], linewidth=lw)
ax[1][1].plot(result_WP2['att_farmer'][:20], linewidth=lw)
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
# ax[2].set_ylim([0,2])
# ax[2].set_yticks([0,0.5,1,1.5,2])
# ax[2].set_yticklabels(['0.00','0.50','1.00','1.50','2.00'])
ax[1][1].set_xlabel('Year',fontsize = ft_size)
ax[1][1].set_ylabel('Ave. Farmer \nWillingness',fontsize = ft_size)
ax[1][1].grid(True)
fig.subplots_adjust(top=0.8, left=0.05, right=0.95, bottom=0.05)
fig.legend(['SQ','CRP1','newtech','CNT','SSB','PC1','WP2'],loc = 'upper center', bbox_to_anchor = (0.53, 1.15),
           ncol=3,fontsize = 14,frameon=False)
plt.tight_layout()
fig.savefig("./commu_tool.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)



plt.close('all')
ft_size=14
lw=2.5
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(7, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][0].plot(result_CNT['land_use_areas'][:20,2]/10**6, linewidth=lw,linestyle='dashed')
ax[0][0].plot(result_CRP1['land_use_areas'][:20,2]/10**6, linewidth=lw,linestyle='dotted')
ax[0][0].set_ylim([-0.01,0.13])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].legend(['SQ','C-MARKET','CRP'],fontsize=ft_size)
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[1][0].plot(result_newtech['land_use_areas'][:20,2]/10**6, linewidth=lw,linestyle='dashed')
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([-0.01,0.13])
ax[1][0].legend(['SQ','TECH'],fontsize=ft_size)
ax[1][0].grid(True)

ax[0][1].plot(result_SQ['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[0][1].plot(result_SSB_0['land_use_areas'][:20,2]/10**6, linewidth=lw,linestyle='dashed')
ax[0][1].plot(result_LBD0['land_use_areas'][:20,2]/10**6, linewidth=lw,linestyle='dotted')
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
ax[0][1].set_ylim([-0.01,0.13])
ax[0][1].set_yticklabels([])
ax[0][1].legend(['SQ','S-FACILITY','RFS'],fontsize=ft_size)
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['land_use_areas'][:20,2]/10**6, linewidth=lw)
ax[1][1].plot(result_WP2['land_use_areas'][:20,2]/10**6, linewidth=lw,linestyle='dashed')
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
ax[1][1].set_ylim([-0.01,0.13])
ax[1][1].set_yticklabels([])
ax[1][1].legend(['SQ','EDU'],fontsize=ft_size)
ax[1][1].grid(True)

ax[0][0].set_title('Economic Point of View',fontsize = ft_size)
ax[1][0].set_title('Technology Point of View',fontsize = ft_size)
ax[0][1].set_title('Industry Stakeholder \nPoint of View',fontsize = ft_size)
ax[1][1].set_title('Community Stakeholder \nPoint of View',fontsize = ft_size)
fig.text(0.5, 0, 'Year', ha='center', va='center',fontsize = ft_size)
fig.text(0, 0.5, 'Miscanthus Acreage (M ha)', ha='center', va='center', rotation='vertical',fontsize = ft_size)
plt.tight_layout()
fig.savefig("./commu_tool_mis.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


plt.close('all')
ft_size=14
lw=2.5
mpl.style.use('default')
fig, ax = plt.subplots(2,2,figsize=(7, 6))
ax[0][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[0][0].plot(result_CNT['biofuel_production'].sum(axis=0)/10**9, linewidth=lw,linestyle='dashed')
ax[0][0].plot(result_CRP1['biofuel_production'].sum(axis=0)/10**9, linewidth=lw,linestyle='dotted')
ax[0][0].set_ylim([0,5])
ax[0][0].set_xticks([0,5,10,15,20])
ax[0][0].set_xlim([0,20])
ax[0][0].set_xticklabels([])
ax[0][0].legend(['SQ','C-MARKET','CRP'],fontsize=ft_size)
ax[0][0].grid(True)

ax[1][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[1][0].plot(result_newtech['biofuel_production'].sum(axis=0)/10**9, linewidth=lw,linestyle='dashed')
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlim([0,20])
ax[1][0].set_ylim([0,5])
ax[1][0].legend(['SQ','TECH'],fontsize=ft_size)
ax[1][0].grid(True)

ax[0][1].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[0][1].plot(result_SSB_0['biofuel_production'].sum(axis=0)/10**9, linewidth=lw,linestyle='dashed')
ax[0][1].plot(result_LBD0['biofuel_production'].sum(axis=0)/10**9, linewidth=lw,linestyle='dotted')
ax[0][1].set_xticks([0,5,10,15,20])
ax[0][1].set_xlim([0,20])
ax[0][1].set_xticklabels([])
ax[0][1].set_ylim([0,5])
ax[0][1].set_yticklabels([])
ax[0][1].legend(['SQ','S-FACILITY','RFS'],fontsize=ft_size)
ax[0][1].grid(True)

ax[1][1].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9, linewidth=lw)
ax[1][1].plot(result_WP2['biofuel_production'].sum(axis=0)/10**9, linewidth=lw,linestyle='dashed')
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlim([0,20])
ax[1][1].set_ylim([0,5])
ax[1][1].set_yticklabels([])
ax[1][1].legend(['SQ','EDU'],fontsize=ft_size)
ax[1][1].grid(True)

ax[0][0].set_title('Economic Point of View',fontsize = ft_size)
ax[1][0].set_title('Technology Point of View',fontsize = ft_size)
ax[0][1].set_title('Industry Stakeholder \nPoint of View',fontsize = ft_size)
ax[1][1].set_title('Community Stakeholder \nPoint of View',fontsize = ft_size)
fig.text(0.5, 0, 'Year', ha='center', va='center',fontsize = ft_size)
fig.text(0, 0.5, 'Biofuel Production (Billion L)', ha='center', va='center', rotation='vertical',fontsize = ft_size)
plt.tight_layout()
fig.savefig("./commu_tool_biofuel.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
