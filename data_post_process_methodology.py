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

dir_SQ = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_SQ.out' # status quo
dir_SSB = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_SSB.out' # small scale biofacility
dir_SSB_PC1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/ABM_result_MD_SSB_PC1.out' # small scale biofacility
dir_TMDL1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_TMDL1.out' # TMDL subsidy
dir_PC1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_PC1.out' # no policy commitment
dir_CRP1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_CRP1.out' # CRP harvesting relaxation
dir_LBD0 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_LBD0.out' # no learning by doing
dir_LBD2 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_LBD2.out' # high learning by doing
dir_NI1 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_NI1.out' # lower neighborhood impact
dir_NI3 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_NI3.out' # higher neighborhood impact
dir_WP2 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_WP2.out' # higher neighborhood impact
dir_WP0 = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_WP0.out' # higher neighborhood impact
dir_newtech = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/results/1126/ABM_result_MD_newtech.out' # new technology


# s_V2_baseline = shelve.open(dir_V2_baseline+'/results/ABM_result.out')

result_SQ = cal_stats.extract_ABM_results(dir_SQ,1)
result_SSB = cal_stats.extract_ABM_results(dir_SSB,1)
result_SSB_PC1 = cal_stats.extract_ABM_results(dir_SSB_PC1,1)
result_TMDL1 = cal_stats.extract_ABM_results(dir_TMDL1,1)
result_PC1 = cal_stats.extract_ABM_results(dir_PC1,1)
result_CRP1 = cal_stats.extract_ABM_results(dir_CRP1,1)
result_LBD0 = cal_stats.extract_ABM_results(dir_LBD0,1)
result_LBD2 = cal_stats.extract_ABM_results(dir_LBD2,1)
result_NI1 = cal_stats.extract_ABM_results(dir_NI1,1)
result_NI3 = cal_stats.extract_ABM_results(dir_NI3,1)
result_WP2 = cal_stats.extract_ABM_results(dir_WP2,1)
result_WP0 = cal_stats.extract_ABM_results(dir_WP0,1)
result_newtech = cal_stats.extract_ABM_results(dir_newtech,1)


os.chdir('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/methodology')
# plot the total miscanthus land use
# all
plt.close('all')
fig, ax = plt.subplots(3,2,figsize=(6, 6))
ax[0][0].plot(result_SQ['land_use_areas'][:,2]/10**6)
ax[0][0].plot(result_SSB['land_use_areas'][:,2]/10**6)
ax[0][0].legend(['Status Quo','Small Scale \n Biofacility Subsudy'],fontsize=8)
ax[0][0].set_xticks([])
ax[0][0].set_ylim([0,0.15])
ax[0][0].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[0][1].plot(result_SSB['land_use_areas'][:,2]/10**6)
ax[0][1].plot(result_SSB_PC1['land_use_areas'][:,2]/10**6)
ax[0][1].legend(['No Policy Commitment','Policy Commitment'],fontsize=8)
ax[0][1].set_xticks([])
ax[0][1].set_ylim([0,0.15])
ax[0][1].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[1][0].plot(result_PC1['land_use_areas'][:,2]/10**6)
ax[1][0].plot(result_CRP1['land_use_areas'][:,2]/10**6)
ax[1][0].legend(['No CRP relaxation','CRP relaxation'],fontsize=8)
ax[1][0].set_xticks([])
ax[1][0].set_ylim([0,0.15])
ax[1][0].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[1][1].plot(result_LBD0['land_use_areas'][:,2]/10**6)
ax[1][1].plot(result_PC1['land_use_areas'][:,2]/10**6)
# ax[1][1].plot(result_LBD2['land_use_areas'][:,2]/10**6)
ax[1][1].legend(['No learning-by-doing','Learning-by-doing'],fontsize=8)
ax[1][1].set_xticks([])
ax[1][1].set_ylim([0,0.15])
ax[1][1].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

# ax[2][0].plot(result_NI1['land_use_areas'][:,2]/10**6)
ax[2][0].plot(result_PC1['land_use_areas'][:,2]/10**6)
ax[2][0].plot(result_NI3['land_use_areas'][:,2]/10**6)
ax[2][0].legend(['Low neighborhood impact','High neighborhood impact'],fontsize=8)
# ax[2][0].set_xticks([0,10,20,30])
# ax[2][0].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][0].set_ylim([0,0.15])
ax[2][0].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax[2][1].plot(result_PC1['land_use_areas'][:,2]/10**6)
ax[2][1].plot(result_WP2['land_use_areas'][:,2]/10**6)
ax[2][1].legend(['Low willingness to pay','High willingness to pay'],fontsize=8)
# ax[2][1].set_xticks([0,10,20,30])
# ax[2][1].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][1].set_ylim([0,0.15])
ax[2][1].set_yticklabels(['0','0.05','0.1','0.15'],fontsize=8)

ax1 = fig.add_subplot(111, frame_on=False)
ax1.tick_params(labelcolor="none", bottom=False, left=False)
ax1.set_xlabel('Year',fontsize=14)
ax1.set_ylabel('Area (Million ha)',fontsize=14)
fig.savefig("./mis_area_all.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# commitment
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_SSB['land_use_areas'][:,2]/10**6)
ax.plot(result_PC1['land_use_areas'][:,2]/10**6)
ax.legend(['No Policy Commitment','Policy Commitment'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Area (Million ha)',fontsize=18)
fig.savefig("./mis_area_price_commitment.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# CRP
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_PC1['land_use_areas'][:,2]/10**6)
ax.plot(result_CRP1['land_use_areas'][:,2]/10**6)
ax.legend(['No CRP relaxation','CRP relaxation'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Area (Million ha)',fontsize=18)
fig.savefig("./mis_area_price_CRP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# learning by doing
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_LBD0['land_use_areas'][:,2]/10**6)
ax.plot(result_PC1['land_use_areas'][:,2]/10**6)
ax.plot(result_LBD2['land_use_areas'][:,2]/10**6)
ax.legend(['No learning-by-doing','Low learning-by-doing','High learning-by-doing'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Area (Million ha)',fontsize=18)
fig.savefig("./mis_area_price_LBD.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# neighborhood impact
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_NI1['land_use_areas'][:,2]/10**6)
ax.plot(result_PC1['land_use_areas'][:,2]/10**6)
ax.plot(result_NI3['land_use_areas'][:,2]/10**6)
ax.legend(['Low neighborhood impact','Median neighborhood impact','High neighborhood impact'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Area (Million ha)',fontsize=18)
fig.savefig("./mis_area_price_NI.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# Willingness to pay
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_WP0['land_use_areas'][:,2]/10**6)
ax.plot(result_WP2['land_use_areas'][:,2]/10**6)
ax.legend(['Low willingness to pay','High willingness to pay'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Area (Million ha)',fontsize=18)
fig.savefig("./mis_area_price_WP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


# plot the total biofuel production
# all
plt.close('all')
fig, ax = plt.subplots(3,2,figsize=(6, 6))
ax[0][0].plot(result_SQ['biofuel_production'].sum(axis=0)/10**9)
ax[0][0].plot(result_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[0][0].legend(['Status Quo','Small Scale \n Biofacility Subsudy'],fontsize=8)
ax[0][0].set_xticks([])
ax[0][0].set_ylim([0,4])
ax[0][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[0][1].plot(result_SSB['biofuel_production'].sum(axis=0)/10**9)
ax[0][1].plot(result_SSB_PC1['biofuel_production'].sum(axis=0)/10**9)
ax[0][1].legend(['No Policy Commitment','Policy Commitment'],fontsize=8)
ax[0][1].set_xticks([])
ax[0][1].set_ylim([0,4])
ax[0][1].set_yticklabels(['0','1','2'],fontsize=8)

ax[1][0].plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].plot(result_CRP1['biofuel_production'].sum(axis=0)/10**9)
ax[1][0].legend(['No CRP relaxation','CRP relaxation'],fontsize=8)
ax[1][0].set_xticks([])
ax[1][0].set_ylim([0,4])
ax[1][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[1][1].plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax[1][1].plot(result_LBD0['biofuel_production'].sum(axis=0)/10**9)
# ax[1][1].plot(result_LBD2['biofuel_production'].sum(axis=0)/10**9)
ax[1][1].legend(['No learning-by-doing','Learning-by-doing'],fontsize=8)
ax[1][1].set_xticks([])
ax[1][1].set_ylim([0,4])
ax[1][1].set_yticklabels(['0','1','2'],fontsize=8)

ax[2][0].plot(result_NI1['biofuel_production'].sum(axis=0)/10**9)
# ax[2][0].plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax[2][0].plot(result_NI3['biofuel_production'].sum(axis=0)/10**9)
ax[2][0].legend(['Low neighborhood impact','High neighborhood impact'],fontsize=8)
# ax[2][0].set_xticks([0,10,20,30])
# ax[2][0].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][0].set_ylim([0,4])
ax[2][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[2][1].plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax[2][1].plot(result_WP2['biofuel_production'].sum(axis=0)/10**9)
ax[2][1].legend(['Low willingness to pay','High willingness to pay'],fontsize=8)
# ax[2][1].set_xticks([0,10,20,30])
# ax[2][1].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][1].set_ylim([0,4])
ax[2][1].set_yticklabels(['0','1','2'],fontsize=8)

ax1 = fig.add_subplot(111, frame_on=False)
ax1.tick_params(labelcolor="none", bottom=False, left=False)
ax1.set_xlabel('Year',fontsize=14)
ax1.set_ylabel('Biofuel Production (Billion L)',fontsize=14)
fig.savefig("./biofuel_production_all.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# commitment
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_SSB['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax.legend(['No Policy Commitment','Policy Commitment'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Biofuel Production (Billion L)',fontsize=18)
fig.savefig("./biofuel_production_commitment.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# CRP
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_CRP1['biofuel_production'].sum(axis=0)/10**9)
ax.legend(['No CRP relaxation','CRP relaxation'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Biofuel Production (Billion L)',fontsize=18)
fig.savefig("./biofuel_production_CRP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# learning by doing
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_LBD0['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_LBD2['biofuel_production'].sum(axis=0)/10**9)
ax.legend(['No learning-by-doing','Low learning-by-doing','High learning-by-doing'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Biofuel Production (Billion L)',fontsize=18)
fig.savefig("./biofuel_production_LBD.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# neighborhood impact
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_NI1['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_PC1['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_NI3['biofuel_production'].sum(axis=0)/10**9)
ax.legend(['Low neighborhood impact','Median neighborhood impact','High neighborhood impact'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Biofuel Production (Billion L)',fontsize=18)
fig.savefig("./biofuel_production_NI.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# Willingness to pay
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_WP0['biofuel_production'].sum(axis=0)/10**9)
ax.plot(result_WP2['biofuel_production'].sum(axis=0)/10**9)
ax.legend(['Low willingness to pay','High willingness to pay'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Biofuel Production (Billion L)',fontsize=18)
fig.savefig("./biofuel_production_WP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


# plot the neighborhood impact
# all
plt.close('all')
fig, ax = plt.subplots(3,2,figsize=(6, 6))
ax[0][0].plot(result_SQ['neighbor_impact'])
ax[0][0].plot(result_SSB['neighbor_impact'])
ax[0][0].legend(['Status Quo','Small Scale \n Biofacility Subsudy'],fontsize=8)
ax[0][0].set_xticks([])
ax[0][0].set_ylim([0,2.5])
ax[0][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[0][1].plot(result_SSB['neighbor_impact'])
ax[0][1].plot(result_PC1['neighbor_impact'])
ax[0][1].legend(['No Policy Commitment','Policy Commitment'],fontsize=8)
ax[0][1].set_xticks([])
ax[0][1].set_ylim([0,2.5])
ax[0][1].set_yticklabels(['0','1','2'],fontsize=8)

ax[1][0].plot(result_PC1['neighbor_impact'])
ax[1][0].plot(result_CRP1['neighbor_impact'])
ax[1][0].legend(['No CRP relaxation','CRP relaxation'],fontsize=8)
ax[1][0].set_xticks([])
ax[1][0].set_ylim([0,2.5])
ax[1][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[1][1].plot(result_LBD0['neighbor_impact'])
ax[1][1].plot(result_PC1['neighbor_impact'])
ax[1][1].plot(result_LBD2['neighbor_impact'])
ax[1][1].legend(['No learning-by-doing','Low learning-by-doing','High learning-by-doing'],fontsize=8)
ax[1][1].set_xticks([])
ax[1][1].set_ylim([0,2.5])
ax[1][1].set_yticklabels(['0','1','2'],fontsize=8)

ax[2][0].plot(result_NI1['neighbor_impact'])
ax[2][0].plot(result_PC1['neighbor_impact'])
ax[2][0].plot(result_NI3['neighbor_impact'])
ax[2][0].legend(['Low neighborhood impact','Median neighborhood impact','High neighborhood impact'],fontsize=8)
# ax[2][0].set_xticks([0,10,20,30])
# ax[2][0].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][0].set_ylim([0,2.5])
ax[2][0].set_yticklabels(['0','1','2'],fontsize=8)

ax[2][1].plot(result_WP0['neighbor_impact'])
ax[2][1].plot(result_WP2['neighbor_impact'])
ax[2][1].legend(['Low willingness to pay','High willingness to pay'],fontsize=8)
# ax[2][1].set_xticks([0,10,20,30])
# ax[2][1].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][1].set_ylim([0,2.5])
ax[2][1].set_yticklabels(['0','1','2'],fontsize=8)

ax1 = fig.add_subplot(111, frame_on=False)
ax1.tick_params(labelcolor="none", bottom=False, left=False)
ax1.set_xlabel('Year',fontsize=14)
ax1.set_ylabel('Average Neighborhood Impact',fontsize=14)
fig.savefig("./neighbor_impact_all.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# commitment
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_SSB['neighbor_impact'])
ax.plot(result_PC1['neighbor_impact'])
ax.legend(['No Policy Commitment','Policy Commitment'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Average Neighborhood Impact',fontsize=18)
fig.savefig("./neighbor_impact_commitment.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# CRP
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_PC1['neighbor_impact'])
ax.plot(result_CRP1['neighbor_impact'])
ax.legend(['No CRP relaxation','CRP relaxation'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Average Neighborhood Impact',fontsize=18)
fig.savefig("./neighbor_impact_CRP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# learning by doing
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_LBD0['neighbor_impact'])
ax.plot(result_PC1['neighbor_impact'])
ax.plot(result_LBD2['neighbor_impact'])
ax.legend(['No learning-by-doing','Low learning-by-doing','High learning-by-doing'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Average Neighborhood Impact',fontsize=18)
fig.savefig("./neighbor_impact_LBD.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# neighborhood impact
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_NI1['neighbor_impact'])
ax.plot(result_PC1['neighbor_impact'])
ax.plot(result_NI3['neighbor_impact'])
ax.legend(['Low neighborhood impact','Median neighborhood impact','High neighborhood impact'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Average Neighborhood Impact',fontsize=18)
fig.savefig("./neighbor_impact_NI.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# Willingness to pay
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_WP0['neighbor_impact'])
ax.plot(result_WP2['neighbor_impact'])
ax.legend(['Low willingness to pay','High willingness to pay'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Average Neighborhood Impact',fontsize=18)
fig.savefig("./neighbor_impact_WP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)



# plot the N release
# all
plt.close('all')
fig, ax = plt.subplots(3,2,figsize=(6, 6))
ax[0][0].plot(result_SQ['N_release']/10**3)
ax[0][0].plot(result_SSB['N_release']/10**3)
ax[0][0].legend(['Status Quo','Small Scale \n Biofacility Subsudy'],fontsize=8)
ax[0][0].set_xticks([])
ax[0][0].set_ylim([110,160])
ax[0][0].set_yticklabels(['120','140','160'],fontsize=8)

ax[0][1].plot(result_SSB['N_release']/10**3)
ax[0][1].plot(result_PC1['N_release']/10**3)
ax[0][1].legend(['No Policy Commitment','Policy Commitment'],fontsize=8)
ax[0][1].set_xticks([])
ax[0][1].set_ylim([110,160])
ax[0][1].set_yticklabels(['120','140','160'],fontsize=8)

ax[1][0].plot(result_PC1['N_release']/10**3)
ax[1][0].plot(result_CRP1['N_release']/10**3)
ax[1][0].legend(['No CRP relaxation','CRP relaxation'],fontsize=8)
ax[1][0].set_xticks([])
ax[1][0].set_ylim([110,160])
ax[1][0].set_yticklabels(['120','140','160'],fontsize=8)

ax[1][1].plot(result_LBD0['N_release']/10**3)
ax[1][1].plot(result_PC1['N_release']/10**3)
ax[1][1].plot(result_LBD2['N_release']/10**3)
ax[1][1].legend(['No learning-by-doing','Low learning-by-doing','High learning-by-doing'],fontsize=8)
ax[1][1].set_xticks([])
ax[1][1].set_ylim([110,160])
ax[1][1].set_yticklabels(['120','140','160'],fontsize=8)

ax[2][0].plot(result_NI1['N_release']/10**3)
ax[2][0].plot(result_PC1['N_release']/10**3)
ax[2][0].plot(result_NI3['N_release']/10**3)
ax[2][0].legend(['Low neighborhood impact','Median neighborhood impact','High neighborhood impact'],fontsize=8)
# ax[2][0].set_xticks([0,10,20,30])
# ax[2][0].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][0].set_ylim([110,160])
ax[2][0].set_yticklabels(['120','140','160'],fontsize=8)

ax[2][1].plot(result_WP0['N_release']/10**3)
ax[2][1].plot(result_WP2['N_release']/10**3)
ax[2][1].legend(['Low willingness to pay','High willingness to pay'],fontsize=8)
# ax[2][1].set_xticks([0,10,20,30])
# ax[2][1].set_xticklabels(['0','10','20','30'],fontsize=8)
ax[2][1].set_ylim([110,160])
ax[2][1].set_yticklabels(['120','140','160'],fontsize=8)

ax1 = fig.add_subplot(111, frame_on=False)
ax1.tick_params(labelcolor="none", bottom=False, left=False)
ax1.set_xlabel('Year',fontsize=14)
ax1.set_ylabel(')Total N Release (ton)',fontsize=14)
fig.savefig("./N_release_all.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# commitment
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_SSB['N_release']/10**3)
ax.plot(result_PC1['N_release']/10**3)
ax.legend(['No Policy Commitment','Policy Commitment'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Total N Release (ton)',fontsize=18)
fig.savefig("./N_release_commitment.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# CRP
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_PC1['N_release']/10**3)
ax.plot(result_CRP1['N_release']/10**3)
ax.legend(['No CRP relaxation','CRP relaxation'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Total N Release (ton)',fontsize=18)
fig.savefig("./N_release_CRP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# learning by doing
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_LBD0['N_release']/10**3)
ax.plot(result_PC1['N_release']/10**3)
ax.plot(result_LBD2['N_release']/10**3)
ax.legend(['No learning-by-doing','Low learning-by-doing','High learning-by-doing'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Total N Release (ton)',fontsize=18)
fig.savefig("./N_release_LBD.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# neighborhood impact
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_NI1['N_release']/10**3)
ax.plot(result_PC1['N_release']/10**3)
ax.plot(result_NI3['N_release']/10**3)
ax.legend(['Low neighborhood impact','Median neighborhood impact','High neighborhood impact'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Total N Release (ton)',fontsize=18)
fig.savefig("./N_release_NI.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# Willingness to pay
plt.close('all')
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(result_WP0['N_release']/10**3)
ax.plot(result_WP2['N_release']/10**3)
ax.legend(['Low willingness to pay','High willingness to pay'])
ax.set_xlabel('Year',fontsize=18)
ax.set_ylabel('Total N Release (ton)',fontsize=18)
fig.savefig("./N_release_WP.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


# plot stakeholder-stakeholder interaction
RFS_mandate = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/RFS_volume.csv')
prices_rfs = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/external_provided_prices.csv')
cell_eth_price_rfs = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/endogenous_ethanol_price.csv')
cell_eth_price_baseline = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/data/endogenous_ethanol_price.csv')
data_WP2 = shelve.open(dir_LBD0)
fig, ax = plt.subplots(3,1,figsize=(6, 6))
ax[0].plot(result_CRP1['price_adj_hist'][:,0]/3.78,'b')
ax[0].plot(cell_eth_price_baseline['endogenous_ethanol ($/L)'][:21],'r--')
ax[0].set_ylabel('Ethanol Price ($/L)')
ax[0].legend(['Cellulosic Premium','Corn Ethanol'])
ax[0].set_xticks([])

ax[1].plot(result_CRP1['biofuel_production'][result_CRP1['biofuel_type']==1].sum(0)/10**9)
ax[1].plot(result_CRP1['biofuel_production'][result_CRP1['biofuel_type']==2].sum(0)/10**9)
ax[1].plot(RFS_mandate['RFS_volume (Million L)'][:20]/1000,'r--')
ax[1].legend(['Corn Ethanol','Cellulosic Ethanol','RFS Cellulosic Mandate'])
ax[1].set_ylabel('Ethanol Production \n (Billion L)')
ax[1].set_xticks([])

ax[2].plot(cal_stats.cal_min_IRR_RFS_policy_commitment(data_WP2['ABM_parameters']['RFS_signal'])[:20])
# ax[2].plot(result_WP2['ref_inv_cost_adj_his'][:,1])
ax[2].set_ylabel('Refinery Minimum IRR')
ax[2].set_xticks([0,5,10,15,20])
ax[2].set_xlabel('Year')

plt.tight_layout(pad=2, w_pad=2, h_pad=0.2)
fig.savefig("./ABM_gov_ref_commu_interaction.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# plot perennial adoption & biomass price
corn_eth_produce = result_SSB_PC1['biofuel_production'][result_SSB_PC1['biofuel_type']==1]
corn_eth_produce = corn_eth_produce.sum(0)
cell_eth_produce = result_SSB_PC1['biofuel_production'][result_SSB_PC1['biofuel_type']==2]
cell_eth_produce = cell_eth_produce.sum(0)
RFS_mandate = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/RFS/data/RFS_volume.csv')
data_SSB_PC1 = shelve.open(dir_SSB_PC1)

fig, ax = plt.subplots(2,2,figsize=(12, 6))
lns1=ax[0][0].plot(prices_rfs['ethanol ($/L)'][:20],label='Corn Ethanol',linewidth=2)
lns2=ax[0][0].plot(cell_eth_price_rfs['endogenous_ethanol ($/L)'][:20],label='Cellulosic Ethanol',linewidth=2)
ax[0][0].set_ylabel('Ethanol Price ($/L)')
ax1 = ax[0][0].twinx()
lns3=ax1.plot(prices_rfs['mis ($/ton)'][:20],color='r',label='Biomass Price',linewidth=2)
ax1.set_xticks([])
ax1.tick_params(axis='y')
ax1.set_ylabel('Biomass Price ($/t)')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax[0][0].legend(lns, labs, loc="lower right")

color = 'tab:blue'
ax[0][1].plot(result_SSB_PC1['att_farmer'],color=color,linewidth=2)
ax[0][1].set_ylabel("Farmer's willingness \n to adopt Miscanthus")
ax[0][1].tick_params(axis='y', labelcolor=color)
ax2 = ax[0][1].twinx()
color = 'tab:red'
ax2.plot(result_SSB_PC1['environ_sensi_farmer'],color=color,linewidth=2)
ax2.set_ylabel("Farmer's attitude \n  toward environment")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xticks([])

ax[1][0].plot(corn_eth_produce/10**9,linewidth=2)
ax[1][0].plot(cell_eth_produce/10**9,linewidth=2)
ax[1][0].plot(RFS_mandate['RFS_volume (Million L)'][:20]/1000,'r--',linewidth=2)
ax[1][0].legend(['Corn Ethanol','Cellulosic Ethanol','RFS Cellulosic Mandate'],loc='lower right')
ax[1][0].set_ylabel('Ethanol Production (Billion L)')
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][0].set_xlabel('Year')

ax[1][1].plot(cal_stats.cal_min_IRR_RFS_policy_commitment(data_SSB_PC1['ABM_parameters']['RFS_signal'])[:20],linewidth=2)
ax[1][1].set_ylabel('Refinery Minimum IRR')
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlabel('Year')

plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
fig.savefig("./system_dynamics.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


# plot stakeholder-environment interaction
# marginal land
import parameters as params
marginal_land = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/data/marginal_land.csv')
data_SSB_PC1 = shelve.open(dir_SSB_PC1)
farmer_with_ML = (marginal_land['lql']==1)
farmer_without_ML = (marginal_land['lql']==0)
adopt_farmer_ML = np.zeros((params.simu_horizon, 2))
att_farmer_ML = np.zeros((params.simu_horizon, 2))
farmer_agent_list = data_SSB_PC1['farmer_agent_list']
for i in range(params.simu_horizon):
    for j in range(2):
        temp = []
        for k in range(farmer_agent_list.__len__()):
            if marginal_land['lql'][k]==j:
                temp.append(farmer_agent_list[k].States['SC_Will'][i])
        att_farmer_ML[i, j] = np.asarray(temp).mean()

for i in range(params.simu_horizon):
    for j in range(2):
        temp1 = 0
        temp2 = 0
        for k in range(farmer_agent_list.__len__()):
            if marginal_land['lql'][k]==j:
                adoption = np.isin(farmer_agent_list[k].States['land_use'][i], [3, 4]).sum() > 0 + 0
                temp1 += adoption
                temp2 += 1
        adopt_farmer_ML[i, j] = temp1 / temp2

fig, ax = plt.subplots(2,2,figsize=(8, 6))

ax[0][0].set_prop_cycle(color=['b','r','g','k'],linestyle=['solid','dashed','dotted','dashdot'])
temp_data = np.asarray([result_SSB_PC1['att_farmer_cluster'][:,1],result_SSB_PC1['att_farmer_cluster'][:,0],
                        result_SSB_PC1['att_farmer_cluster'][:,3],result_SSB_PC1['att_farmer_cluster'][:,2]]).T
ax[0][0].plot(temp_data,linewidth=2)
# ax[0][0].set_xlabel('Year')
ax[0][0].set_xticks([])
ax[0][0].set_ylabel("Willingness to adopt Miscanthus")
ax[0][0].legend(['Type I','Type II','Type III','Type IV'],loc='upper left')

ax[1][0].set_prop_cycle(color=['b','r','g','k'],linestyle=['solid','dashed','dotted','dashdot'])
temp_data = np.asarray([result_SSB_PC1['adopt_farmer_cluster'][:,1],result_SSB_PC1['adopt_farmer_cluster'][:,0],
                        result_SSB_PC1['adopt_farmer_cluster'][:,3],result_SSB_PC1['adopt_farmer_cluster'][:,2]]).T
ax[1][0].plot(temp_data,linewidth=2)
ax[1][0].set_xticks([0,5,10,15,20])
ax[1][0].set_xlabel('Year')
ax[1][0].set_ylabel('Adoption Ratio')
ax[1][0].legend(['Type I','Type II','Type III','Type IV'],loc='upper left')

ax[0][1].set_prop_cycle(color=['r','b'],linestyle=['solid','dashed'])
ax[0][1].plot(att_farmer_ML,linewidth=2)
ax[0][1].legend(['No marginal land','Marginal land'],loc='upper left')
ax[0][1].set_xticks([])
ax[0][1].set_yticks([])
# ax[0][1].set_ylabel('Willingness to adopt Miscanthus')

ax[1][1].set_prop_cycle(color=['r','b'],linestyle=['solid','dashed'])
ax[1][1].plot(adopt_farmer_ML,linewidth=2)
ax[1][1].legend(['No marginal land','Marginal land'],loc='upper left')
ax[1][1].set_xticks([0,5,10,15,20])
ax[1][1].set_xlabel('Year')
ax[1][1].set_yticks([])
# ax[1][1].set_ylabel('Adoption Ratio')
plt.tight_layout(pad=2, w_pad=0.5, h_pad=0.5)
fig.savefig("./farmer_heteorogenity.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)


# Neighborhood impact – the change of neighborhood impact and willingness to adopt
fig, ax = plt.subplots(2,1,figsize=(6, 6))
ax[0].plot(result_NI1['neighbor_impact'])
ax[0].set_ylabel("Average Neighborhood Impact")
ax1 = ax[0].twinx()
color = 'tab:red'
ax1.plot(result_NI1['att_farmer'],color=color)
ax1.set_ylabel("Farmer's Willingness to \n  Grow Miscanthus")
ax1.tick_params(axis='y', labelcolor=color)

ax[1].plot(result_PC1['neighbor_impact'])
ax[1].set_ylabel("Average Neighborhood Impact")
ax[1].set_xlabel('Year')
ax2 = ax[1].twinx()
color = 'tab:red'
ax2.plot(result_PC1['att_farmer'],color=color)
ax2.set_ylabel("Farmer's Willingness to \n  Grow Miscanthus")
ax2.tick_params(axis='y', labelcolor=color)
plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
fig.savefig("./Neighor_Willingness.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# Policy commitment – the change of minimum IRR and biofuel production
fig, ax = plt.subplots(2,1,figsize=(6, 6))
data_SSB = shelve.open(dir_SSB)
data_PC1 = shelve.open(dir_PC1)
ax[0].plot(cal_stats.cal_min_IRR_RFS_policy_commitment(data_SSB['ABM_parameters']['RFS_signal'])[:20])
ax[0].plot(cal_stats.cal_min_IRR_RFS_policy_commitment(data_PC1['ABM_parameters']['RFS_signal'])[:20])
ax[0].set_ylabel("Minimum IRR")
ax[0].legend(['No policy commitment','Policy commitment'])
ax[0].set_xticks([])

ax[1].plot(result_SSB['biofuel_production'].sum(0)/10**9)
ax[1].plot(result_PC1['biofuel_production'].sum(0)/10**9)
ax[1].set_ylabel("Biofuel Production (Billion L)")
ax[1].set_xlabel('Year')
ax[1].legend(['No policy commitment','Policy commitment'])
plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
fig.savefig("./IRR_biofuel_production.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)

# new technology
fig, ax = plt.subplots(3,1,figsize=(6, 6))
data_SSB = shelve.open(dir_SSB)
data_PC1 = shelve.open(dir_PC1)
ax[0].plot(result_LBD0['ref_inv_cost_adj_his'][:,1])
ax[0].plot(result_newtech['ref_inv_cost_adj_his'][:,1])
ax[0].set_ylabel("Relative Production Cost")
ax[0].legend(['No new technology','New technology'])
ax[0].set_xticks([])

ax[1].plot(result_LBD0['land_use_areas'][:,2]/10**6)
ax[1].plot(result_newtech['land_use_areas'][:,2]/10**6)
ax[1].set_ylabel("Miscanthus Area (Million ha)")
ax[1].legend(['No new technology','New technology'])
ax[1].set_xticks([])

ax[2].plot(result_LBD0['biofuel_production'].sum(0)/10**9)
ax[2].plot(result_newtech['biofuel_production'].sum(0)/10**9)
ax[2].set_ylabel("Biofuel Production (Billion L)")
ax[2].set_xlabel('Year')
ax[2].legend(['No new technology','New technology'])
plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
fig.savefig("./New_technology_comparison.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)



patch_area = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline/data/patch_attributes.csv')
patch_area = patch_area['area'].to_numpy()
patch_area = np.reshape(patch_area,(33965,1))
patch_area = np.repeat(patch_area,20,1)
gov_agent = shelve.open(dir_TMDL1)['gov_agent']
fig, ax = plt.subplots(3,1,figsize=(6, 6))
ax[0].plot(result_TMDL1['N_release']/1000,'b',linewidth=2)
ax[0].plot(result_SQ['N_release']/1000,'r',linewidth=2,linestyle='dotted')
ax[0].plot(gov_agent.Attributes['TMDL'][0:20],'k--',linewidth=2)
ax[0].set_ylabel('N Release (ton)')
ax[0].legend(['ABM simulated','Status quo','N cap'], loc="upper right")
ax[0].set_xticks([])

lns1 = ax[1].plot((100*result_TMDL1['TMDL_eligibilities']*patch_area).sum(0)/patch_area.sum(0)[0],'b',label='Land Eligible for TMDL',linewidth=2)
ax[1].set_ylabel("Land Eligible for TMDL (%)")
ax1 = ax[1].twinx()
lns2 = ax1.plot(result_TMDL1['land_use_areas'][:20,2]/10**3,'g',label='Miscanthus Area',linewidth=2,linestyle='dashed')
ax1.tick_params(axis='y')
ax1.set_xticks([])
ax1.set_ylabel("Miscanthus Area (thousand ha")
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax[1].legend(lns, labs, loc="lower right")

ax[2].plot(result_TMDL1['biofuel_production'][result_TMDL1['biofuel_type']==2].sum(0)/10**9,linewidth=2)
ax[2].set_ylabel('Cellulosic Biofuel \nProduction (Billion L)')
ax[2].set_xticks([0,5,10,15,20])
ax[2].set_xlabel('Year')

plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.2)
fig.savefig("./ABM_environment_interaction.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)




#
# # plot the total miscanthus land use
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_baseline['land_use_areas'][30,2],result_V4_PB50['land_use_areas'][30,2],result_V4_RFS['land_use_areas'][30,2]])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['$40/t','$50/t','RFS'],fontsize=14)
# ax.set_xlabel('Biomass Price',fontsize=18)
# ax.set_ylabel('Area (ha)',fontsize=18)
# fig.savefig("./mis_area_price.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['land_use_areas'][30,2],result_V4_CRP['land_use_areas'][30,2],result_V4_BCAP['land_use_areas'][30,2],result_V4_BCAP_CRP['land_use_areas'][30,2]])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['RFS','CRP','BCAP','BCAP_CRP'],fontsize=14)
# ax.set_xlabel('Policy',fontsize=18)
# ax.set_ylabel('Area (ha)',fontsize=18)
# fig.savefig("./mis_area_policy.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['land_use_areas'][30,2],result_V4_tech_90['land_use_areas'][30,2],result_V4_tech_80['land_use_areas'][30,2],result_V4_tech_70['land_use_areas'][30,2]])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['T100','T90','T80','T70'],fontsize=14)
# ax.set_xlabel('Technology',fontsize=18)
# ax.set_ylabel('Area (ha)',fontsize=18)
# fig.savefig("./mis_area_tech.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_behavior50['land_use_areas'][30,2],result_V4_behavior75['land_use_areas'][30,2],
#          result_V4_RFS['land_use_areas'][30,2],result_V4_behavior125['land_use_areas'][30,2],result_V4_behavior150['land_use_areas'][30,2]])
# ax.set_xticks([0,1,2,3,4])
# ax.set_xticklabels(['B50','B75','B100','B125','B150'],fontsize=14)
# ax.set_xlabel('Behavior',fontsize=18)
# ax.set_ylabel('Area (ha)',fontsize=18)
# fig.savefig("./mis_area_behavior.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V2_baseline['land_use_areas'][30,2],result_V2_PB50['land_use_areas'][30,2],result_V2_RFS['land_use_areas'][30,2]])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['$40/t','$50/t','RFS'],fontsize=14)
# ax.set_xlabel('Biomass Price',fontsize=18)
# ax.set_ylabel('Area (ha)',fontsize=18)
# fig.savefig("./v2_mis_area_price.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# # plot the adoption and contract percentage
# plt.close('all')
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_baseline['farmer_adoption'].sum(0)[30]/10,result_V4_PB50['farmer_adoption'].sum(0)[30]/10,result_V4_RFS['farmer_adoption'].sum(0)[30]/10])
# ax.plot([result_V4_baseline['contract_farm'].sum(0)[30]/10,result_V4_PB50['contract_farm'].sum(0)[30]/10,result_V4_RFS['contract_farm'].sum(0)[30]/10])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['$40/t','$50/t','RFS'],fontsize=14)
# ax.set_xlabel('Biomass Price',fontsize=18)
# ax.set_ylabel("Percent",fontsize=18)
# ax.legend(['Perennial Grass Adoption','Farmer Accepting Contract'],fontsize=14)
# fig.savefig("./farmer_adoption_his_price.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['farmer_adoption'].sum(0)[30]/10,result_V4_CRP['farmer_adoption'].sum(0)[30]/10,result_V4_BCAP['farmer_adoption'].sum(0)[30]/10,result_V4_BCAP_CRP['farmer_adoption'].sum(0)[30]/10])
# ax.plot([result_V4_RFS['contract_farm'].sum(0)[30]/10,result_V4_CRP['contract_farm'].sum(0)[30]/10,result_V4_BCAP['contract_farm'].sum(0)[30]/10,result_V4_BCAP_CRP['contract_farm'].sum(0)[30]/10])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['RFS','CRP','BCAP','BCAP_CRP'],fontsize=14)
# ax.set_xlabel('Policy',fontsize=18)
# ax.set_ylabel("Percent",fontsize=18)
# ax.legend(['Perennial Grass Adoption','Farmer Accepting Contract'],fontsize=14)
# fig.savefig("./farmer_adoption_his_policy.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['farmer_adoption'].sum(0)[30]/10,result_V4_tech_90['farmer_adoption'].sum(0)[30]/10,result_V4_tech_80['farmer_adoption'].sum(0)[30]/10,result_V4_tech_70['farmer_adoption'].sum(0)[30]/10])
# ax.plot([result_V4_RFS['contract_farm'].sum(0)[30]/10,result_V4_tech_90['contract_farm'].sum(0)[30]/10,result_V4_tech_80['contract_farm'].sum(0)[30]/10,result_V4_tech_70['contract_farm'].sum(0)[30]/10])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['T100','T90','T80','T70'],fontsize=14)
# ax.set_xlabel('Technology',fontsize=18)
# ax.set_ylabel("Percent",fontsize=18)
# ax.legend(['Perennial Grass Adoption','Farmer Accepting Contract'],fontsize=14)
# fig.savefig("./farmer_adoption_his_tech.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_behavior50['farmer_adoption'].sum(0)[30]/10,result_V4_behavior75['farmer_adoption'].sum(0)[30]/10,
#          result_V4_RFS['farmer_adoption'].sum(0)[30]/10,result_V4_behavior125['farmer_adoption'].sum(0)[30]/10,
#          result_V4_behavior150['farmer_adoption'].sum(0)[30]/10])
# ax.plot([result_V4_behavior50['contract_farm'].sum(0)[30]/10,result_V4_behavior75['contract_farm'].sum(0)[30]/10,
#          result_V4_RFS['contract_farm'].sum(0)[30]/10,result_V4_behavior125['contract_farm'].sum(0)[30]/10,
#          result_V4_behavior150['contract_farm'].sum(0)[30]/10])
# ax.set_xticks([0,1,2,3,4])
# ax.set_xticklabels(['B50','B75','B100','B125','B150'],fontsize=14)
# ax.set_xlabel('Behavior',fontsize=18)
# ax.set_ylabel("Percent",fontsize=18)
# ax.legend(['Perennial Grass Adoption','Farmer Accepting Contract'],fontsize=14)
# fig.savefig("./farmer_adoption_his_behavior.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# # plot the biofuel production
# plt.close('all')
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_baseline['biofuel_production'].sum(0)[-1]/10**9,result_V4_PB50['biofuel_production'].sum(0)[-1]/10**9,
#          result_V4_RFS['biofuel_production'].sum(0)[-1]/10**9])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['$40/t','$50/t','RFS'],fontsize=14)
# ax.set_xlabel('Biomass Price',fontsize=18)
# ax.set_ylabel("Biofuel Production (Billion L)",fontsize=18)
# fig.savefig("./biofuel_production_price.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['biofuel_production'].sum(0)[-1]/10**9,result_V4_CRP['biofuel_production'].sum(0)[-1]/10**9,
#          result_V4_BCAP['biofuel_production'].sum(0)[-1]/10**9,result_V4_BCAP_CRP['biofuel_production'].sum(0)[-1]/10**9])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['RFS','CRP','BCAP','BCAP_CRP'],fontsize=14)
# ax.set_xlabel('Policy',fontsize=18)
# ax.set_ylabel("Biofuel Production (Billion L)",fontsize=18)
# fig.savefig("./biofuel_production_policy.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['biofuel_production'].sum(0)[-1]/10**9,result_V4_tech_90['biofuel_production'].sum(0)[-1]/10**9,
#          result_V4_tech_80['biofuel_production'].sum(0)[-1]/10**9,result_V4_tech_70['biofuel_production'].sum(0)[-1]/10**9])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['T100','T90','T80','T70'],fontsize=14)
# ax.set_xlabel('Technology',fontsize=18)
# ax.set_ylabel("Biofuel Production (Billion L)",fontsize=18)
# fig.savefig("./biofuel_production_tech.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_behavior50['biofuel_production'].sum(0)[-1]/10**9,result_V4_behavior75['biofuel_production'].sum(0)[-1]/10**9,
#          result_V4_RFS['biofuel_production'].sum(0)[-1]/10**9,result_V4_behavior125['biofuel_production'].sum(0)[-1]/10**9,
#          result_V4_behavior150['biofuel_production'].sum(0)[-1]/10**9])
# ax.set_xticks([0,1,2,3,4])
# ax.set_xticklabels(['B50','B75','B100','B125','B150'],fontsize=14)
# ax.set_xlabel('Behavior',fontsize=18)
# ax.set_ylabel("Biofuel Production (Billion L)",fontsize=18)
# fig.savefig("./biofuel_production_behavior.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# # plot the N loading
# plt.close('all')
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_baseline['N_release'].mean()/10**3,result_V4_PB50['N_release'].mean()/10**3,
#          result_V4_RFS['N_release'].mean()/10**3])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['$40/t','$50/t','RFS'],fontsize=14)
# ax.set_xlabel('Biomass Price',fontsize=18)
# ax.set_ylabel("N Release (Ton)",fontsize=18)
# fig.savefig("./N_release_price.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['N_release'].mean()/10**3,result_V4_CRP['N_release'].mean()/10**3,
#          result_V4_BCAP['N_release'].mean()/10**3,result_V4_BCAP_CRP['N_release'].mean()/10**3])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['RFS','CRP','BCAP','BCAP_CRP'],fontsize=14)
# ax.set_xlabel('Policy',fontsize=18)
# ax.set_ylabel("N Release (Ton)",fontsize=18)
# fig.savefig("./N_release_policy.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_RFS['N_release'].mean()/10**3,result_V4_tech_90['N_release'].mean()/10**3,
#          result_V4_tech_80['N_release'].mean()/10**3,result_V4_tech_70['N_release'].mean()/10**3])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['T100','T90','T80','T70'],fontsize=14)
# ax.set_xlabel('Technology',fontsize=18)
# ax.set_ylabel("N Release (Ton)",fontsize=18)
# fig.savefig("./N_release_tech.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([result_V4_behavior50['N_release'].mean()/10**3,result_V4_behavior75['N_release'].mean()/10**3,
#          result_V4_RFS['N_release'].mean()/10**3,result_V4_behavior125['N_release'].mean()/10**3,
#          result_V4_behavior150['N_release'].mean()/10**3])
# ax.set_xticks([0,1,2,3,4])
# ax.set_xticklabels(['B50','B75','B100','B125','B150'],fontsize=14)
# ax.set_xlabel('Behavior',fontsize=18)
# ax.set_ylabel("N Release (Ton)",fontsize=18)
# fig.savefig("./N_release_behavior.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# # system level profit
# plt.close('all')
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([(result_V4_baseline['revenue_farmer'].sum()+result_V4_baseline['ref_profit'].sum())/10**9,
#          (result_V4_PB50['revenue_farmer'].sum()+result_V4_PB50['ref_profit'].sum())/10**9,
#          (result_V4_RFS['revenue_farmer'].sum()+result_V4_RFS['ref_profit'].sum())/10**9])
# ax.set_xticks([0,1,2])
# ax.set_xticklabels(['$40/t','$50/t','RFS'],fontsize=14)
# ax.set_xlabel('Biomass Price',fontsize=18)
# ax.set_ylabel("Total Profit (Billion $)",fontsize=18)
# fig.savefig("./total_profit_price.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([(result_V4_RFS['revenue_farmer'].sum()+result_V4_RFS['ref_profit'].sum())/10**9,
#          (result_V4_CRP['revenue_farmer'].sum()+result_V4_CRP['ref_profit'].sum())/10**9,
#          (result_V4_BCAP['revenue_farmer'].sum()+result_V4_BCAP['ref_profit'].sum())/10**9,
#          (result_V4_BCAP_CRP['revenue_farmer'].sum()+result_V4_BCAP_CRP['ref_profit'].sum())/10**9])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['RFS','CRP','BCAP','BCAP_CRP'],fontsize=14)
# ax.set_xlabel('Policy',fontsize=18)
# ax.set_ylabel("Total Profit (Billion $)",fontsize=18)
# fig.savefig("./total_profit_policy.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([(result_V4_RFS['revenue_farmer'].sum()+result_V4_RFS['ref_profit'].sum())/10**9,
#          (result_V4_tech_90['revenue_farmer'].sum()+result_V4_tech_90['ref_profit'].sum())/10**9,
#          (result_V4_tech_80['revenue_farmer'].sum()+result_V4_tech_80['ref_profit'].sum())/10**9,
#          (result_V4_tech_70['revenue_farmer'].sum()+result_V4_tech_70['ref_profit'].sum())/10**9])
# ax.set_xticks([0,1,2,3])
# ax.set_xticklabels(['T100','T90','T80','T70'],fontsize=14)
# ax.set_xlabel('Technology',fontsize=18)
# ax.set_ylabel("Total Profit (Billion $)",fontsize=18)
# fig.savefig("./total_profit_tech.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.plot([(result_V4_behavior50['revenue_farmer'].sum()+result_V4_behavior50['ref_profit'].sum())/10**9,
#          (result_V4_behavior75['revenue_farmer'].sum()+result_V4_behavior75['ref_profit'].sum())/10**9,
#          (result_V4_RFS['revenue_farmer'].sum()+result_V4_RFS['ref_profit'].sum())/10**9,
#          (result_V4_behavior125['revenue_farmer'].sum()+result_V4_behavior125['ref_profit'].sum())/10**9,
#          (result_V4_behavior150['revenue_farmer'].sum()+result_V4_behavior150['ref_profit'].sum())/10**9])
# ax.set_xticks([0,1,2,3,4])
# ax.set_xticklabels(['B50','B75','B100','B125','B150'],fontsize=14)
# ax.set_xlabel('Behavior',fontsize=18)
# ax.set_ylabel("Total Profit (Billion $)",fontsize=18)
# fig.savefig("./total_profit_behavior.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# '''
# time series
# '''
# # perennial adoption & biomass price
# prices_rfs = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Version_4/simulations/RFS/data/external_provided_prices.csv')
# cell_eth_price_rfs = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Version_4/simulations/RFS/data/endogenous_ethanol_price.csv')
# corn_eth_produce = result_V4_RFS['biofuel_production'][result_V4_RFS['biofuel_type']==1]
# corn_eth_produce = corn_eth_produce.sum(0)
# cell_eth_produce = result_V4_RFS['biofuel_production'][result_V4_RFS['biofuel_type']==2]
# cell_eth_produce = cell_eth_produce.sum(0)
#
# fig, ax = plt.subplots(2,2,figsize=(12, 6))
# lns1=ax[0][0].plot(prices_rfs['ethanol ($/L)'],label='Corn Ethanol')
# lns2=ax[0][0].plot(cell_eth_price_rfs['endogenous_ethanol ($/L)'],label='Cellulosic Ethanol')
# ax[0][0].set_ylabel('Ethanol Price ($/L)')
# ax1 = ax[0][0].twinx()
# lns3=ax1.plot(prices_rfs['mis ($/ton)'],color='r',label='Biomass Price')
# ax1.set_xticks([])
# ax1.tick_params(axis='y')
# ax1.set_ylabel('Biomass Price ($/t)')
# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]
# ax[0][0].legend(lns, labs, loc="lower right")
#
# color = 'tab:blue'
# ax[0][1].plot(result_V4_RFS['att_farmer'],color=color)
# ax[0][1].set_ylabel("Farmer's attitude \n toward perennial grass")
# ax[0][1].tick_params(axis='y', labelcolor=color)
# ax2 = ax[0][1].twinx()
# color = 'tab:red'
# ax2.plot(result_V4_RFS['environ_sensi_farmer'],color=color)
# ax2.set_ylabel("Farmer's environmental \n  sensitivity")
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.set_xticks([])
#
# ax[1][0].plot(corn_eth_produce/10**9)
# ax[1][0].plot(cell_eth_produce/10**9)
# ax[1][0].legend(['Corn Ethanol','Cellulosic Ethanol'])
# ax[1][0].set_ylabel('Ethanol Production (Billion L)')
# ax[1][0].set_xlabel('Year')
#
# ax[1][1].plot(result_V4_RFS['land_use_areas'][:, [0, 1, 2, 6, 7]] / 10 ** 6)
# ax[1][1].legend(['contcorn', 'cornsoy', 'miscanthus', 'fallow', 'CRP'])
# ax[1][1].set_ylabel('Area (Million ha)')
# ax[1][1].set_xlabel('Year')
#
# plt.tight_layout(pad=2, w_pad=2, h_pad=0.5)
# fig.savefig("./system_dynamics.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
# # time series of different scenarios
# fig, ax = plt.subplots(4,1,figsize=(9, 9))
#
# ax[0].plot(result_V4_baseline['land_use_areas'][:,2]/10**6)
# ax[0].plot(result_V4_PB50['land_use_areas'][:,2]/10**6)
# ax[0].plot(result_V4_RFS['land_use_areas'][:,2]/10**6)
# ax[0].legend(['PB40','PB50','RFS'])
# ax[0].set_ylabel('Area (Million ha)')
# ax[0].set_xticks([])
#
# ax[1].plot(result_V4_RFS['land_use_areas'][:,2]/10**6)
# ax[1].plot(result_V4_CRP['land_use_areas'][:,2]/10**6)
# ax[1].plot(result_V4_BCAP['land_use_areas'][:,2]/10**6)
# ax[1].plot(result_V4_BCAP_CRP['land_use_areas'][:,2]/10**6)
# ax[1].legend(['RFS','RFS + CRP','RFS + BCAP','RFS + CRP + BCAP'])
# ax[1].set_ylabel('Area (Million ha)')
# ax[1].set_xticks([])
#
# ax[2].plot(result_V4_tech_70['land_use_areas'][:,2]/10**6)
# ax[2].plot(result_V4_tech_80['land_use_areas'][:,2]/10**6)
# ax[2].plot(result_V4_tech_90['land_use_areas'][:,2]/10**6)
# ax[2].plot(result_V4_RFS['land_use_areas'][:,2]/10**6)
# ax[2].legend(['T70','T80','T90','T100'])
# ax[2].set_ylabel('Area (Million ha)')
# ax[2].set_xticks([])
#
# ax[3].plot(result_V4_behavior50['land_use_areas'][:,2]/10**6)
# ax[3].plot(result_V4_behavior75['land_use_areas'][:,2]/10**6)
# ax[3].plot(result_V4_RFS['land_use_areas'][:,2]/10**6)
# ax[3].plot(result_V4_behavior125['land_use_areas'][:,2]/10**6)
# ax[3].plot(result_V4_behavior150['land_use_areas'][:,2]/10**6)
# ax[3].legend(['B50','B75','B100','B125','B150'])
# ax[3].set_ylabel('Area (Million ha)')
# ax[3].set_xlabel('Year')
#
# plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
# fig.savefig("./Mis_area_scenarios.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
#
#
#
#
# patch_area = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Version_4/simulations/baseline/data/patch_attributes.csv')
# patch_area = patch_area['area'].to_numpy()
# patch_area = np.reshape(patch_area,(33965,1))
# patch_area = np.repeat(patch_area,30,1)
# fig, ax = plt.subplots(2,1,figsize=(6, 6))
# ax[0].plot(result_TMDL['N_release']/1000,'b')
# ax[0].plot(result_baseline['N_release']/1000,'r')
# ax[0].plot(gov_agent.Attributes['TMDL'],'k--')
# ax[0].set_ylabel('N Release (ton)')
# ax[0].legend(['ABM simulated','Status quo','N cap'], loc="upper right")
# ax[0].set_xticks([])
#
# lns1 = ax[1].plot((100*result_TMDL['TMDL_eligibilities']*patch_area).sum(0)/patch_area.sum(0)[0],'b',label='Land Eligible for TMDL')
# ax[1].set_ylabel("Land Eligible for TMDL (%)")
# ax1 = ax[1].twinx()
# lns2 = ax1.plot(result_TMDL['land_use_areas'][:,2]/10**3,'g',label='Miscanthus Area')
# ax1.tick_params(axis='y')
# ax1.set_xticks([])
# ax1.set_ylabel("Miscanthus Area (thousand ha")
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax[1].legend(lns, labs, loc="lower right")
#
# plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.2)
# fig.savefig("./ABM_environment_interaction.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
#
#
#
#
#
#
#
#
#
#
#
#
# cell_eth_price_baseline = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Version_4/simulations/baseline/data/endogenous_ethanol_price.csv')
# fig, ax = plt.subplots(3,1,figsize=(6, 6))
# ax[0].plot(result_V4_nopolicy['N_release']/1000,'b')
# ax[0].plot(result_V2_baseline['N_release']/1000,'r--')
# ax[0].set_ylabel('N Release (ton)')
# ax[0].legend(['ABM simulated','Status quo'], loc="lower right")
# ax[0].set_xticks([])
#
# lns1 = ax[1].plot(result_V4_nopolicy['commu_atti'].mean(0),'b',label='Community Sensivity to Environment')
# lns2 = ax[1].plot(result_V4_nopolicy['environ_sensi_farmer'],'r',label='Farmer Sensivity to Environment')
# ax[1].set_ylabel("Sensitivity to \n Environment")
# ax1 = ax[1].twinx()
# lns3 = ax1.plot(result_V4_nopolicy['att_farmer'],'g',label='Farmer Attitude to Perennial Grass')
# ax1.tick_params(axis='y')
# ax1.set_xticks([])
# ax1.set_ylabel("Farmer's attitude \n toward perennial grass")
# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]
# ax[1].legend(lns, labs, loc="lower right")
#
# lns1 = ax[2].plot(result_V4_nopolicy['farmer_adoption'].mean(0)[:30],'b',label='Farmer Adoption')
# ax[2].set_ylabel('Adoption of \n Perennial Grass')
# ax2 = ax[2].twinx()
# lns2 = ax2.plot(result_V4_nopolicy['land_use_areas'][:30,2]/10**6,'r',label='Miscanthus Area')
# ax2.tick_params(axis='y')
# ax2.set_ylabel('Area of Perennial Grass \n (Million ha)')
# ax[2].set_xlabel('Year')
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax[2].legend(lns, labs, loc="lower right")
#
# plt.tight_layout(pad=0.2, w_pad=1, h_pad=0.2)
# fig.savefig("./ABM_environment_interaction.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
#
# RFS_mandate = pd.read_csv('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Version_4/simulations/RFS/data/RFS_volume.csv')
# fig, ax = plt.subplots(3,1,figsize=(6, 6))
# ax[0].plot(result_TMDL['price_adj_hist'][:,0]/3.78,'b')
# ax[0].plot(cell_eth_price_baseline['endogenous_ethanol ($/L)'],'r--')
# ax[0].set_ylabel('Ethanol Price ($/L)')
# ax[0].legend(['Cellulosic Premium','Corn Ethanol'])
# ax[0].set_xticks([])
#
# ax[1].plot(result_TMDL['biofuel_production'][result_TMDL['biofuel_type']==1].sum(0)/10**9)
# ax[1].plot(result_TMDL['biofuel_production'][result_TMDL['biofuel_type']==2].sum(0)/10**9)
# ax[1].plot(RFS_mandate['RFS_volume (Million L)'][:30]/1000,'r--')
# ax[1].legend(['Corn Ethanol','Cellulosic Ethanol','RFS Cellulosic Mandate'])
# ax[1].set_ylabel('Ethanol Production \n (Billion L)')
# ax[1].set_xticks([])
#
# ax[2].plot(result_TMDL['ref_inv_cost_adj_his'][:,1])
# ax[2].set_ylabel('Refinery Cost \n Reduction Factor')
# ax[2].set_xlabel('Year')
#
# plt.tight_layout(pad=2, w_pad=2, h_pad=0.2)
# fig.savefig("./ABM_gov_ref_commu_interaction.jpg",bbox_inches='tight',pad_inches=0.05, dpi=300)
#
#
