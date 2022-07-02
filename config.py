import numpy as np
import os
import pandas as pd
import parameters as params
# storing all global variables
os.chdir(params.folder)

'''
rewrite the farmer attribute and initial state to change farmer composition
'''
def rewrite_farmer_composition(farmer_change_rate):
    # farmer_change_rate: the percentage of change for Type I and Type II faremers
    agent_cluster_ABM = pd.read_excel('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/BN_rule/agent_cluster_ABM.xlsx')
    cluster_ID = agent_cluster_ABM['cluster']
    N_Type_I_old = (cluster_ID==1).sum()/cluster_ID.size
    N_Type_II_old = (cluster_ID == 2).sum() / cluster_ID.size
    N_Type_III_old = (cluster_ID == 3).sum() / cluster_ID.size
    N_Type_IV_old = (cluster_ID == 4).sum() / cluster_ID.size

    N_Type_I_new = N_Type_I_old * (1+farmer_change_rate)
    N_Type_II_new = N_Type_II_old * (1 + farmer_change_rate)
    III_to_remaining = N_Type_III_old / (N_Type_III_old + N_Type_IV_old)
    N_Type_III_new = (1-N_Type_I_new-N_Type_II_new) * III_to_remaining
    N_Type_IV_new = (1 - N_Type_I_new - N_Type_II_new) * (1-III_to_remaining)

    farmer_attribute = pd.read_csv('./data/farmer_attributes_backup.csv')
    farmer_ini = pd.read_csv('./data/farmer_ini_state_backup.csv')
    N_farmer = farmer_attribute.shape[0]
    N_Type_I_new = np.round(N_Type_I_new * N_farmer).astype(int)
    N_Type_II_new = np.round(N_Type_II_new * N_farmer).astype(int)
    N_Type_III_new = np.round(N_Type_III_new * N_farmer).astype(int)
    N_Type_IV_new = N_farmer - N_Type_I_new - N_Type_II_new - N_Type_III_new

    ID_Type_I = cluster_ID.index[cluster_ID==1].to_frame()
    ID_Type_II = cluster_ID.index[cluster_ID == 2].to_frame()
    ID_Type_III = cluster_ID.index[cluster_ID == 3].to_frame()
    ID_Type_IV = cluster_ID.index[cluster_ID == 4].to_frame()

    ID_Type_I_new = ID_Type_I.sample(N_Type_I_new, replace=True)
    ID_Type_II_new = ID_Type_II.sample(N_Type_II_new, replace=True)
    ID_Type_III_new = ID_Type_III.sample(N_Type_III_new, replace=True)
    ID_Type_IV_new = ID_Type_IV.sample(N_Type_IV_new, replace=True)
    ID_new = pd.concat([ID_Type_I_new, ID_Type_II_new, ID_Type_III_new, ID_Type_IV_new])
    ID_new = ID_new.sample(frac=1)

    farmer_attribute['info_use'] = agent_cluster_ABM['info_use'].to_numpy()[ID_new.to_numpy().flatten()]
    farmer_attribute['benefit'] = agent_cluster_ABM['benefit'].to_numpy()[ID_new.to_numpy().flatten()]
    farmer_attribute['concern'] = agent_cluster_ABM['concern'].to_numpy()[ID_new.to_numpy().flatten()]
    farmer_attribute['lql'] = agent_cluster_ABM['lql'].to_numpy()[ID_new.to_numpy().flatten()]
    farmer_attribute['type'] = agent_cluster_ABM['cluster'].to_numpy()[ID_new.to_numpy().flatten()]
    farmer_ini['max_fam'] = agent_cluster_ABM['max_fam'].to_numpy()[ID_new.to_numpy().flatten()]

    farmer_attribute.to_csv('./data/farmer_attributes.csv',index=False)
    farmer_ini.to_csv('./data/farmer_ini_state.csv', index=False)

# rewrite_farmer_composition(0.5)
"""
end of the rewrite, comment off if rewrite is not necessary
"""

inflation_rate = 0.02
simu_horizon = params.simu_horizon
install_fail_rate_switch = 0
# cost_reduction = 1
maintain_RFS = params.maintain_RFS

land_rent = params.land_rent  # in $/ha
marginal_land_rent_adj = params.marginal_land_rent_adj
# land_rent = 218 * 2.47  # in $/ha
# farmer_dist_matrix = np.loadtxt('./data/ref_farmer_dist_matrix.csv',delimiter=',',skiprows = 0)
N_can_ref_per_loc = 6 # number of candidate refineries in each location

# for farmer
enhanced_neighbor_impact = params.enhanced_neighbor_impact

average_crop_yields = np.asarray([13,4,26,10,8,65])

N_patch = 33965
N_in_soil = 1.05
year = 0
ferti_price = 0.55
# stover_harvest_cost = 51.5 # $/t
stover_harvest_cost = 31.5 # $/t
stover_harvest_ratio = 0.1
install_fail_rate_mis = 0.1
# farmer_attitude_threshold = 0.8
mis_carbon_credit = 4.7 # t CO2e/ha/year
mis_N_credit = 66.2 # kg/ha
mis_harvest_loss = 0.2
mis_trans_loss = 0.02
mis_storage_loss = 0.07


patch_yield_table=[]
for i in range(30):
    temp = np.loadtxt('./data/resp_matrix/patch_yield_table_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)
    temp[:,1] *= 1.18
    temp[:,2] *= 0.64
    temp[:, 3] *= 0.64
    patch_yield_table.append(np.loadtxt('./data/resp_matrix/patch_yield_table_'+str(i+1)+'.csv',delimiter=',',skiprows = 1))
patch_yield = np.asarray(patch_yield_table)
patch_yield[7,:,3] = patch_yield[6,:,3]
patch_yield[17,:,3] = patch_yield[16,:,3]
patch_yield[27,:,3] = patch_yield[26,:,3]
patch_yield[:,:,3] *= (1 - mis_harvest_loss)  # discount for the harvest loss of miscanthus
patch_yield_table_mean = patch_yield.mean(0)
patch_yield_table_mean_sub1 = patch_yield[0::2,:,:].mean(0) # for estimating expected soybean yield of corn soy rotation
patch_yield_table_mean_sub2 = patch_yield[1::2,:,:].mean(0) # for estimating expected corn yield of corn soy rotation
patch_yield_table_std = patch_yield.std(0)
patch_yield_table_std_sub1 = patch_yield[0::2,:,:].std(0)
patch_yield_table_std_sub2 = patch_yield[1::2,:,:].std(0)

fertilizer_table = np.loadtxt('./data/fertilizer_table.csv',delimiter=',',skiprows = 1)
perennial_yield_adj_table = np.loadtxt('./data/perennial_yield_adj_table.csv',delimiter=',',skiprows = 1,usecols=(1,2))
# soil_erosion_table = np.loadtxt('./data/soil_erosion_table.csv',delimiter=',',skiprows = 1, usecols=(1,2,3,4,5,6,7,8))
farmer_cost_table = np.loadtxt('./data/farmer_cost_table.csv',delimiter=',',skiprows = 1, usecols=(1,2,3,4,5,6))

patch_N_loads = []
for i in range(30):
    patch_N_loads.append(np.loadtxt('./data/resp_matrix/patch_N_loads_'+str(i+1)+'.csv',delimiter=',',skiprows = 1, usecols=(1,2,3,4,5,6,7,8)))
patch_N_loads_mean = np.asarray(patch_N_loads).mean(0)
patch_N_loads_std = np.asarray(patch_N_loads).std(0)

patch_water_yield = []
for i in range(30):
    patch_water_yield.append(np.loadtxt('./data/resp_matrix/patch_water_use_'+str(i+1)+'.csv',delimiter=',',skiprows = 1, usecols=(1,2,3,4,5,6,7,8)))
patch_water_yield_mean = np.asarray(patch_water_yield).mean(0)
patch_water_yield_std = np.asarray(patch_water_yield).std(0)

patch_C_table = []
for i in range(30):
    patch_C_table.append(np.loadtxt('./data/resp_matrix/patch_carbon_table_'+str(i+1)+'.csv',delimiter=',',skiprows = 1, usecols=(1,2,3,4,5,6,7,8)))
patch_C = np.asarray(patch_C_table)
patch_C_sequest = np.zeros(patch_C.shape)
patch_C_sequest[0:29,:,:] = patch_C[1:30,:,:] - patch_C[0:29,:,:]
patch_C_sequest[29,:,:] = patch_C_sequest[28,:,:]
patch_C_table_mean = patch_C.mean(0)
patch_C_table_std = patch_C.std(0)


# for refinery
learn_by_do_rate = params.learn_by_do_rate # the cost updating rate for learning by doing
# learn_by_do_rate = 0 # the cost updating rate for learning by doing, assuming no learning by doing
base_cap_for_learn_by_do = params.base_cap_for_learn_by_do # the base capacity for learning by doing, L
base_feed_amount_for_learn_by_do = params.base_feed_amount_for_learn_by_do # the base feedstock amount for the learning by doing of supply chain business, ton
base_BCHP_cap_for_learn_by_do = params.base_BCHP_cap_for_learn_by_do # the base capacity for BCHP
N_can_ref_locs = 10 # the total number of candidate refinery locations
N_can_BCHP = 50
N_can_cofire = 4
ref_ref_dist_matrix = np.loadtxt('./data/ref_ref_dist_matrix.csv',delimiter=',',skiprows = 0)
ref_farmer_dist_matrix = np.loadtxt('./data/ref_farmer_dist_matrix.csv',delimiter=',',skiprows = 0)
refinery_feedstock_table = np.loadtxt('./data/refinery_feedstock_table.csv',delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7,8))
refinery_product_yield_table = np.loadtxt('./data/refinery_product_yield_table.csv',delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7))
refinery_water_use_table = np.loadtxt('./data/refinery_water_use_table.csv',delimiter=',',skiprows=1,usecols=(1))

biofacility_product_yield_table = np.loadtxt('./data/biofacility_product_yield_table.csv',delimiter=',',skiprows=1,usecols=(1,2,3,4,5,6,7))
BCHP_farmer_dist_matrix = np.loadtxt('./data/BCHP_farmer_dist_matrix.csv',delimiter=',',skiprows = 0)
cofire_farmer_dist_matrix = np.loadtxt('./data/cofire_farmer_dist_matrix.csv',delimiter=',',skiprows = 0)
# ref_cost_adj = np.loadtxt('./data/ref_cost_adj.csv',delimiter=',',skiprows=1,usecols=(1,2))
ethanol_equivilents = [1, 1.5]
patch_influence_range = 50 # the range of influence for refinery feedstock collection
system_boundary_radius = 100
refinery_esti_farmer_discount_rate = 0.08
allowable_defecit = params.allowable_defecit
refinery_life_span = 50
refinery_delta_LU_limit = 0.5
max_ref_cap = 1.2*10**9  # maximum capacity of a single refinery in L ethanol
max_watersh_cap_corn = 3*10**9  # maximum corn ref capacity of all the watershed
max_watersh_cap_cell = 3*10**9  # maximum cell ref capacity of all the watershed

ref_cost_table = np.loadtxt('./data/ref_cost_table.csv',delimiter=',',skiprows=1)
# ref_fix_invest_costs = [15.67* 10**6, 51 * 10**6,65* 10**6,65* 10**6]  # fixed and viable investment costs for refinery, 4 elements reflects the 4 types of refinerys
# ref_via_invest_costs = [0,0,0,0]
# ref_base_cap = [150* 10**6,150* 10**6,48* 10**6,48* 10**6]
# ref_fix_oper_costs = [0.025,0.05,0.1,0.16] # fixed and viable production costs for refinery, 4 elements reflects the 4 types of refinerys
# ref_via_oper_costs = [0.019,0.03,0.1,0.133]
ref_fix_invest_costs = ref_cost_table[:,1]  # fixed and viable investment costs for refinery, 4 elements reflects the 4 types of refinerys
ref_via_invest_costs = ref_cost_table[:,2]
ref_base_cap = ref_cost_table[:,3]
ref_fix_oper_costs = ref_cost_table[:,4] # fixed and viable production costs for refinery, 4 elements reflects the 4 types of refinerys
ref_via_oper_costs = ref_cost_table[:,5]

can_refinery_attribute = np.loadtxt('./data/can_refinery_attributes.csv', delimiter=',', skiprows=1)

trans_costs = np.asarray([0.16,0.16,0.25,0.25,0.25,0.25,0.25,0.25])*1.5 # different for each crop, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
                                #               5 for bagasse, 6 for sorghum, 7 for lipidcane
storage_costs = np.asarray([13.227648,33.06933,16.47,16.47,16.47,16.47,16.47,16.47])  # assuming the average storage period is 6 month
empirical_risks = np.asarray([0.3, 0.2, 0.1]) # [P_flood, P_drought, P_fail]

ref_job_table = np.loadtxt('./data/ref_job_table.csv',delimiter=',',skiprows=1)

ref_interest_rate = [0.04,0.04] # the interest rate for the refinery, 0 for large agrobusiness owned, 1 for co-op local owned

# for consumer
ini_WP = params.ini_WP    # initial willingness to pay extra for cellulosic ethanol in $/gallon
IRW = params.IRW      # increasing rate factor of WP
max_WP = params.max_WP    # maximum value of WP
ini_ethanol_price = 1.4 # the initial ethanol price before consumer pay extra

# ini_WP = 0    # initial willingness to pay extra for cellulosic ethanol in $/gallon
# IRW = 0      # increasing rate of WP
# max_WP = 0    # maximum value of WP
# ini_ethanol_price = 1.4 # the initial ethanol price before consumer pay extra

# for community
community_attribute = np.loadtxt('./data/community_attributes.csv', delimiter=',', skiprows=1)

# for model simulation
# price_update_rate = 0.1
price_update_rate = params.price_update_rate
external_provided_prices = np.loadtxt('./data/external_provided_prices.csv',delimiter=',',skiprows=1)
external_provided_prices = external_provided_prices[:,1:]
gov_related_prices = np.loadtxt('./data/gov_related_prices.csv',delimiter=',',skiprows=1)
gov_related_prices = gov_related_prices[:,1:]
endogenous_ethanol_price = np.loadtxt('./data/endogenous_ethanol_price.csv',delimiter=',',skiprows=1,usecols=(1))

ref_pred_prices = []
Prcp_data = np.loadtxt('./data/Prcp_data.csv',delimiter=',',skiprows=1,usecols=(1))

