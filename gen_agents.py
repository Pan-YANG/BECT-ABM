# all functions to generate agents
import numpy as np
import agents
import government as gov
import pandas as pd

survey_data_loc = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/BN_rule/'
survey_data = pd.read_csv(survey_data_loc+'BN_data_start1_adj.csv')
bn_df=pd.read_csv(survey_data_loc+'BN_df.csv') # import the MC samples for BN inferencing

def generate_farmer(agent_num,patch_attribute,farmer_attribute, patch_ini_state, farmer_ini_state, adjacency_matrix, neighbor_influence_matrix):
    # function to generate farmer agents
    # agent_num: total number of farmer agents
    # patch_attribute: a matrix showing the attributes of all patches, 0 for the patch ID, 1 for the farmer ID, 2 for the area,
    #       3 for the CRP eligibility, 4 for the slope, 5 for the soil loss coefficient for regular crop
    # farmer_attribute: a matrix showing the attributes of all farmers, 0 for the farmer ID, 1 for the time preference,
    #       2 for the risk aversion, 3 for the loss aversion, 4 for the learning rate, 5 for the sensitivity to community pressure,
    #       6 for the minimum sensitivity to environment, 7 for the maximum sensitivity to environment, 8 for the price updating rate,
    #       9 for the flooding risk updating rate, 10 for the community ID, 11 for the economic_of_scale_factor
    # patch_ini_state: initial state variable value for all patches, 0 for the patch ID, 1 for the farmer ID, 2 for the land use, 3 for yield
    # farmer_ini_state: initial state variable value for all farmers, 0 for the farmer ID, 1-6 the forecasted price (1 for corn,
    #                   2 for soy, 3 mis, 4 switch, 5 sorghum, 6 cane; price of 3 and 4 are always equal), 7-8 the forecasted climate (7 for flooding probability, 8 for drought)
    #                   9 for the initial environmental sensitivity, 10 for the initial attitude toward perennial grass
    # adjacency_matrix: a matrix showing the neighbors of each farmer
    # neighbor_influence_matrix: a matrix showing the level of influence of each neighbor

    agent_list = []
    farmer_ID = farmer_attribute[:,0]
    for idx in range(agent_num):
        ID = int(farmer_ID[idx])
        temp = farmer_attribute[ID,:]

        neighbors = np.where(adjacency_matrix[ID,:]==1)
        neighbors = np.asarray(neighbors).flatten() # identify the neighbors of each farmer
        # neighbor_weights = neighbor_influence_matrix[ID,neighbors]
        # neighbor_weights = np.asarray(neighbor_weights).flatten() # retrieve the influence of each neighbor

        patch_IDs = np.argwhere(patch_attribute[:, 1] == ID).flatten() # identify the land patches owned by each farmer
        N_patch = patch_IDs.size
        Areas = patch_attribute[patch_IDs,2]
        Farm_areas = Areas.sum() * 2.47 # convert total area into acre
        if Farm_areas/11.7 <= 500: # on average, one farmer agent represents 11.7 farmers
            farm_size = 1
        elif Farm_areas/11.7 < 1000:
            farm_size = 2
        else:
            farm_size = 3
        CRP_eligible = patch_attribute[patch_IDs,3]
        slopes = patch_attribute[patch_IDs,4]
        soil_loss = patch_attribute[patch_IDs,5]

        Attributes = {'discount_factor': temp[1], 'risk_factor': temp[2],'farm_size': farm_size, 'info_use':temp[3],
                      'benefit':temp[4], 'concern': temp[5], 'lql': temp[6], 'PT_imp_env':[], 'adopt_priority':[],
                      'loss_factor': temp[7], 'learning_rate': temp[8], 'price_update_rate': temp[9], 'flood_update_rate':temp[10],
                      'community_ID': int(temp[11]),'cost_adj_factor':temp[12],'type':temp[13],'sensi_community': temp[14],'neighbors': neighbors,
                      'patch_ID': patch_IDs,'patch_areas': Areas, 'patch_CRP_eligible': CRP_eligible,
                      'patch_slope': slopes, 'soil_loss': soil_loss}

        temp = farmer_ini_state[ID,:]
        land_uses = patch_ini_state[patch_IDs, 2].astype(int)
        crop_yield = patch_ini_state[patch_IDs, 3]
        States = {'price_prior': [temp[1:7]], 'price_post': [], 'price_received': [], 'climate_forecasts': [[temp[7],temp[8],0.1]],
                  'land_use': [land_uses], 'contract': [np.repeat(-1,N_patch)], 'peren_age': [np.repeat(-2,N_patch)],
                  'failed': [np.repeat(False,N_patch)], 'yield': [crop_yield],'N_release': [], 'C_sequest':[], 'ferti_use': [],'revenue': [],
                  'TMDL_eligible':[], 'BCAP_eligible':[],'imp_env':[],'environ_sen':[temp[9]], 'peer_ec':[0], 'max_fam':[temp[10]],
                  'SC_Will':[], 'SC_Ratio':[], 'is_adopt':[]}

        Temp = {'contract_land_use': [],'opt_land_use':[],'attitude':[],'opt_peren':[],'peren_age_refresh':[],
                'most_likely_peren_ID':[],'patch_received_prices':[],'patch_available_for_sale':[],
                'stover_available_for_sale':[], 'already_negeotiated':[]}

        agent = agents.Farmer(ID, Attributes, States,Temp)
        agent_list.append(agent)
    return agent_list

def gen_community(agent_num,community_attribute,community_ini_state):
    # function to generate community classes
    # agent_num: total number of communities
    # community_attribute: a matrix showing the attributes of all communities, 0 for the community ID, 1 for accept_threshold,
    #       2 for base_increase_environ, 3 for max_attitude, 4 for sensi_N, 5 for ratio_farmer, 6 for N_limit,
    #       7 for maximum allowable land use, 8 for maximum jobs, 9 for maximum refinery capacity
    # community_ini_state: a matrix showing the initial states of all communities, 0 for the community ID, 1 for attitude,
    #       2 for current revenue, 3 for current water availability, 4 for w_r, 5 for w_j, 6 for w_e, 7 for w_w, 8 for w_l,
    #       9 for w

    agent_list = []
    community_ID = community_attribute[:,0]
    for idx in range(agent_num):
        ID = int(community_ID[idx])
        temp = community_attribute[ID, :]

        Attributes = {'accept_threshold': temp[1], 'base_increase_environ': temp[2], 'max_attitude': temp[3],
                      'sensi_N': temp[4], 'ratio_farmer': temp[5], 'N_limit': temp[6],'max_LU': temp[7],'max_job':temp[8],
                      'max_cap':temp[9],'average_N_loads':[]}
        # here base_increase_environ is the base increase rate of environmental awareness, sensi_N is the sensitivity to N leaching
        # N_limit is the maximum N leaching allowed by TMDL

        temp = community_ini_state[ID, :]
        States = {'attitude': [temp[1]], 'revenue': [temp[2]], 'water_avail': [temp[3]], 'N_release':[temp[4]],'denial':[0]}
        # here revenue and water_avail are both revenue and water availability at the end of previous time step
        Temp = {'WU': []}

        agent = agents.Community(ID, Attributes, States,Temp)
        agent_list.append(agent)
    return agent_list

def gen_candidate_refinery(agent_num,refinery_attribute,dist_matrix):
    # function to generate candidate refinery classes
    # agent_num: total number of candidate refineries
    # refinery_attribute: a matrix showing the attributes of all candidate refinery, 0 for the refinery ID, 1 for location ID,
    #       2 for community ID, 3 for technology type, 4 for capacity, 5 for IRR_min, 6 for a binary variable if the refinery is co-op
    # dist_matrix: distance matrix with farmer

    agent_list = []
    candidate_ref_ID = refinery_attribute[:, 0]
    for idx in range(agent_num):
        ID = int(candidate_ref_ID[idx])
        temp = refinery_attribute[ID, :]

        dist_farmer = dist_matrix[ID, :]

        Attributes = { 'loc_ID':int(temp[1]), 'community_ID':int(temp[2]), 'refinery_type': int(temp[3]),'capacity': temp[4],
                      'IRR_min': temp[5], 'co-op':int(temp[6]), 'dist_farmer': dist_farmer}

        States = {'WU':[],'NPV':[],'IRR':[],'feedstock_amount':[],'invest_cost':[],'feed_stock_avail':[],
                  'invest_cost_adj':[],'interest_payment':[],'aver_dist':[],'fix_cost':[],'via_cost':[]}

        agent = agents.Can_ref(ID, Attributes, States)
        agent_list.append(agent)
    return agent_list

def gen_refinery(agent_num,refinery_attribute,dist_matrix,bagasse_amount):
    # function to generate refinery classes
    # agent_num: total number of refineries
    # refinery_attribute: a matrix showing the attributes of all refinery, 0 for the refinery ID, 1 for the location ID, 2-11 for feedstock type,
    #       12 for capacity, 13 for technology type, 14 for viable cost, 15 for fixed cost,
    #       16 for allowable deficit, 17 for the total initial capital investment in the refinery, 18 for annual interest payment, 19 for binary variable of co-op
    # refinery_ini_state: initial state variable value for all refineries, 0 for the refinery ID, 1 for contracted feedstock,
    #       2 for purchased feedstock, 3 for sold feedstock, 4 for year of production,
    #       5 for accepted subsidy, 6 for profit, 7 for water use
    # biofuel_yields: a list of yields for different biofuels, [biodisel, ethanol]
    # byproduct_yields: a list of yields for byproducts, [DDSG, XX, XX]
    # dist_matrix: distance matrix with farmer

    agent_list = []
    refinery_ID = refinery_attribute[:, 0].astype(int)
    for idx in range(agent_num):
        ID = int(refinery_ID[idx])
        temp = refinery_attribute[ID, :]

        dist_farmer = dist_matrix[ID, :]
        tech_type = int(temp[13])
        tax_reduction = gov.ref_taxs[tech_type,0]
        tax_rate = gov.ref_taxs[tech_type,1]
        subsidy = gov.ref_subsidys[tech_type,2]

        Attributes = {'loc_ID': int(temp[1]), 'feedstock_amount':temp[2:12],'capacity': [temp[12]],
                      'tech_type': tech_type, 'tax_reduction':tax_reduction, # tech_type, 1 for corn ethanol, 2 for cellulosic ethanol, 3 for biodiesel, 4 for co-production of biodiesel and ethanol
                      'tax_rate': tax_rate,'subsidy':subsidy, 'via_cost':temp[14], 'fix_cost':temp[15],
                      'max_deficit':temp[16],'invest': temp[17], 'interest_payment':[temp[18]],'co-op':int(temp[19]), 'dist_farmer': dist_farmer,
                       'bagasse_amount':bagasse_amount[ID],'start_year':[],'aver_dist':[]}

        States = {'purchased_feedstock': [], 'sold_feedstock': [], 'purchased_feed_dist':[],
                  'production_year':[0], 'accepted_subsidy':[], 'profit':[], 'WU':[], # 'production_year' start from -1, meaning the building of refinery takes 2 years
                  'biofuel_production':[],'byproduct_production':[],'contracted_patch_amount':[np.empty((0,10))],
                  'contracted_patch_price':[np.empty(0)],'contracted_patch_supply':[np.empty((0,10))],
                  'contracted_patch_dist':[np.empty(0)],'contracted_patch_ID':[np.empty(0)],'contracted_farmer_ID':[np.empty(0)]}
        # contacted_patch_ will be list of matrixs, with the patch ID of contracted patch as the matrix row
        Temp = {'purchased_feedstock': [],'temp_cap':[]}

        agent = agents.Refinery(ID, Attributes, States,Temp)
        agent_list.append(agent)
    return agent_list

def add_new_refinery(ref_list,agent_num,refinery_attribute,dist_matrix,bagasse_amount):
    # function to generate refinery classes
    # agent_num: total number of refineries
    # refinery_attribute: a matrix showing the attributes of all refinery, 0 for the refinery ID, 1 for the location ID, 2-11 for feedstock type,
    #       12 for capacity, 13 for technology type, 14 for viable cost, 15 for fixed cost,
    #       16 for allowable deficit, 17 for the total initial capital investment in the refinery, 18 for annual interest payment, 19 for binary variable of co-op
    # refinery_ini_state: initial state variable value for all refineries, 0 for the refinery ID, 1 for contracted feedstock,
    #       2 for purchased feedstock, 3 for sold feedstock, 4 for year of production,
    #       5 for accepted subsidy, 6 for profit, 7 for water use
    # biofuel_yields: a list of yields for different biofuels, [biodisel, ethanol]
    # byproduct_yields: a list of yields for byproducts, [DDSG, XX, XX]
    # dist_matrix: distance matrix with farmer

    agent_list = ref_list
    refinery_ID = refinery_attribute[:, 0]
    for idx in range(agent_num):
        ID = int(refinery_ID[idx])
        temp = refinery_attribute[idx, :]

        if dist_matrix.shape.__len__()==1:
            dist_farmer = dist_matrix
        else:
            dist_farmer = dist_matrix[idx, :]
        tech_type = int(temp[13])
        tax_reduction = gov.ref_taxs[tech_type-1, 0]
        tax_rate = gov.ref_taxs[tech_type-1, 1]
        subsidy = gov.ref_subsidys[tech_type-1, 2]

        Attributes = {'loc_ID': int(temp[1]), 'feedstock_amount':temp[2:12],'capacity': [temp[12]],
                      'tech_type': tech_type, 'tax_reduction':tax_reduction, # tech_type, 1 for corn ethanol, 2 for cellulosic ethanol, 3 for biodiesel, 4 for co-production of biodiesel and ethanol
                      'tax_rate': tax_rate,'subsidy':subsidy, 'via_cost':temp[14], 'fix_cost':temp[15],
                      'max_deficit':temp[16],'invest': temp[17], 'interest_payment':[temp[18]],'co-op':int(temp[19]), 'dist_farmer': dist_farmer,
                       'bagasse_amount':bagasse_amount[idx],'start_year':[],'aver_dist':[]}

        States = {'purchased_feedstock': [], 'sold_feedstock': [],'purchased_feed_dist':[],
                  'production_year':[-1], 'accepted_subsidy':[], 'profit':[], 'WU':[], # 'production_year' start from -1, meaning the building of refinery takes 2 years
                  'biofuel_production':[],'byproduct_production':[],'contracted_patch_amount':[np.empty((0,10))],
                  'contracted_patch_price':[np.empty(0)],'contracted_patch_supply':[np.empty((0,10))],
                  'contracted_patch_dist':[np.empty(0)],'contracted_patch_ID':[np.empty(0)],'contracted_farmer_ID':[np.empty(0)]}
        # contacted_patch_ will be list of matrixs, with the patch ID of contracted patch as the matrix row
        Temp = {'purchased_feedstock': [],'temp_cap':[]}

        agent = agents.Refinery(ID, Attributes, States,Temp)
        agent_list.append(agent)
    return agent_list

def gen_comsumer(agent_num,ini_WP,IRW, max_WP, ini_ethanol_price):
    # function to generate consumer classes
    # agent_num: total number of consumers
    # ini_WP: initial willingness to pay
    # IRW: increasing rate of WP
    # max_WP: maximum value of WP
    # ini_ethanol_price: initial ethanol price payed by consumer

    agent_list = []
    for idx in range(agent_num):
        ID = idx

        Attributes = {'IRW': IRW, 'max_WP': max_WP}

        States = {'willingness_to_pay': [ini_WP], 'ethanol_price': [ini_ethanol_price]}

        agent = agents.consumer(ID, Attributes, States)
        agent_list.append(agent)
    return agent_list

def gen_gov(agent_num,RFS_volume,TMDL, slope_limits,TMDL_N_limits,scaling_factor):
    # function to generate government classes
    # agent_num: total number of consumers
    # TMDL: N limit for TMDL
    # slope_limits: a series of slope limits for identifying critical area
    # RFS_volume: the mandated amount of cellulosic biofuel from RFS

    agent_list = []
    for idx in range(agent_num):
        ID = idx

        Attributes = {'TMDL':TMDL,'slope_limits':slope_limits,'TMDL_N_limits':TMDL_N_limits,'scaling_factor':scaling_factor}

        States = {'CWC_price': [], 'RFS_volume': RFS_volume,'RFS_adjusted_cell_ethanol_price':[],
                  'slope_limit_ID':[0],'N_limit_ID':[0]}

        agent = agents.govern_agent(ID, Attributes, States)
        agent_list.append(agent)
    return agent_list

