import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import os
import cv2
import shelve
# import config
import parameters as params

def extract_ABM_results(dir,is_ABM):
    '''
    function to extract all needed variables from the ABM simulation data
    :param dir: the directory of ABM simulation
    :param is_ABM: whether the simulation considers non-economic behavior, 1 for yes, 0 for no
    :return: a dictionary of ABM simulation results
    '''
    s_data = shelve.open(dir)
    simu_horizon = params.simu_horizon
    # farmer land use history
    farmer_agent_list = s_data['farmer_agent_list']
    land_use_areas = np.zeros((params.simu_horizon + 1, 8))
    for i in range(8):
        land_use_areas[:, i] = compile_land_use(farmer_agent_list, i + 1)
    # farmer revenue history
    revenue_farmer = compile_farmer_states_nopatch(farmer_agent_list, 'revenue', 1)
    # farmer's attitude toward perennial grass
    att_farmer = compile_farmer_states_nopatch(farmer_agent_list,'SC_Will',2)
    # farmer's attitude and adoption by type, only valid for the case with non-economic behavior
    if is_ABM == 1:
        att_farmer_cluster = np.zeros((params.simu_horizon, 4))
        for i in range(params.simu_horizon):
            for j in range(4):
                temp = []
                for k in range(farmer_agent_list.__len__()):
                    if farmer_agent_list[k].Attributes['type'] == j + 1:
                        temp.append(farmer_agent_list[k].States['SC_Will'][i])
                att_farmer_cluster[i, j] = np.asarray(temp).mean()
        adopt_farmer_cluster = np.zeros((params.simu_horizon, 4))
        for i in range(params.simu_horizon):
            for j in range(4):
                temp1 = 0
                temp2 = 0
                for k in range(farmer_agent_list.__len__()):
                    if farmer_agent_list[k].Attributes['type'] == j + 1:
                        adoption = np.isin(farmer_agent_list[k].States['land_use'][i], [3, 4]).sum() > 0 + 0
                        temp1 += adoption
                        temp2 += 1
                adopt_farmer_cluster[i, j] = temp1 / temp2
    # farmer environmental sensitivity
    environ_sensi_farmer = compile_farmer_states_nopatch(farmer_agent_list, 'environ_sen', 2)
    # N release
    N_release = compile_farmer_states_patch(farmer_agent_list, 'N_release', 1)
    # overall land use and contract history
    farmer_adoption = extract_farmer_adaption(farmer_agent_list)
    land_use_his = extract_farmer_land_use(farmer_agent_list, 33965)
    contract_farm, contract_patch = extract_farmer_contract(farmer_agent_list, 33965)
    # biofuel production from refineries
    ref_list_backup = s_data['ref_list_backup']
    N_refs = ref_list_backup.__len__()
    biofuel_production = np.zeros((N_refs, simu_horizon))
    biofuel_type = np.zeros(N_refs, int)
    for i in range(N_refs):
        biofuel_production[i, :] = extract_ref_biofuel_production(ref_list_backup, i, 0, simu_horizon)
        biofuel_type[i] = ref_list_backup[i].Attributes['tech_type']
    # refinery profit
    ref_profit = np.zeros((N_refs, simu_horizon))
    for i in range(N_refs):
        ref_profit[i, :] = extract_ref_var(ref_list_backup, i, 'profit', simu_horizon)
    # community water availability
    community_list = s_data['community_list']
    N_community = community_list.__len__()
    water_available = np.zeros((N_community, simu_horizon + 1))
    commu_N_release = np.zeros((N_community, simu_horizon + 1))
    commu_atti = np.zeros((N_community, simu_horizon + 1))
    commu_denail = np.zeros(N_community)
    for i in range(N_community):
        water_available[i, :] = np.asarray(community_list[i].States['water_avail'])
        commu_N_release[i, :] = np.asarray(community_list[i].States['N_release'])
        commu_atti[i, :] = np.asarray(community_list[i].States['attitude'])
        commu_denail[i] = np.asarray(community_list[i].States['denial']).sum()
    # TMDL patches
    TMDL_eligibilities = extract_farmer_states_patch(farmer_agent_list, 'TMDL_eligible', 33965)
    # BCAP patches
    BCAP_eligibilities = extract_farmer_states_patch(farmer_agent_list, 'BCAP_eligible', 33965)

    # ref locs
    ref_cap_look_up_table = np.asarray([[0, 1, 2, 3, 4], [0, 200, 400, 800, 800]]).T
    corn_ref_pd,cell_ref_pd,bio_facility_pd= extract_ref_loc_his(ref_list_backup, 64,ref_cap_look_up_table,params.simu_horizon)

    if is_ABM == 1:
        # ref_inv_cost_adj_his and ref_pro_cost_adj_his
        ref_inv_cost_adj_his = s_data['ref_inv_cost_adj_his']
        ref_pro_cost_adj_his = s_data['ref_pro_cost_adj_his']

        # ethanol prices
        price_adj_hist = s_data['price_adj_hist']
        price_adj_hist = np.asarray(price_adj_hist)

        neighbor_impact = np.zeros(params.simu_horizon+1)
        for i in range(params.simu_horizon+1):
            neighbor_impact_temp = np.zeros(1000)
            for j in range(1000):
                peer_ec = farmer_agent_list[j].States['peer_ec'][i]
                if peer_ec <= 1:
                    neighbor_impact_temp[j] = s_data['ABM_parameters']['enhanced_neighbor_impact'][0]
                elif peer_ec == 2:
                    neighbor_impact_temp[j] = s_data['ABM_parameters']['enhanced_neighbor_impact'][1]
                elif peer_ec == 3:
                    neighbor_impact_temp[j] = s_data['ABM_parameters']['enhanced_neighbor_impact'][2]
            neighbor_impact[i] = neighbor_impact_temp.mean()


    if is_ABM==0:
        return {
            'land_use_areas': land_use_areas,
            'revenue_farmer': revenue_farmer,
            'att_farmer': att_farmer,
            'environ_sensi_farmer': environ_sensi_farmer,
            'N_release': N_release,
            'farmer_adoption': farmer_adoption,
            'land_use_his': land_use_his,
            'contract_farm': contract_farm,
            'contract_patch': contract_patch,
            'biofuel_production': biofuel_production,
            'biofuel_type': biofuel_type,
            'ref_profit': ref_profit,
            'water_available': water_available,
            'commu_N_release': commu_N_release,
            'commu_atti': commu_atti,
            'commu_denail': commu_denail,
            'TMDL_eligibilities': TMDL_eligibilities,
            'BCAP_eligibilities': BCAP_eligibilities,
            'corn_ref_pd': corn_ref_pd,
            'cell_ref_pd': cell_ref_pd,
            'bio_facility_pd': bio_facility_pd,
        }
    else:
        return {
            'land_use_areas': land_use_areas,
            'revenue_farmer': revenue_farmer,
            'att_farmer': att_farmer,
            'att_farmer_cluster': att_farmer_cluster,
            'adopt_farmer_cluster': adopt_farmer_cluster,
            'environ_sensi_farmer': environ_sensi_farmer,
            'N_release': N_release,
            'farmer_adoption': farmer_adoption,
            'land_use_his': land_use_his,
            'contract_farm': contract_farm,
            'contract_patch': contract_patch,
            'biofuel_production': biofuel_production,
            'biofuel_type': biofuel_type,
            'ref_profit': ref_profit,
            'water_available': water_available,
            'commu_N_release': commu_N_release,
            'commu_atti': commu_atti,
            'commu_denail': commu_denail,
            'TMDL_eligibilities': TMDL_eligibilities,
            'BCAP_eligibilities': BCAP_eligibilities,
            'corn_ref_pd': corn_ref_pd,
            'cell_ref_pd': cell_ref_pd,
            'bio_facility_pd': bio_facility_pd,
            'ref_inv_cost_adj_his': ref_inv_cost_adj_his,
            'ref_pro_cost_adj_his': ref_pro_cost_adj_his,
            'price_adj_hist': price_adj_hist,
            'neighbor_impact': neighbor_impact,
        }

def cal_min_IRR_RFS_policy_commitment(RFS_singal):
    IRR = [0.25]
    for i in range(RFS_singal.__len__()):
        if RFS_singal[i] == -1:
            IRR.append(min(IRR[-1]+0.03,0.35))
        elif RFS_singal[i] == 1:
            if RFS_singal[i-1] == 1:
                IRR.append(max(IRR[-1]-0.03,0.15))
            else:
                IRR.append(IRR[-1])
        else:
            IRR.append(IRR[-1])
    return IRR

def compile_land_use(farmer_list,land_use):
    # function to compile the land use areas for a specific land use
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States['land_use'].__len__()
    land_use_area = np.zeros(N_year)
    for i in range(N_year):
        area_temp = 0
        for j in range(N_farmer):
            N_patch = farmer_list[j].Attributes['patch_ID'].__len__()
            for k in range(N_patch):
                if farmer_list[j].States['land_use'][i][k] == land_use:
                    area_temp += farmer_list[j].Attributes['patch_areas'][k]
        land_use_area[i] = area_temp
    return land_use_area

def compile_farmer_states_nopatch(farmer_list,arg_name,stat_type):
    # a general function to compile a non-patch-specific state variables for farmers
    # stat_type: 1 for sum, 2 for mean
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States[arg_name].__len__()
    state_var = np.zeros(N_year)
    for i in range(N_year):
        var_temp = 0
        for j in range(N_farmer):
            try:
                var_temp += farmer_list[j].States[arg_name][i]
            except:
                var_temp += farmer_list[j].States[arg_name][i-1]
        if stat_type == 1:
            state_var[i] = var_temp
        elif stat_type == 2:
            state_var[i] = var_temp/N_farmer
    return state_var

def compile_farmer_states_patch(farmer_list,arg_name,is_time_area):
    # a general function to compile a patch-specific state variables for farmers
    # is_time_area: 1 needs to time area, 0 not
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States[arg_name].__len__()
    state_var = np.zeros(N_year)
    for i in range(N_year):
        var_temp = 0
        for j in range(N_farmer):
            N_patch = farmer_list[j].Attributes['patch_ID'].__len__()
            for k in range(N_patch):
                if is_time_area == 0:
                    var_temp += farmer_list[j].States[arg_name][i][k]
                else:
                    var_temp += farmer_list[j].States[arg_name][i][k] * farmer_list[j].Attributes['patch_areas'][k]
        state_var[i] = var_temp
    return state_var

def extract_farmer_states_nopatch(farmer_list,arg_name):
    # a general function to extract a specific state variables for all farmers
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States[arg_name].__len__()
    state_var = np.zeros((N_farmer,N_year))
    for i in range(N_year):
        var_temp = 0
        for j in range(N_farmer):
            state_var[j,i] = farmer_list[j].States[arg_name][i]
    return state_var

def extract_farmer_states_patch(farmer_list,arg_name,N_patch):
    # a general function to extract a patch-specific state variables for farmers
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States[arg_name].__len__()
    state_var = np.zeros((N_patch, N_year))
    for i in range(N_year):
        for j in range(N_farmer):
            N_patch = farmer_list[j].Attributes['patch_ID'].__len__()
            for k in range(N_patch):
                state_var[farmer_list[j].Attributes['patch_ID'][k],i] = farmer_list[j].States[arg_name][i][k]
    return state_var

def extract_farmer_adaption(farmer_list):
    # function to extract all the perennial grass adopters
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States['land_use'].__len__()
    adoption = np.zeros((N_farmer,N_year))
    for i in range(N_year):
        for j in range(N_farmer):
            adoption[j,i]=np.isin(farmer_list[j].States['land_use'][i],[3,4]).sum()>0 + 0
    return adoption

def extract_farmer_land_use(farmer_list,N_patch):
    # function to extract all the farmer's land use histories
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States['land_use'].__len__()
    land_use_his = np.zeros((N_patch, N_year))
    for i in range(N_year):
        for j in range(N_farmer):
            N_patch_temp = farmer_list[j].Attributes['patch_ID'].size
            for k in range(N_patch_temp):
                land_use_his[farmer_list[j].Attributes['patch_ID'][k],i] = farmer_list[j].States['land_use'][i][k]
    return land_use_his.astype(int)

def extract_farmer_contract(farmer_list,N_patch):
    # function to extract all the farmer's contract condition
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States['land_use'].__len__()
    contract_patch = np.zeros((N_patch, N_year))
    contract_farm = np.zeros((N_farmer,N_year))
    for i in range(N_year):
        for j in range(N_farmer):
            N_patch_temp = farmer_list[j].Attributes['patch_ID'].size
            contract_farm[j, i] = np.isin(farmer_list[j].States['contract'][i], [1]).sum() > 0 + 0
            for k in range(N_patch_temp):
                contract_patch[farmer_list[j].Attributes['patch_ID'][k], i] = farmer_list[j].States['contract'][i][k]
    return contract_farm, contract_patch

def extract_farmer_states_nopatch(farmer_list,arg_name):
    # a general function to compile a non-patch-specific state variables for farmers
    # stat_type: 1 for sum, 2 for mean
    N_farmer = farmer_list.__len__()
    N_year = farmer_list[0].States[arg_name].__len__()
    state_var = np.zeros((N_farmer,N_year))
    for i in range(N_year):
        for j in range(N_farmer):
            try:
                state_var[j,i] = farmer_list[j].States[arg_name][i]
            except:
                state_var[j, i] = farmer_list[j].States[arg_name][i-1]

    return state_var

def extract_ref_biofuel_production(ref_list,ref_no,biofuel_type,N_year):
    # function to extract biofuel and byproduct production amount for each refinery
    # biofuel_type: 0 for ethanol, 1 for biodiesel, 2 for bagasse, 3 for DDGS, 4 for glycerol
    biofuel_production = np.zeros((1, N_year))
    temp = 0
    for i in range(N_year):
        if i < ref_list[ref_no].Attributes['start_year']:
            biofuel_production[0,i] = 0
            temp = i + 1
        elif (i - temp) < ref_list[ref_no].States['biofuel_production'].__len__():
            if biofuel_type < 2: # if it is about biofuel
                biofuel_production[0,i] = ref_list[ref_no].States['biofuel_production'][i - temp][biofuel_type]
            else:
                biofuel_production[0, i] = ref_list[ref_no].States['byproduct_production'][i - temp][biofuel_type-2]
    return biofuel_production

def extract_ref_var(ref_list,ref_no,var_name,N_year):
    # function to extract specific refinery state variable
    # var_name: the name of state variable
    state_var = np.zeros(N_year)
    temp = 0
    for i in range(N_year):
        if i < ref_list[ref_no].Attributes['start_year']:
            state_var[i] = 0
            temp = i + 1
        elif (i - temp) < ref_list[ref_no].States['biofuel_production'].__len__():
            state_var[i] = ref_list[ref_no].States[var_name][i-temp]
    return state_var

def look_up_ref_cap(ref_cap,ref_cap_look_up_table):
    # a function to convert refinery capcacity into categories
    # ref_cap: an 1d array of refinery capacity
    # ref_cap_look_up_table: a lookup table for conversion
    N_ref = ref_cap.size
    N_cat = ref_cap_look_up_table.shape[0]
    ref_cap_cat = np.zeros(ref_cap.shape,int)
    for i in range(N_ref):
        for j in range(N_cat-1):
            if (ref_cap[i]>=ref_cap_look_up_table[j,1]) & (ref_cap[i]<ref_cap_look_up_table[j+1,1]):
                ref_cap_cat[i] = ref_cap_look_up_table[j, 0]
        if ref_cap[i] > ref_cap_look_up_table[-1, 1]:
            ref_cap_cat[i] = ref_cap_look_up_table[-1, 0]
    return ref_cap_cat

def extract_ref_loc_his(ref_list,N_can_loc,ref_cap_look_up_table,N_year):
    # function to extract the history of refinery location and capacity
    # N_can_loc: the total number of candidate location
    # ref_cap_look_up_table: a lookup table to convert refinery capcacity into categories
    # returns a pandas dataframe
    N_ref = ref_list.__len__()
    # N_year=N_year+1
    corn_ref_pd = pd.DataFrame()
    cell_ref_pd = pd.DataFrame()
    bio_facility_pd = pd.DataFrame()
    temp_cap_corn = np.zeros(N_can_loc)
    temp_cap_cell = np.zeros(N_can_loc)
    temp_type_bio_facility = np.zeros(N_can_loc)
    for i in range(N_year):
        temp_loc_corn = np.zeros(N_can_loc)
        temp_loc_cell = np.zeros(N_can_loc)
        temp_loc_bio_facility = np.zeros(N_can_loc)
        for j in range(N_ref):
            if (i >= ref_list[j].Attributes['start_year']) &\
                (ref_list[j].Attributes['start_year'] + ref_list[j].States['production_year'].__len__()-1 > i):
                if ref_list[j].Attributes['tech_type'] == 1:
                    temp_loc_corn[ref_list[j].Attributes['loc_ID']] = 1
                    # temp_his = np.asarray(ref_list[j].States['production_year'])
                    # cap_ID = (temp_his[0:ref_list[j].States['production_year'].__len__()+ i-N_year] ==0).sum()
                    # temp_cap_corn[ref_list[j].Attributes['loc_ID']] += ref_list[j].Attributes['capacity'][cap_ID-1]/10**6
                    temp_cap_corn[ref_list[j].Attributes['loc_ID']] = ref_list[j].Attributes['capacity'][i-ref_list[j].Attributes['start_year']] / 10 ** 6
                elif ref_list[j].Attributes['tech_type'] == 2:
                    temp_loc_cell[ref_list[j].Attributes['loc_ID']] = 1
                    # temp_his = np.asarray(ref_list[j].States['production_year'])
                    # cap_ID = (temp_his[0:ref_list[j].States['production_year'].__len__() + i - N_year] == 0).sum()
                    # temp_cap_cell[ref_list[j].Attributes['loc_ID']] += ref_list[j].Attributes['capacity'][cap_ID - 1]/10**6
                    temp_cap_cell[ref_list[j].Attributes['loc_ID']] += ref_list[j].Attributes['capacity'][i-ref_list[j].Attributes['start_year']] / 10 ** 6
                elif ref_list[j].Attributes['tech_type'] >= 5:
                    temp_loc_bio_facility[ref_list[j].Attributes['loc_ID']] = 1
                    temp_type_bio_facility[ref_list[j].Attributes['loc_ID']] = ref_list[j].Attributes['tech_type']
        temp_cap_corn=look_up_ref_cap(temp_cap_corn, ref_cap_look_up_table)
        temp_cap_cell = look_up_ref_cap(temp_cap_cell, ref_cap_look_up_table)
        corn_ref_pd['corn_loc_year'+str(i)] = temp_loc_corn
        corn_ref_pd['corn_cap_year'+str(i)] = temp_cap_corn
        cell_ref_pd['cell_loc_year'+str(i)] = temp_loc_cell
        cell_ref_pd['cell_cap_year'+str(i)] = temp_cap_cell
        bio_facility_pd['bio_loc_year'+str(i)] = temp_loc_bio_facility
        bio_facility_pd['bio_type_year'+str(i)] = temp_type_bio_facility
    return corn_ref_pd,cell_ref_pd, bio_facility_pd

def draw_gifs(farmer_map,corn_loc_map,cell_loc_map,cmap_farm,cmap_ref_corn,cmap_ref_cell,feature_attributes_farm,feature_attributes_ref_corn,
            loc_attributes_ref_corn,feature_attributes_ref_cell,loc_attributes_ref_cell,patches_farm,patches_ref_corn,patches_ref_cell,prefix):
    # function to generate gif based on geopandas
    N_fig = feature_attributes_farm.__len__()
    ims = []
    for i in range(N_fig):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 10))

        # Loop through each attribute type and plot it using the colors assigned in the dictionary
        for ctype, data in farmer_map.groupby(feature_attributes_farm[i]):
            color = cmap_farm[ctype]
            data.plot(color=color,ax=ax,label=ctype)

        select_loc_corn = corn_loc_map.loc[corn_loc_map[loc_attributes_ref_corn[i]] == 1]
        for ctype, data in select_loc_corn.groupby(feature_attributes_ref_corn[i]):
            color = cmap_ref_corn[ctype]
            data.plot(color=color, ax=ax, label=ctype, markersize=400)

        select_loc_cell = cell_loc_map.loc[cell_loc_map[loc_attributes_ref_cell[i]] == 1]
        for ctype, data in select_loc_cell.groupby(feature_attributes_ref_cell[i]):
            color = cmap_ref_cell[ctype]
            data.plot(color=color, ax=ax, label=ctype, marker="^", markersize=400)

        handle = [ax]
        handle.append(patches_farm[0])
        handle.append(patches_farm[1])
        handle.append(patches_farm[2])
        handle.append(patches_farm[3])
        handle.append(patches_farm[4])
        handle.append(patches_farm[5])

        legend1 = ax.legend(handles=patches_ref_corn, loc='upper center', bbox_to_anchor=(1.035, 0.79), prop={'size': 11},
                            frameon=False, title='Corn Refinery \n Capacity', title_fontsize=14)
        legend2 = ax.legend(handles=patches_ref_cell, loc='upper center', bbox_to_anchor=(1.035, 0.54), prop={'size': 11},
                            frameon=False, title='Cellulosic Refinery \n Capacity', title_fontsize=14)
        ax.legend(handles=handle, loc='upper center', bbox_to_anchor=(1.035, 0.28), prop={'size': 11},
                  frameon=False, title='Land Use', title_fontsize=14)
        ax.add_artist(legend1)
        ax.add_artist(legend2)

        ax.set_title('Land Use Map', fontsize=30)
        ax.annotate('Year '+ str(i+1),
                     xy=(0.5, .15), xycoords='figure fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=24)
        ax.set_axis_off()
        # plt.show()
        # chart = fig.get_figure()
        fig.savefig(
            prefix + feature_attributes_farm[i] + ".jpg",
            dpi=300)
    #     img = Image.open(prefix + feature_attributes_farm[i] + ".jpg")
    #     ims.append(img)
    # # ims.insert(0,ims[0])
    # ims.insert(0, ims[9])
    # ims[1].save(prefix + '.gif', save_all=True, append_images=ims[0:],duration=800)

    img_array = []
    for filename in feature_attributes_farm:
        img = cv2.imread(prefix+filename+'.jpg')
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    img_array.insert(0, img_array[0])
    out = cv2.VideoWriter(prefix+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 0.6, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def draw_gifs_multipatch(farmer_map,marginal_land_map,corn_loc_map,cell_loc_map,bio_loc_map,cmap_farm,cmap_ref_corn,cmap_ref_cell,
                         cmap_bio_facility,feature_attributes_farm,feature_attributes_ref_corn,
                        loc_attributes_ref_corn,feature_attributes_ref_cell,loc_attributes_ref_cell,
                         feature_attributes_bio_facility,loc_attributes_biofacility,patches_farm,patches_ref_corn,patches_ref_cell,
                         patches_bio_facility,land_use_areas,biofuel_production_sum,att_farmer,environ_sensi_farmer,N_release,prefix):
    # function to generate gif based on geopandas
    N_fig = feature_attributes_farm.__len__()
    ims = []
    for i in range(1,N_fig+1):
        plt.close('all')
        # fig, ax = plt.subplots(figsize=(12, 10))
        fig = plt.figure(figsize=(11, 6),dpi=300)
        # fig.set_size_inches(6, 4, forward=False)
        # fig = plt.figure(dpi=300)
        grid = plt.GridSpec(4, 3, figure=fig, hspace=0.2, wspace=0.3)
        ax1 = plt.subplot(grid[:3, :2])
        ax2 = plt.subplot(grid[:2, 2])
        ax3 = plt.subplot(grid[3, 0])
        ax4 = plt.subplot(grid[2:, 2])
        ax5 = plt.subplot(grid[3, 1])

        # Loop through each attribute type and plot it using the colors assigned in the dictionary
        for ctype, data in farmer_map.groupby(feature_attributes_farm[i-1]):
            color = cmap_farm[ctype]
            data.plot(color=color,ax=ax1,label=ctype)

        marginal_land_map.geometry.boundary.plot(color=None, edgecolor='k', linewidth=1.5, ax=ax1)

        select_loc_corn = corn_loc_map.loc[corn_loc_map[loc_attributes_ref_corn[i-1]] == 1]
        for ctype, data in select_loc_corn.groupby(feature_attributes_ref_corn[i-1]):
            color = cmap_ref_corn[ctype]
            data.plot(color=color, ax=ax1, label=ctype, markersize=70)

        select_loc_cell = cell_loc_map.loc[cell_loc_map[loc_attributes_ref_cell[i-1]] == 1]
        for ctype, data in select_loc_cell.groupby(feature_attributes_ref_cell[i-1]):
            color = cmap_ref_cell[ctype]
            data.plot(color=color, ax=ax1, label=ctype, marker="^", markersize=70)

        select_loc_bio = bio_loc_map.loc[bio_loc_map[loc_attributes_biofacility[i-1]] == 1]
        for ctype, data in select_loc_bio.groupby(feature_attributes_bio_facility[i - 1]):
            color = cmap_bio_facility[ctype]
            data.plot(color=color, ax=ax1, label=ctype, marker="D", markersize=15)

        handle = [ax1]
        handle.append(patches_farm[0])
        handle.append(patches_farm[1])
        handle.append(patches_farm[2])
        handle.append(patches_farm[3])
        handle.append(patches_farm[4])
        # handle.append(patches_farm[5])

        legend1 = ax1.legend(handles=patches_ref_corn, loc='upper center', bbox_to_anchor=(.975, 0.68), prop={'size': 8},
                            frameon=False, title='Corn Refinery \n Capacity', title_fontsize=10)
        legend2 = ax1.legend(handles=patches_ref_cell, loc='upper center', bbox_to_anchor=(.975, 0.33), prop={'size': 8},
                            frameon=False, title='Cellulosic Refinery \n Capacity', title_fontsize=10)
        legend3 = ax1.legend(handles=patches_bio_facility, loc='upper center', bbox_to_anchor=(.005, 0.54), prop={'size': 10},
                            frameon=False, title='Biofacility Type', title_fontsize=10)
        ax1.legend(handles=handle, loc='upper center', bbox_to_anchor=(.005, 0.35), prop={'size': 10},
                  frameon=False, title='Land Use', title_fontsize=10)
        ax1.add_artist(legend1)
        ax1.add_artist(legend2)
        ax1.add_artist(legend3)

        # ax1.set_title('Land Use Map', fontsize=10)
        # ax1.annotate('Year '+ str(i+1),
        #              xy=(0.38, .32), xycoords='figure fraction',
        #              horizontalalignment='center', verticalalignment='top',
        #              fontsize=10)
        ax1.set_axis_off()

        ax4.plot(land_use_areas[1:i+1, [0, 1, 2, 6, 7]]/10**6)
        ax4.set_xlim(0, 20)
        ax4.set_ylim(0, 1.6)
        # ax4.legend(['contcorn', 'cornsoy', 'miscanthus', 'fallow', 'CRP'], loc='upper right', fontsize='small',fancybox=True, framealpha=0.5)
        ax4.legend(['corn', 'miscanthus', 'fallow', 'CRP'], loc='upper right', fontsize='large',
                   fancybox=True, framealpha=0.5)
        ax4.set_xlabel('Year', fontsize='large')
        ax4.set_ylabel('Area (Million ha)', fontsize='large')
        ax4.tick_params(labelsize ='small')

        ax3.plot(biofuel_production_sum[:i]/10**9)
        ax3.set_xlim(0, 20)
        ax3.set_ylim(0, 1.1 * biofuel_production_sum.max()/10**9)
        ax3.legend(['Biofuel Production'], fontsize='large', loc='lower right')
        ax3.set_ylabel('Billion L', fontsize='large')
        ax3.set_xlabel('Year', fontsize='large')
        ax3.tick_params(labelsize='small')

        ax5.plot(N_release[:i]/10**6)
        ax5.set_xlim(0, 20)
        ax5.set_ylim(0.6*N_release.max() / 10 ** 6, 1.1 * N_release.max() / 10 ** 6)
        ax5.legend(['Nitrate N load'], fontsize='large', loc='upper right')
        ax5.set_ylabel('Thousand Ton', fontsize='large')
        ax5.set_xlabel('Year', fontsize='large')
        ax5.tick_params(labelsize='small')

        color = 'tab:blue'
        ax2.plot(att_farmer[:i],color=color)
        ax2.set_xlim(0, 20)
        ax2.set_ylim(0, 0.5)
        ax2.set_ylabel("Farmer's willingness \n to adopt Miscanthus", fontsize='large')
        ax2.set_xticks([])
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(labelsize='small')

        ax6 = ax2.twinx()
        color = 'tab:red'
        ax6.plot(environ_sensi_farmer[:i],color=color)
        ax6.set_xlim(0, 20)
        ax6.set_ylim(0, 1.2)
        ax6.tick_params(labelsize='small')
        ax6.set_xticks([])
        ax6.tick_params(axis='y', labelcolor=color)
        # ax5.set_xlabel('Year')
        ax6.set_ylabel("Farmer's attitude \n  toward environment", fontsize='large')
        # plt.show()
        # chart = fig.get_figure()
        fig.savefig(prefix + feature_attributes_farm[i-1] + ".jpg",dpi=100)
    # #     img = Image.open(prefix + feature_attributes_farm[i] + ".jpg")
    # #     ims.append(img)
    # # ims.insert(0,ims[0])
    # # ims.insert(0, ims[9])
    # # ims[1].save(prefix + '.gif', save_all=True, append_images=ims[0:],duration=800)

    img_array = []
    for filename in feature_attributes_farm:
        img = cv2.imread(prefix+filename+'.jpg')
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    img_array.insert(0, img_array[0])
    out = cv2.VideoWriter(prefix+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 0.75, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def draw_gifs_multipatch_commu(farmer_map, marginal_land_map, corn_loc_map, cell_loc_map, cmap_farm, cmap_ref_corn,
                         cmap_ref_cell, feature_attributes_farm, feature_attributes_ref_corn,
                         loc_attributes_ref_corn, feature_attributes_ref_cell, loc_attributes_ref_cell, patches_farm,
                         patches_ref_corn, patches_ref_cell,
                         land_use_areas, biofuel_production_sum, att_farmer, environ_sensi_farmer, N_release,
                         commu_atti, prefix):
    # function to generate gif based on geopandas
    N_fig = feature_attributes_farm.__len__()
    ims = []
    for i in range(1, N_fig+1):
        plt.close('all')
        # fig, ax = plt.subplots(figsize=(12, 10))
        fig = plt.figure(figsize=(14.67, 6), dpi=300)
        # fig.set_size_inches(6, 4, forward=False)
        # fig = plt.figure(dpi=300)
        grid = plt.GridSpec(4, 4, figure=fig, hspace=0.2, wspace=0.2)
        ax1 = plt.subplot(grid[:3, :2])
        ax2 = plt.subplot(grid[:2, 2])
        ax3 = plt.subplot(grid[3, 0])
        ax4 = plt.subplot(grid[2:, 2])
        ax5 = plt.subplot(grid[3, 1])
        ax7 = plt.subplot(grid[:4, 3])

        # Loop through each attribute type and plot it using the colors assigned in the dictionary
        for ctype, data in farmer_map.groupby(feature_attributes_farm[i - 1]):
            color = cmap_farm[ctype]
            data.plot(color=color, ax=ax1, label=ctype)

        marginal_land_map.geometry.boundary.plot(color=None, edgecolor='r', linewidth=0.7, ax=ax1)

        select_loc_corn = corn_loc_map.loc[corn_loc_map[loc_attributes_ref_corn[i - 1]] == 1]
        for ctype, data in select_loc_corn.groupby(feature_attributes_ref_corn[i - 1]):
            color = cmap_ref_corn[ctype]
            data.plot(color=color, ax=ax1, label=ctype, markersize=70)

        select_loc_cell = cell_loc_map.loc[cell_loc_map[loc_attributes_ref_cell[i - 1]] == 1]
        for ctype, data in select_loc_cell.groupby(feature_attributes_ref_cell[i - 1]):
            color = cmap_ref_cell[ctype]
            data.plot(color=color, ax=ax1, label=ctype, marker="^", markersize=70)

        handle = [ax1]
        handle.append(patches_farm[0])
        handle.append(patches_farm[1])
        handle.append(patches_farm[2])
        handle.append(patches_farm[3])
        handle.append(patches_farm[4])
        handle.append(patches_farm[5])

        legend1 = ax1.legend(handles=patches_ref_corn, loc='upper center', bbox_to_anchor=(.975, 0.68),
                             prop={'size': 8},
                             frameon=False, title='Corn Refinery \n Capacity', title_fontsize=8)
        legend2 = ax1.legend(handles=patches_ref_cell, loc='upper center', bbox_to_anchor=(.975, 0.33),
                             prop={'size': 8},
                             frameon=False, title='Cellulosic Refinery \n Capacity', title_fontsize=8)
        ax1.legend(handles=handle, loc='upper center', bbox_to_anchor=(.005, 0.33), prop={'size': 8},
                   frameon=False, title='Land Use', title_fontsize=8)
        ax1.add_artist(legend1)
        ax1.add_artist(legend2)

        # ax1.set_title('Land Use Map', fontsize=10)
        # ax1.annotate('Year '+ str(i+1),
        #              xy=(0.38, .32), xycoords='figure fraction',
        #              horizontalalignment='center', verticalalignment='top',
        #              fontsize=10)
        ax1.set_axis_off()

        ax4.plot(land_use_areas[1:i + 1, [0, 1, 2, 6, 7]] / 10 ** 6)
        ax4.set_xlim(0, 20)
        ax4.set_ylim(0, 1.6)
        # ax4.legend(['contcorn', 'cornsoy', 'miscanthus', 'fallow', 'CRP'], loc='upper right', fontsize='small',
        #            fancybox=True, framealpha=0.5)
        ax4.legend(['corn', 'miscanthus', 'fallow', 'CRP'], loc='upper right', fontsize='small',
                   fancybox=True, framealpha=0.5)
        ax4.set_xlabel('Year', fontsize='small')
        ax4.set_ylabel('Area (Million ha)', fontsize='small')
        ax4.tick_params(labelsize='small')

        ax3.plot(biofuel_production_sum[:i] / 10 ** 9)
        ax3.set_xlim(0, 20)
        ax3.set_ylim(0, 1.1 * biofuel_production_sum.max() / 10 ** 9)
        ax3.legend(['Biofuel Production'], fontsize='small', loc='lower right')
        ax3.set_ylabel('Billion L', fontsize='small')
        ax3.set_xlabel('Year', fontsize='small')
        ax3.tick_params(labelsize='small')

        ax5.plot(N_release[:i] / 10 ** 6)
        ax5.set_xlim(0, 20)
        ax5.set_ylim(0, 1.1 * N_release.max() / 10 ** 6)
        ax5.legend(['Nitrate N load'], fontsize='small', loc='upper right')
        ax5.set_ylabel('Thousand Ton', fontsize='small')
        ax5.set_xlabel('Year', fontsize='small')
        ax5.tick_params(labelsize='small')

        ax7.plot(commu_atti[:i,:])
        ax7.yaxis.set_label_position("right")
        ax7.yaxis.tick_right()
        ax7.set_xlim(0, 20)
        ax7.set_ylim(0, 0.8)
        ax7.legend(['community 1', 'community 2', 'community 3', 'community 4'], fontsize='small', loc='upper left')
        ax7.set_ylabel('Community Environmental Attitude', fontsize='small')
        ax7.set_xlabel('Year', fontsize='small')
        ax7.tick_params(labelsize='small')

        color = 'tab:blue'
        ax2.plot(att_farmer[:i], color=color)
        ax2.set_xlim(0, 20)
        ax2.set_ylim(0.15, 0.35)
        ax2.set_ylabel("Farmer's willingness \n to adopt Miscanthus", fontsize='small')
        ax2.set_xticks([])
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(labelsize='small')

        ax6 = ax2.twinx()
        color = 'tab:red'
        ax6.plot(environ_sensi_farmer[:i], color=color)
        ax6.set_xlim(0, 20)
        ax6.set_ylim(0, 1.05)
        ax6.tick_params(labelsize='small')
        ax6.set_xticks([])
        ax6.tick_params(axis='y', labelcolor=color)
        # ax5.set_xlabel('Year')
        ax6.set_ylabel("Farmer's attitude \n  toward environment", fontsize='small')
        # plt.show()
        # chart = fig.get_figure()
        fig.savefig(prefix + feature_attributes_farm[i - 1] + ".jpg", dpi=100)
    # #     img = Image.open(prefix + feature_attributes_farm[i] + ".jpg")
    # #     ims.append(img)
    # # ims.insert(0,ims[0])
    # # ims.insert(0, ims[9])
    # # ims[1].save(prefix + '.gif', save_all=True, append_images=ims[0:],duration=800)

    img_array = []
    for filename in feature_attributes_farm:
        img = cv2.imread(prefix + filename + '.jpg')
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    img_array.insert(0, img_array[0])
    out = cv2.VideoWriter(prefix + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 2, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()