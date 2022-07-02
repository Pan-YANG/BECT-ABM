# all functions related to physical models
import numpy as np
import  agents
import  gen_agents
import config
import copy
import biorefineries.corn.abm as abm_1
import biorefineries.cornstover.abm as abm_2
import pandas as pd

def divide(divident,denominator):
    denominator += 10**(-100)
    return divident/denominator

def TEA_model_1(operating_days=350.4,plant_capacity=10**8,price_corn=132,price_DDGS=6,
        price_ethanol=0.45,IRR=0.15,duration=(2007, 2027),):
    """
    convert the corn ethanol TEA model from Biosteam to follow the units in ABM
    Return a dictionary of biorefinery metrics for the production of corn ethanol.
    Parameters
    ----------
    operating_days : float
        Number of operating days per year.
    plant_capacity : float
        Plat capacity in L/yr of ethanol.
    price_corn : float
        Price of corn in USD/ton.
    price_DDGS : float
        Price of DDGS in USD/ton.
    price_ethanol : float
        Price of ethanol in USD/ton.
    IRR : float
        Internal rate of return as a fraction (not percent!).
    duration : tuple(int, int)
        Years of operation.
    Returns
    -------
    metrics: dict
        Includes MESP [USD/L], MFPP [USD/ton], IRR [-], NPV [USD],
        TCI [USD], FOC [USD/L], VOC [USD/L], Electricity consumption [MWhr/yr],
        Electricity production [MWhr/yr], and Production [L/yr].
    """
    plant_capacity = plant_capacity/0.382 # convert L/year to kg/year for the Biosteam model
    price_corn = price_corn/1000 # convert $/ton to $/kg
    price_DDGS = price_DDGS/1000 # convert $/ton to $/kg
    price_ethanol = price_ethanol/0.7851 # convert $/L to $/kg

    t=0
    while t<5:
        try:
            TEA_result = abm_1.ABM_TEA_function(operating_days=operating_days,plant_capacity=plant_capacity,
                                                price_corn=price_corn,price_DDGS=price_DDGS,price_ethanol=price_ethanol,
                                                IRR=IRR,duration=duration)
            break
        except RuntimeError:
            t += 1

    production = TEA_result['Production'] / 0.7851 # convert the Biosteam output kg/year to L/year
    feed_cost = plant_capacity * price_corn / production  # the cost of feedstock in ethanol, $/L

    return {
        'MESP': TEA_result['MESP']*0.7851, # convert $/kg to $/L
        'MFPP': TEA_result['MFPP']*1000, # convert $/kg to $/ton
        'IRR': TEA_result['IRR'],
        'NPV': TEA_result['NPV'],
        'TCI': TEA_result['TCI'],
        'VOC': (TEA_result['VOC']/production - feed_cost), # convert $ to $/L
        'FOC': (TEA_result['FOC']/production), # convert $ to $/L
        'Electricity consumption [MWhr/yr]': TEA_result['Electricity consumption [MWhr/yr]'],
        'Electricity production [MWhr/yr]': TEA_result['Electricity production [MWhr/yr]'],
        'Production': production,
    }

def TEA_model_2(cornstover_fraction=1.0, operating_days=350.4, plant_capacity=10**8, price_cornstover=45, price_miscanthus=45,
        price_ethanol=0.45, IRR=0.10, duration=(2007, 2037),):
    """
    convert the cellulosic ethanol TEA model from Biosteam to follow the units in ABM
    Return a dictionary of biorefinery metrics for the production of cellulosic
    ethanol from mixed feedstocks.
    Parameters
    ----------
    cornstover_fraction : float
        Fractino of cornstover in feedstock.
    operating_days : float
        Number of operating days per year.
    plant_capacity : float
        Plat capacity in L/yr of ethanol.
    price_cornstover : float
        Price of cornstover in USD/ton.
    price_miscanthus : float
        Price of miscanthus in USD/ton.
    price_ethanol : float
        Price of ethanol in USD/L.
    IRR : float
        Internal rate of return as a fraction (not percent!).
    duration : tuple(int, int)
        Years of operation.
    Returns
    -------
    metrics: dict
        Includes MESP [USD/L], MFPP [USD/ton], IRR [-], NPV [USD],
        TCI [USD], FOC [USD/L], VOC [USD/L], Electricity consumption [MWhr/yr],
        Electricity production [MWhr/yr], and Production [L/yr].
    """
    plant_capacity = plant_capacity/0.257 # convert L/year to kg/year for the Biosteam model
    price_cornstover = price_cornstover/1000 # convert $/ton to $/kg
    price_miscanthus = price_miscanthus/1000 # convert $/ton to $/kg
    price_ethanol = price_ethanol/0.7851 # convert $/L to $/kg
    t=0
    while t<5:
        try:
            TEA_result = abm_2.ABM_TEA_function(cornstover_fraction,operating_days,plant_capacity,price_cornstover,price_miscanthus,price_ethanol,IRR,duration)
            break
        except:
            print('TEA model error')
            t+=1

    production = TEA_result['Production'] / 0.7851 # convert the Biosteam output kg/year to L/year
    feed_cost = plant_capacity * (price_cornstover * cornstover_fraction + (1-cornstover_fraction) * price_miscanthus) / production  # the cost of feedstock in ethanol, $/L

    return {
        'MESP': TEA_result['MESP']*0.7851, # convert $/kg to $/L
        'MFPP': TEA_result['MFPP']*1000, # convert $/kg to $/ton
        'IRR': TEA_result['IRR'],
        'NPV': TEA_result['NPV'],
        'TCI': TEA_result['TCI'],
        'VOC': (TEA_result['VOC']/production - feed_cost), # convert $ to $/L
        'FOC': (TEA_result['FOC']/production), # convert $ to $/L
        'Electricity consumption [MWhr/yr]': TEA_result['Electricity consumption [MWhr/yr]'],
        'Electricity production [MWhr/yr]': TEA_result['Electricity production [MWhr/yr]'],
        'Production': production,
    }


def identify_adopt_id(adopt_ratio, adopt_priority,patch_areas):
    # function to identify the patch id to adopt perennial energy crop based on Bayesian network
    # adopt ratio: % of land to plant perennial grass
    # adopt priority: the priority of patches to adopt perennial grass
    # patch_areas: array of areas for patches

    adopt_id = []
    i=0
    patch_ratios = patch_areas/patch_areas.sum()
    while adopt_ratio>0:
        adopt_id.append(adopt_priority[i])
        adopt_ratio = adopt_ratio - patch_ratios[adopt_priority[i]]
    return adopt_id

def soil_erosion(base_rate, alpha, beta, slope):
    # function to calculate soil erosion rate
    # base_rate: the erosion rate at 4% slope
    # slope: the slope in 0.0?
    # alpha, beta: parameters in the soil erosion model

    erosion_rate = base_rate * (alpha * slope + beta)/(alpha * 0.04 + beta)
    return erosion_rate

def look_up_table_crop_no_physical_model(patch_ID,is_flood,is_drought,slope,land_use_b,land_use_n,rotation_stage,perennial_age,year,is_stocha):
    # function to calculate the crop yield, fertilizer use, N release for a land patch, assuming there is no physical model
    # patch_ID: the ID of land patch
    # prcp, slope: the precipitation and slope of the land
    # land_use_b, land_use_n: the previous year and current year land uses
    # rotation_stage: 0 for soybean, 1 for corn
    # perennial age: the previous year age of perennial grass
    # config.patch_yield_table: a table showing the yield distribution of each land patch given the climate condition
    # config.fertilizer_table: a table showing the fertilizer application under different land use decision
    # config.perennial_yield_adj_table: a table showing the adjusting factor for perennial grass yields
    # config.soil_erosion_table: a table showing the soil erosion parameters
    # year: the number of year in the simulation
    # is_stocha: 0 for deterministic, 1 for stochastic
    # yield: the yield of crop
    # N_release: the N release
    # ferti_use: the fertilizer use

    # check the look-up table for crop yields and fertilizer applications
    if land_use_n == 1:  # land use: 1 for corn, 2 for soy, 3 for mis, 4 for switch, 5 for sorghum, 6 for cane, 7 for fallow, 8 for CRP
        yield_mean = copy.deepcopy(config.patch_yield_table_mean[patch_ID, 1])
        yield_sto = copy.deepcopy(config.patch_yield_table[year][patch_ID, 1])
        ferti_use = copy.deepcopy(config.fertilizer_table[0])
    elif land_use_n == 2:  # corn soy rotation
        if rotation_stage==0:
            yield_mean = copy.deepcopy(config.patch_yield_table_mean_sub1[patch_ID, 2])
        else:
            yield_mean = copy.deepcopy(config.patch_yield_table_mean_sub2[patch_ID, 2])
        yield_sto = copy.deepcopy(config.patch_yield_table[year][patch_ID, 2])
        ferti_use = copy.deepcopy(config.fertilizer_table[2])
    elif land_use_n == 3:  # mis
        yield_mean = copy.deepcopy(config.patch_yield_table_mean[patch_ID, 3])
        yield_sto = copy.deepcopy(config.patch_yield_table[year][patch_ID, 3])
        if perennial_age <= 0:  # adjust the perennial grass yield and fertilizer use based on the age
            peren_adj = 0
            ferti_use = copy.deepcopy(config.fertilizer_table[3])
        elif perennial_age == 1:
            peren_adj = copy.deepcopy(config.perennial_yield_adj_table[perennial_age, 0])
            ferti_use = copy.deepcopy(config.fertilizer_table[4])
        else:
            peren_adj = copy.deepcopy(config.perennial_yield_adj_table[perennial_age, 0])
            ferti_use = copy.deepcopy(config.fertilizer_table[5])
        yield_mean = yield_mean * peren_adj
        yield_sto = yield_sto * peren_adj
        perennial_age = max(perennial_age,0) + 1  # the perennial age is only updated when there is perennial grass growing

    elif land_use_n == 4:  # switch
        yield_mean = copy.deepcopy(config.patch_yield_table_mean[patch_ID, 4])
        yield_sto = copy.deepcopy(config.patch_yield_table[year][patch_ID, 4])
        if perennial_age <= 0:
            peren_adj = 0
            ferti_use = copy.deepcopy(config.fertilizer_table[6])
        elif perennial_age == 1:
            peren_adj = copy.deepcopy(config.perennial_yield_adj_table[perennial_age, 1])
            ferti_use = copy.deepcopy(config.fertilizer_table[7])
        else:
            peren_adj = copy.deepcopy(config.perennial_yield_adj_table[perennial_age, 1])
            ferti_use = copy.deepcopy(config.fertilizer_table[8])
        yield_mean = yield_mean * peren_adj
        yield_sto = yield_sto * peren_adj
        perennial_age = max(perennial_age, 0) + 1

    elif land_use_n == 5:
        yield_mean = copy.deepcopy(config.patch_yield_table_mean[patch_ID, 5])
        yield_sto = copy.deepcopy(config.patch_yield_table[year][patch_ID, 5])
        ferti_use = copy.deepcopy(config.fertilizer_table[9])
    elif land_use_n == 6:
        yield_mean = copy.deepcopy(config.patch_yield_table_mean[patch_ID, 6])
        yield_sto = copy.deepcopy(config.patch_yield_table[year][patch_ID, 6])
        ferti_use = copy.deepcopy(config.fertilizer_table[10])
    elif land_use_n > 6:
        yield_mean = 0
        yield_sto = 0
        ferti_use = 0

    if is_stocha == 0:
        yield_out = copy.deepcopy(yield_mean)
        patch_N_load_temp = config.patch_N_loads_mean
    else:
        # yield_out = yield_std * np.random.randn()+ yield_mean
        yield_out = copy.deepcopy(yield_sto)
        patch_N_load_temp = config.patch_N_loads[year]

    N_release = patch_N_load_temp[patch_ID,land_use_n-1]
    C_sequest = config.patch_C_sequest[year,patch_ID,land_use_n-1]

    output = {'yield': yield_out, 'N_release': N_release, 'ferti_use':ferti_use,'peren_age':perennial_age, 'C_sequest': C_sequest}
    return output

def cal_PM_for_ref(farmer_list,N_patch,N_crop):
    # function to calculate the expected production of each crop for each crop
    # farmer_list: a list of farmer agents
    # N_patch: total number of land patches considered in the model
    # N_crop: total number of crops
    PM = np.zeros((N_patch,N_crop))

    for i in range(farmer_list.__len__()):
        for j in range(farmer_list[i].Attributes['patch_ID'].__len__()):
            for k in range(N_crop):
                patch_ID = farmer_list[i].Attributes['patch_ID'][j]
                area = farmer_list[i].Attributes['patch_areas'][j]
                slope = farmer_list[i].Attributes['patch_slope'][j]
                land_use_b = farmer_list[i].States['land_use'][-1][j]
                land_use_n = k + 1

                output = look_up_table_crop_no_physical_model(patch_ID,0,0,slope,land_use_b,land_use_n,0,2,1,0)
                yield_best = copy.deepcopy(output['yield'])
                output = look_up_table_crop_no_physical_model(patch_ID,1,0,slope,land_use_b,land_use_n,0,2,1,0)
                yield_flood = copy.deepcopy(output['yield'])
                output = look_up_table_crop_no_physical_model(patch_ID,0,1,slope,land_use_b,land_use_n,0,2,1,0)
                yield_drought = copy.deepcopy(output['yield'])

                PM[patch_ID, k] = area * (config.empirical_risks[0]*yield_flood + config.empirical_risks[1]*yield_drought +
                                  (1-config.empirical_risks[0]-config.empirical_risks[1])*yield_best)
    return PM

def farm_dist_to_patch_dist(farm_dist,farmer_list,N_patch):
    # function to convert an array of refinery distance to farmer to an array of refinery distance to land patch
    # farm_dist: refinery distance to farmer
    # farmer_list: a list of farmers
    # N_patch: the total number of land patches considered in this model
    patch_dist = 10000 * np.ones(N_patch)
    for i in farmer_list:
        patch_ID = copy.deepcopy(i.Attributes['patch_ID'])
        patch_dist[patch_ID] = copy.deepcopy(farm_dist[i.ID])

    return patch_dist

def coarse_reallocate_land_use(initial_LU,target_supply_area):
    # function to quickly reallocate land use to meet refinery feedstock requirement, this is use for community to
    # quick check the expected land use change after refinery investment
    # initial_LU: is the historical land use
    # target_supply_area: the area required to provide enough feedstock for refinery
    temp = target_supply_area > initial_LU # identify the crops that are in short for supplying feedstock
    total_area_required = target_supply_area[temp].sum() - initial_LU[temp].sum()
    total_reallocate = min(initial_LU[~temp].sum(),total_area_required)
    reallocated_LU = copy.deepcopy(initial_LU)
    if total_reallocate == 0:
        pass
    else:
        reallocated_LU[temp] = reallocated_LU[temp] * (1 + divide(total_area_required,reallocated_LU[temp].sum()))
        reallocated_LU[~temp] = reallocated_LU[~temp] * (1 - divide(total_area_required, reallocated_LU[~temp].sum()))
    # proportionally reallocate the land to meet supply
    return reallocated_LU

def cal_delta_LU(ini_LU,new_LU):
    # function to calculate the delta_LU for community willingness calculation
    # ini_LU: the initial land use
    # new_LU: the new land use
    delta_LU = abs(ini_LU - new_LU)
    delta_LU = delta_LU.sum()/2
    return delta_LU

def coarse_cal_PI(land_use_areas,average_slope):
    # function to quickly estimate pollution intensity (PI) based on land use
    PI = 0
    for idx in range(land_use_areas.__len__()):
        PI = PI + soil_erosion(config.soil_erosion_table[0,idx],config.soil_erosion_table[1,idx],
                               config.soil_erosion_table[2, idx],average_slope)
    PI = PI * config.N_in_soil
    return PI

# def coarse_cal_PI_response_matrix(land)

def land_use_to_feed_ID(land_use,tech_type):
    # function to convert land use to feedstock ID
    # land_use: 1 for corn, 2 for soy, 3 for mis, 4 for switch, 5 for sorghum, 6 for cane, 7 for fallow, 8 for CRP
    # tech_type: 1 for corn ethanol, 2 for cellulosic ethanol, 3 for biodiesel, 4 for co-production of biodiesel and ethanol
    #           5 for 5% cofire, 6 for 15% cofire, 7 for BCHP
    # feed_ID: 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus,
    #           4 for switchgrass, 5 for bagasse, 6 for sorghum_oil, 7 for lipidcane_oil, 8 for sorghum_joint, 9 for lipidcane_joint
    if (land_use <= 2) & (tech_type == 1):
        feed_ID = land_use - 1
    elif (land_use == 1) & (tech_type == 2):
        feed_ID = 2
    elif (land_use == 1) & (tech_type == 5):
        feed_ID = 2
    elif (land_use == 1) & (tech_type == 6):
        feed_ID = 2
    elif (land_use == 1) & (tech_type == 7):
        feed_ID = 2
    elif (land_use == 3) | (land_use == 4):
        feed_ID = land_use
    elif (land_use >= 5) & (land_use <= 6) & (tech_type == 3):
        feed_ID = land_use + 1
    elif (land_use >= 5) & (land_use <= 6) & (tech_type == 4):
        feed_ID = land_use + 3
    elif (land_use >= 7) & (land_use <= 8):
        feed_ID = 5
    else:
        feed_ID = float('nan')
        # print('++++++++++++++++++++++++++++++++++++++++LAND USE IS' + str(int(land_use)) + '++++++++++++++++++++++++++++++++++++++++')
        # print('++++++++++++++++++++++++++++++++++++++++TECH TYPE IS' + str(int(tech_type)) + '++++++++++++++++++++++++++++++++++++++++')
    return feed_ID

def feed_ID_to_land_use(feed_ID):
    # function to convert the feedstock ID to land use
    # feed_ID: 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus,
    #           4 for switchgrass, 5 for bagasse, 6 for sorghum_oil, 7 for lipidcane_oil, 8 for sorghum_joint, 9 for lipidcane_joint
    # land_use: 1 for corn, 2 for soy, 3 for mis, 4 for switch, 5 for sorghum, 6 for cane, 7 for fallow, 8 for CRP
    if feed_ID <= 1:
        land_use = feed_ID + 1
    elif feed_ID == 2:
        land_use = 1
    elif (feed_ID >= 3) & (feed_ID <= 4):
        land_use = copy.deepcopy(feed_ID)
    elif feed_ID == 5:
        land_use = float('nan')
    elif (feed_ID >= 6) & (feed_ID <= 7):
        land_use = feed_ID - 1
    elif (feed_ID >= 8) & (feed_ID <= 9):
        land_use = feed_ID - 3
    else:
        land_use = float('nan')
    return land_use

# def quick_cal_ref_WU(can_ref_agent):
#     # function to quickly calculate the water use of one candidate refinery
#     if can_ref_agent.Attributes['refinery_type'] <= 2:
#         fuel_ID = 1
#     else:
#         fuel_ID = 0
#     feed_demands = can_ref_agent.Attributes['capacity']/config.refinery_product_yield_table[:,fuel_ID-1]
#     WU_list = feed_demands * config.refinery_water_use_table
#     WU = WU_list.max()
#     return WU

def cal_occupied_feed(farmer_list,ref_list):
    # function to calculate the feedstocks that are already occupied by the existing refineries
    # N_can_loc: total number of candidate refinery locations
    # ref_list: the existing refineries
    # farmer_list: the list of farmers

    # calculate total feedstock supplied by the whole watershed
    total_feed = np.zeros(10)
    for i in range(farmer_list.__len__()):
        N_patch = farmer_list[i].Attributes['patch_ID'].__len__()
        for j in range(N_patch):
            if farmer_list[i].States['land_use'][-1][j] == 1:
                total_feed[0] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
                total_feed[2] += config.stover_harvest_ratio * farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
            elif farmer_list[i].States['land_use'][-1][j] == 2:
                total_feed[1] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
            elif farmer_list[i].States['land_use'][-1][j] == 3:
                total_feed[3] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
            elif farmer_list[i].States['land_use'][-1][j] == 4:
                total_feed[4] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
            elif farmer_list[i].States['land_use'][-1][j] == 5:
                total_feed[6] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
                total_feed[7] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
            elif farmer_list[i].States['land_use'][-1][j] == 6:
                total_feed[8] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]
                total_feed[9] += farmer_list[i].States['yield'][-1][j] * farmer_list[i].Attributes['patch_areas'][j]

    occupied_feed = [] # N_can_loc elements, each representing the feedstock occupied by existing refineries for each candidate location
    for i in range(config.N_can_ref_locs + config.N_can_BCHP + config.N_can_cofire):
        if i < config.N_can_ref_locs: # for biorefineries
            ref_farm_dist_matrix = config.ref_farmer_dist_matrix[i*config.N_can_ref_per_loc,:]
        elif i<config.N_can_ref_locs + config.N_can_BCHP: # for BCHP
            ref_farm_dist_matrix = config.BCHP_farmer_dist_matrix[i-config.N_can_ref_locs, :]
        else:
            ref_farm_dist_matrix = config.cofire_farmer_dist_matrix[i-config.N_can_ref_locs - config.N_can_BCHP, :]
        farmer_with_boundary = np.argwhere(ref_farm_dist_matrix <= config.patch_influence_range).flatten()
        # first identify the total feedstock supplied by the influence region of a candidate location
        feed_with_boundary = np.zeros(10)
        for k in range(farmer_with_boundary.size):
            N_patch = farmer_list[farmer_with_boundary[k]].Attributes['patch_ID'].__len__()
            for kk in range(N_patch):
                if farmer_list[farmer_with_boundary[k]].States['land_use'][-1][kk] == 1:
                    feed_with_boundary[0] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                    feed_with_boundary[2] += config.stover_harvest_ratio * farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                elif farmer_list[farmer_with_boundary[k]].States['land_use'][-1][kk] == 2:
                    feed_with_boundary[1] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                elif farmer_list[farmer_with_boundary[k]].States['land_use'][-1][kk] == 3:
                    feed_with_boundary[3] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                elif farmer_list[farmer_with_boundary[k]].States['land_use'][-1][kk] == 4:
                    feed_with_boundary[4] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                elif farmer_list[farmer_with_boundary[k]].States['land_use'][-1][kk] == 5:
                    feed_with_boundary[6] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                    feed_with_boundary[7] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                elif farmer_list[farmer_with_boundary[k]].States['land_use'][-1][kk] == 6:
                    feed_with_boundary[8] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]
                    feed_with_boundary[9] += farmer_list[farmer_with_boundary[k]].States['yield'][-1][kk] * farmer_list[farmer_with_boundary[k]].Attributes['patch_areas'][kk]

        contracted_feed = np.zeros(10)
        purchased_feed = np.zeros(10)
        for j in range(ref_list.__len__()):
            contracted_farmer_ID = ref_list[j].States['contracted_farmer_ID'][-1]
            if contracted_farmer_ID.size==0:
                pass
            else:
                contracted_farmer_ID_within_boundary = ref_farm_dist_matrix[contracted_farmer_ID] <= config.patch_influence_range
                if contracted_farmer_ID_within_boundary.sum() == 0:
                    pass
                elif contracted_farmer_ID_within_boundary.sum() == 1:
                    contracted_feed += ref_list[j].States['contracted_patch_amount'][-1][contracted_farmer_ID_within_boundary, :].flatten()
                else:
                    contracted_feed += ref_list[j].States['contracted_patch_amount'][-1][contracted_farmer_ID_within_boundary,:].sum(0)
            if ref_list[j].States['purchased_feedstock'].__len__()==0:
                pass
            else:
                purchased_feed += ref_list[j].States['purchased_feedstock'][-1][0].sum(0) * (feed_with_boundary/(total_feed+0.01))

        occupied_feed.append(contracted_feed + purchased_feed)
    return occupied_feed


def quick_cal_ref_feed_amount(can_ref_agent,farmer_list,bagasse_avaiable,delta_LU_limit,occupied_feed):
    # function to quickly calculate the feedstock amount required by each refinery
    # bagasse_avaiable: the available bagasse in the market
    # delta_LU_limit: the portion of maximum land use change
    # occupied_feed: as list showing the feedstocks already occupied for each of the candidate refinery location

    loc_ID = can_ref_agent.Attributes['loc_ID']
    occupied_feed_loc = copy.deepcopy(occupied_feed[loc_ID])
    dist_matrix = copy.deepcopy(can_ref_agent.Attributes['dist_farmer'])
    tech_type = copy.deepcopy(can_ref_agent.Attributes['refinery_type'])
    ID_within_range = np.argwhere(dist_matrix <= config.patch_influence_range).flatten()
    feed_available = np.zeros(10)
    land_use_areas = np.zeros(8)
    dist_accu = np.zeros(10)
    for ind in ID_within_range:
        farmer_agent = farmer_list[ind]
        dist_temp = dist_matrix[ind]
        N_patch = farmer_agent.Attributes['patch_ID'].__len__()
        for i in range(N_patch):
            if farmer_agent.States['contract'][-1][i]>0:
                continue
            area = copy.deepcopy(farmer_agent.Attributes['patch_areas'][i])
            land_use = int(farmer_agent.States['land_use'][-1][i])
            land_use_areas[land_use-1] = land_use_areas[land_use-1] + area
            feed_ID = land_use_to_feed_ID(land_use,can_ref_agent.Attributes['refinery_type'])
            if np.isnan(feed_ID):
                continue
            elif land_use_tech_type_match(can_ref_agent.Attributes['refinery_type'],land_use)==0:
                continue
            else:
                if feed_ID == 2: # if feedstock is corn stover, use the stover harvest ratio to adjust corn yield
                    crop_yield = config.stover_harvest_ratio * farmer_agent.States['yield'][-1][i]
                else:
                    crop_yield = copy.deepcopy(farmer_agent.States['yield'][-1][i])
                feed_available[feed_ID] = feed_available[feed_ID] + area * crop_yield
                dist_accu[feed_ID] += area * crop_yield * dist_temp
    feed_available[5] = copy.deepcopy(bagasse_avaiable)
    feed_available_negative = np.argwhere(feed_available<=0).flatten()
    occupied_feed_loc[feed_available_negative]=0 # identify the feedstocks that is not in need and assign 0 to their occupation
    if tech_type < 5:# the occupied feedstock should be discounted in estimating biofuel production
        productions = np.dot(feed_available-occupied_feed_loc, config.refinery_product_yield_table)
    else:
        productions = np.dot((feed_available-occupied_feed_loc)[2:6].sum(), config.biofacility_product_yield_table[tech_type - 5, :])

    if (max(productions[0:2]) >= can_ref_agent.Attributes['capacity'])|(productions[5] > can_ref_agent.Attributes['capacity']): # if the current available feedstock can support the refinery, use the current ones
        feed_stock_enough = 1
        if tech_type<5:
            feed_amount = (feed_available-occupied_feed_loc) * can_ref_agent.Attributes['capacity'] / max(productions[0:2])
        else:
            feed_amount = (feed_available - occupied_feed_loc) * can_ref_agent.Attributes['capacity'] / productions[5]
        feed_amount = feed_amount*(feed_amount>0)+0 # an easy fix for rare cases when feed available < occupied feed
        aver_dist = divide(dist_accu,feed_available)
    else: # otherwise, the refinery calculates if changing delta_LU_limit% if all other land use to produce the preferred feedstock could support the proposed capacity
        land_use_areas_adj = copy.deepcopy(land_use_areas) # calculate the potential maximum land use change to provide feedstock
        if tech_type == 1:
            land_use_areas_adj[1:] = land_use_areas[1:] * (1-delta_LU_limit) # assuming that up to 20% of all other land use are changed to the primary feedstocks
            land_use_areas_adj[0] = land_use_areas[0] + land_use_areas[1:].sum() - land_use_areas_adj[1:].sum()
        elif (tech_type == 2)|(tech_type == 5)|(tech_type == 6)|(tech_type == 7):
            land_use_areas_adj[[0,1,4,5,6,7]] = land_use_areas[[0,1,4,5,6,7]] * (1-delta_LU_limit)
            land_use_areas_adj[[2,3]] = divide(land_use_areas_adj[[2,3]] * (land_use_areas[[0,1,4,5,6,7]].sum() - land_use_areas_adj[[0,1,4,5,6,7]].sum()),
                                        land_use_areas_adj[[2,3]].sum())
        elif tech_type > 2:
            land_use_areas_adj[[0, 1, 2, 3, 6, 7]] = land_use_areas[[0, 1, 2, 3, 6, 7]] * (1-delta_LU_limit)
            land_use_areas_adj[[4, 5]] = divide(land_use_areas_adj[[4, 5]] * (land_use_areas[[0, 1, 2, 3, 6, 7]].sum() - land_use_areas_adj[[0, 1, 2, 3, 6, 7]].sum()),
                                         land_use_areas_adj[[4, 5]].sum())
        feed_available_adj = np.zeros(10)
        for k in range(8):
            feed_ID = land_use_to_feed_ID(k+1, tech_type)
            if np.isnan(feed_ID):
                continue
            elif land_use_tech_type_match(tech_type, k+1) == 0:
                continue
            else:
                feed_available_adj[feed_ID] = divide(feed_available[feed_ID] * land_use_areas_adj[k],land_use_areas[k]) # adjust the available feedstock based on the maximum land use change

        if tech_type < 5:  # the occupied feedstock should be discounted in estimating biofuel production
            productions_adj = np.dot(feed_available_adj-occupied_feed_loc, config.refinery_product_yield_table)
        else:
            productions_adj = np.dot((feed_available_adj - occupied_feed_loc)[2:6].sum(),
                                 config.biofacility_product_yield_table[tech_type - 5, :])

        if (max(productions_adj[0:2]) >= can_ref_agent.Attributes['capacity']/(1-config.mis_trans_loss-config.mis_storage_loss))|\
                (productions_adj[5] > can_ref_agent.Attributes['capacity']/(1-config.mis_trans_loss-config.mis_storage_loss)):
            feed_stock_enough = 1
            if tech_type < 5:
                feed_amount = (feed_available_adj-occupied_feed_loc) * can_ref_agent.Attributes['capacity'] / max(productions_adj[0:2])
            else:
                feed_amount = (feed_available_adj - occupied_feed_loc) * can_ref_agent.Attributes['capacity'] / productions[5]
            feed_amount /= 1-config.mis_trans_loss-config.mis_storage_loss
            if feed_amount[2]<0:
                feed_amount[3] = feed_amount[3] + feed_amount[2]
                feed_amount[2] = 0
            elif feed_amount[3]<0:
                feed_amount[2] = feed_amount[2] + feed_amount[3]
                feed_amount[3] = 0
            # dist_accu = (feed_available - occupied_feed_loc).sum() * (dist_accu.sum()/(feed_available.sum()+0.001)) + \
            #             (feed_amount - feed_available + occupied_feed_loc).sum()*config.patch_influence_range
            dist_accu = feed_amount * config.patch_influence_range
            aver_dist = divide(dist_accu,feed_amount)
        else:
            feed_stock_enough = np.asarray([0.])
            feed_amount = np.asarray([0.])
            feed_amount /= 1 - config.mis_trans_loss - config.mis_storage_loss
            aver_dist = np.asarray([config.patch_influence_range])

    for i in range(feed_amount.size):
        if feed_amount[i] == 0:
            aver_dist[i] =0
    feed_available = feed_available - occupied_feed_loc
    if feed_available[2] < 0:
        feed_available[3] = feed_available[3] + feed_available[2]
        feed_available[2] = 0
    elif feed_available[3] < 0:
        feed_available[2] = feed_available[2] + feed_available[3]
        feed_available[3] = 0
    return feed_amount, feed_stock_enough, feed_available, aver_dist

def initiate_farmer_new_time_step(farmer_list,stover_price):
    # function to initiate all the farmer contract information to the new year
    for farm_agent in farmer_list:
        # farm_agent.States['contract'].append(farm_agent.Attributes['contract'][-1])
        farm_agent.Temp['contract_land_use'] = -999 * np.ones(farm_agent.Attributes['patch_ID'].__len__())
        farm_agent.Temp['patch_received_prices'] = np.zeros(farm_agent.Attributes['patch_ID'].__len__())
        farm_agent.Temp['patch_available_for_sale'] = np.ones(farm_agent.Attributes['patch_ID'].__len__())
        if stover_price > config.stover_harvest_cost:
            farm_agent.Temp['stover_available_for_sale'] = np.ones(farm_agent.Attributes['patch_ID'].__len__())
        else:
            farm_agent.Temp['stover_available_for_sale'] = np.zeros(farm_agent.Attributes['patch_ID'].__len__())
        farm_agent.Temp['peren_age_refresh'] = np.zeros(farm_agent.Attributes['patch_ID'].__len__(),int)
        farm_agent.Temp['already_negeotiated'] = 0
        farm_agent.cal_peer_ec(farmer_list)

def initiate_community_new_time_step(community_list):
    # function to initiate all the community Temp information for the new year
    for community_agent in community_list:
        community_agent.Temp['WU'] = [0]

def initiate_ref_new_time_step(ref_list):
    # function to initiate all the community Temp information for the new year
    for ref_agent in ref_list:
        ref_agent.Temp['purchased_feedstock'] = np.zeros((1, 10))

def check_feed_demand(ref_list,ref_type):
    # function to check the total demand of certain feedstock
    demand = 0
    for ref_agent in ref_list:
        if (ref_type == 1) & (ref_agent.Attributes['tech_type'] == 1):
            # temp_demands = ref_agent.Attributes['feedstock_amount'] - ref_agent.States['contracted_patch_amount'][-1].sum(0)
            # temp_demands[5] = 0  # bagasse is not considered in the contract demand
            # prod_amount = np.dot(ref_agent.Attributes['feedstock_amount'], config.refinery_product_yield_table)[0:2]
            # main_product = np.argmax(prod_amount)
            # temp_demands = temp_demands * (config.refinery_product_yield_table[:, main_product]/config.refinery_product_yield_table[0, main_product])
            # demand = demand + temp_demands.sum()
            continue
        elif (ref_type == 2) & ((ref_agent.Attributes['tech_type'] == np.asarray([2,5,6,7])).sum()>0):
            temp_demands = ref_agent.Attributes['feedstock_amount'] - ref_agent.States['contracted_patch_amount'][-1].sum(0)
            temp_demands[5] = 0  # bagasse is not considered in the contract demand
            prod_amount = np.dot(ref_agent.Attributes['feedstock_amount'], config.refinery_product_yield_table)[0:2]
            main_product = np.argmax(prod_amount)
            temp_demands = temp_demands * (config.refinery_product_yield_table[:, main_product] / config.refinery_product_yield_table[3, main_product]) # use miscanthus as the benchmark
            demand = demand + temp_demands.sum()
        elif (ref_type >=3) & ((ref_agent.Attributes['tech_type'] == np.asarray([3,4])).sum()>0):
            temp_demands = ref_agent.Attributes['feedstock_amount'] - ref_agent.States['contracted_patch_amount'][-1].sum(0)
            temp_demands[5] = 0  # bagasse is not considered in the contract demand
            prod_amount = np.dot(ref_agent.Attributes['feedstock_amount'], config.refinery_product_yield_table)[0:2]
            main_product = np.argmax(prod_amount)
            if ref_agent.Attributes['tech_type'] == 3:
                benchmark_feed_ID = 6
            else:
                benchmark_feed_ID = 8
            temp_demands = temp_demands * (config.refinery_product_yield_table[:, main_product] / config.refinery_product_yield_table[benchmark_feed_ID, main_product])
            demand += temp_demands.sum()
            demand /= 1 - config.mis_storage_loss - config.mis_trans_loss

    return demand

def convert_demand_array_to_single_number(demand_array,tech_type):
    # function to convert an array of demands in to one single number of demand
    if tech_type == 1:
        demand = copy.deepcopy(demand_array[0])
    elif tech_type == 2:
        demand = (demand_array[2]* config.refinery_product_yield_table[2, 0] +
                  demand_array[3]* config.refinery_product_yield_table[3, 0] +
                  demand_array[4]* config.refinery_product_yield_table[4, 0]) /config.refinery_product_yield_table[3, 0]
    elif tech_type == 3:
        demand = (demand_array[6]* config.refinery_product_yield_table[6, 1] +
                  demand_array[7]* config.refinery_product_yield_table[7, 1]) /config.refinery_product_yield_table[6, 1]
    elif tech_type == 4:
        demand = (demand_array[6] * config.refinery_product_yield_table[8, 1] +
                  demand_array[7] * config.refinery_product_yield_table[9, 1]) / config.refinery_product_yield_table[8, 1]
    elif (tech_type == np.asarray([5,6,7])).sum() > 0:
        demand = demand_array[2] + demand_array[3] + demand_array[4]
    else:
        demand = 0
    return demand

def convert_production_to_supply(land_use,crop_yield,area,tech_type):
    prod_amount = crop_yield * area
    feed_ID = land_use_to_feed_ID(land_use, tech_type)
    if tech_type == 1:
        supply = copy.deepcopy(prod_amount)
    elif np.isin(tech_type,[2,5,6,7]):
        supply = prod_amount * config.refinery_product_yield_table[feed_ID, 0] / config.refinery_product_yield_table[3, 0]
    elif tech_type == 3:
        supply = prod_amount * config.refinery_product_yield_table[feed_ID, 0] / \
                 config.refinery_product_yield_table[6, 0]
    elif tech_type == 4:
        supply = prod_amount * config.refinery_product_yield_table[feed_ID, 0] / \
                 config.refinery_product_yield_table[8, 1]
    return supply, feed_ID

def land_use_tech_type_match(ref_type,land_use):
    # function to identify if the refinery tech type matches with the land use
    if (ref_type == 1) & (land_use == 1):
        is_match = 1
    elif (ref_type ==2) & ((land_use == np.asarray([1,3,4])).sum()>0):
        is_match = 1
    elif (ref_type >=3) & ((land_use == np.asarray([5,6])).sum()>0):
        is_match = 1
    elif (ref_type ==5) & ((land_use == np.asarray([1,3,4])).sum()>0):
        is_match = 1
    elif (ref_type == 6) & ((land_use == np.asarray([1, 3, 4])).sum() > 0):
        is_match = 1
    elif (ref_type ==7) & ((land_use == np.asarray([1,3,4])).sum()>0):
        is_match = 1
    else:
        is_match =0
    return is_match

def feed_stock_tech_type_match(feed_type,ref_type):
    # function to identify if the feedstock type and refinery type is matched
    if (ref_type==1) & (feed_type==0):
        is_match = 1
    elif ((ref_type == np.asarray([2,5,6,7])).sum()>0) & ((feed_type == np.asarray([2,3,4,5])).sum()>0):
        is_match = 1
    elif (ref_type==3) & ((feed_type == np.asarray([6,7])).sum()>0):
        is_match = 1
    elif (ref_type==4) & ((feed_type == np.asarray([8,9])).sum()>0):
        is_match = 1
    else:
        is_match = 0
    return is_match

def check_feed_supply(farmer_list,feed_prices,ref_type):
    # function to check the total supply of certain feedstock, corn refinery is not considered
    supply = 0
    for farmer_agent in farmer_list:
        risks = farmer_agent.States['climate_forecasts'][-1]
        is_flood = copy.deepcopy(risks[0])
        is_drought = copy.deepcopy(risks[1])
        land_use = copy.deepcopy(farmer_agent.States['land_use'][-1])
        contract = copy.deepcopy(farmer_agent.States['contract'][-1])
        N_patch = land_use.__len__()
        for i in range(N_patch):
            if contract[i] == 1:
                continue
            elif land_use_tech_type_match(ref_type,land_use[i]) == 1:
                output = look_up_table_crop_no_physical_model(farmer_agent.Attributes['patch_ID'][i], is_flood,
                                                                  is_drought, farmer_agent.Attributes['patch_slope'][i],
                                                                  farmer_agent.States['land_use'][-1][i],
                                                                  land_use[i],0,
                                                                  farmer_agent.States['peren_age'][-1][i], 1, 0)
                crop_amount_patch = farmer_agent.Attributes['patch_areas'][i] * output['yield'] # corn refinery does not make contract
                if np.isin(ref_type,[2,5,6,7]) & (land_use[i] == 1) & (feed_prices[2]>=config.stover_harvest_cost):
                    supply = supply + crop_amount_patch * config.stover_harvest_ratio * config.refinery_product_yield_table[2,0]/config.refinery_product_yield_table[3,0]
                elif np.isin(ref_type,[2,5,6,7]):
                    supply = supply + crop_amount_patch * config.refinery_product_yield_table[land_use[i], 0] / config.refinery_product_yield_table[3, 0]
                elif ref_type == 3:
                    supply = supply + crop_amount_patch * config.refinery_product_yield_table[land_use[i] + 1, 1] / config.refinery_product_yield_table[6, 1]
                elif ref_type == 4:
                    supply = supply + crop_amount_patch * config.refinery_product_yield_table[land_use[i] + 3, 1] / config.refinery_product_yield_table[8, 1]

    return supply

def cal_contract_length(land_uses):
    # function to calculate the length of contract for each land use
    land_uses = np.atleast_1d(land_uses)
    N = land_uses.size
    contract_length = np.zeros(N,int)
    for i in range(N):
        if (land_uses[i] == np.asarray([1,2])).sum()>0:
            contract_length[i] = 1
        elif (land_uses[i] == np.asarray([3])).sum()>0:
            contract_length[i] = 30
        elif (land_uses[i] == np.asarray([1,4])).sum()>0:
            contract_length[i] = 30
        elif (land_uses[i] == np.asarray([1,5])).sum()>0:
            contract_length[i] = 1
        elif (land_uses[i] == np.asarray([1,6])).sum()>0:
            contract_length[i] = 1
        else:
            contract_length[i] = 0
    return contract_length

def collecting_farmer_attitude(farmer_list):
    # function to compile of all farmers' attitudes for calculating their influence to each other
    N_farmer = farmer_list.__len__()
    AT = np.zeros(N_farmer)
    for i in range(N_farmer):
        AT[i] = copy.deepcopy(farmer_list[i].Temp['attitude'])
    return AT

def cal_flood_drought(Prcp):
    # function to identify if the land is being flooded

    P_thre = [1200,200]
    flood = Prcp >= P_thre[0] # all prcp in mm/yr
    drought = Prcp <= P_thre[1]

    return flood, drought

def cal_contracted_feedstock_amount(farmer_list,ref_list):
    # function to calculate the actual amount of feedstock from contracts for each refinery
    N_ref = ref_list.__len__()
    N_farmer = farmer_list.__len__()
    contracted_feedstock_amount = np.zeros((N_ref,10))
    for i in range(N_ref):
        contracted_farmer_ID = copy.deepcopy(ref_list[i].States['contracted_farmer_ID'][-1].astype(int))
        contracted_patch_ID = copy.deepcopy(ref_list[i].States['contracted_patch_ID'][-1].astype(int))
        contracted_patch_price = copy.deepcopy(ref_list[i].States['contracted_patch_price'][-1])
        N_contracts = contracted_patch_ID.__len__()
        contracted_patch_supply = np.zeros((N_contracts,10))
        for j in range(N_contracts):
            temp_patch_ID = np.argwhere(farmer_list[contracted_farmer_ID[j]].Attributes['patch_ID'] == contracted_patch_ID[j])[0][0]
            land_use = copy.deepcopy(farmer_list[contracted_farmer_ID[j]].States['land_use'][-1][temp_patch_ID])
            crop_yield = copy.deepcopy(farmer_list[contracted_farmer_ID[j]].States['yield'][-1][temp_patch_ID])
            patch_area = copy.deepcopy(farmer_list[contracted_farmer_ID[j]].Attributes['patch_areas'][temp_patch_ID])
            contract_price = copy.deepcopy(contracted_patch_price[j])
            farmer_list[contracted_farmer_ID[j]].Temp['patch_received_prices'][temp_patch_ID] = copy.deepcopy(contract_price)
            feed_type = land_use_to_feed_ID(land_use, ref_list[i].Attributes['tech_type'])
            contracted_patch_supply[j,feed_type] = crop_yield * patch_area
        ref_list[i].States['contracted_patch_supply'].append(contracted_patch_supply)
        contracted_feedstock_amount[i,:] = contracted_patch_supply.sum(0)
    return contracted_feedstock_amount

def cal_ref_production_amount(feedstock_amount,tech_type):
    # function to quickly calculate the amount of biofuel production based on the feedstock amount
    if tech_type < 5:
        production = max(np.dot(feedstock_amount, config.refinery_product_yield_table)[0:2])
    else:
        production = np.dot(feedstock_amount[2:6].sum(), config.biofacility_product_yield_table[tech_type-5,:])[5]
    return production

def change_farmer_type(farmer_list,non_type_I_farmer,prob):
    # function to change the farmer type during simulation
    # non_type_I_farmer: a database to store the currently non-type I farmers
    # prob: the probability of a non-type I farmer to convert to type I farmer
    agent_cluster_ABM = pd.read_excel('C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/BN_rule/agent_cluster_ABM.xlsx')
    database_cluster_ID = agent_cluster_ABM['cluster']
    type_I_farmer_database = agent_cluster_ABM[database_cluster_ID==1]

    convert_farmers = non_type_I_farmer.sample(frac=prob, replace=False)
    non_type_I_farmer = non_type_I_farmer.drop(convert_farmers.index)
    convert_farmer_IDs = convert_farmers[0].to_numpy()
    convert_farmer_IDs = convert_farmer_IDs.astype(int)

    for i in convert_farmer_IDs:
        target_attributes = type_I_farmer_database.sample(1)
        farmer_list[i].Attributes['info_use'] = target_attributes['info_use'].to_numpy()[0]
        farmer_list[i].Attributes['benefit'] = target_attributes['benefit'].to_numpy()[0]
        farmer_list[i].Attributes['concern'] = target_attributes['concern'].to_numpy()[0]
        farmer_list[i].Attributes['lql'] = target_attributes['lql'].to_numpy()[0]
        farmer_list[i].Attributes['type'] = target_attributes['cluster'].to_numpy()[0]
        farmer_list[i].States['max_fam'][-1] = target_attributes['max_fam'].to_numpy()[0]

    return farmer_list, non_type_I_farmer


