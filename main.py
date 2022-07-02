import numpy as np
import config
import government as gov
import general_methods_physical as GMP
import general_methods_economic as GME
import gen_agents
import copy
import imp
import os
from mortgage import Loan
import parameters as params
import pandas as pd


# dir_loc = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline'

def main():
    os.chdir(params.folder)
    # generate agents
    # farmer
    N_farmer = 1000
    patch_attribute = np.loadtxt('./data/patch_attributes.csv', delimiter=',', skiprows=1)
    farmer_attribute = np.loadtxt('./data/farmer_attributes.csv', delimiter=',', skiprows=1)
    patch_ini_state = np.loadtxt('./data/patch_ini_state.csv', delimiter=',', skiprows=1)
    farmer_ini_state = np.loadtxt('./data/farmer_ini_state.csv', delimiter=',', skiprows=1)
    adjacency_matrix = np.loadtxt('./data/adjacency_matrix.csv', delimiter=',', skiprows=0)
    neighbor_influence_matrix = np.loadtxt('./data/neighbor_influence_matrix.csv', delimiter=',', skiprows=0)

    farmer_agent_list = gen_agents.generate_farmer(N_farmer, patch_attribute, farmer_attribute, patch_ini_state,
                                                   farmer_ini_state, adjacency_matrix, neighbor_influence_matrix)

    non_type_I_farmer = farmer_attribute[farmer_attribute[:, 13] > 1, :]
    non_type_I_farmer = pd.DataFrame(non_type_I_farmer)
    # i=0
    # while i<farmer_agent_list.__len__():
    #     if farmer_agent_list[i].Attributes['patch_ID'].size == 0:
    #         farmer_agent_list.pop(i)
    #     else:
    #         i += 1
    # N_farmer = farmer_agent_list.__len__()

    # community
    N_community = 4
    community_attribute = config.community_attribute
    community_ini_state = np.loadtxt('./data/community_ini_states.csv', delimiter=',', skiprows=1)

    community_list = gen_agents.gen_community(N_community, community_attribute, community_ini_state)
    for i in range(community_list.__len__()):
        community_list[i].ini_community(farmer_agent_list)

    # government
    N_gov = 1
    gov_agent = gen_agents.gen_gov(N_gov, gov.RFS_volume, gov.TMDL, gov.slope_limits, gov.TMDL_N_limits,
                                   gov.scaling_factor)
    gov_agent = gov_agent[0]

    # consumer
    N_consumer = 1
    consumer_agent = gen_agents.gen_comsumer(N_consumer, config.ini_WP, config.IRW, config.max_WP,
                                             config.ini_ethanol_price)
    consumer_agent = consumer_agent[0]

    # refinery
    N_can_ref = 60
    can_refinery_attribute = config.can_refinery_attribute
    ref_farmer_dist_matrix = config.ref_farmer_dist_matrix

    N_can_BCHP = config.N_can_BCHP
    can_BCHP_attribute = np.loadtxt('./data/can_BCHP_attributes.csv', delimiter=',', skiprows=1)
    BCHP_farmer_dist_matrix = config.BCHP_farmer_dist_matrix
    can_BCHP_list = gen_agents.gen_candidate_refinery(N_can_BCHP, can_BCHP_attribute,
                                                      BCHP_farmer_dist_matrix)  # initiate candidate BCHP every year
    can_BCHP_IDs = np.asarray(range(N_can_BCHP))

    N_can_cofire = config.N_can_cofire
    can_cofire_attribute = np.loadtxt('./data/can_cofire_attributes.csv', delimiter=',', skiprows=1)
    cofire_farmer_dist_matrix = config.cofire_farmer_dist_matrix
    can_cofire_list = gen_agents.gen_candidate_refinery(N_can_cofire, can_cofire_attribute,
                                                        cofire_farmer_dist_matrix)  # initiate candidate BCHP every year
    can_cofire_IDs = np.asarray(range(N_can_cofire))

    # initial corn refineries
    N_ref = 3
    refinery_attribute = np.loadtxt('./data/ini_ref_attributes.csv', delimiter=',', skiprows=1)
    for i in range(N_ref):
        TEA_result = GMP.TEA_model_1(plant_capacity=refinery_attribute[i, 12],
                                     price_corn=config.external_provided_prices[0, 7],
                                     price_DDGS=config.external_provided_prices[0, 3],
                                     price_ethanol=config.external_provided_prices[0, 0])
        refinery_attribute[i, 12] = TEA_result[
            'Production']  # refinery capacity in L/year, calculated from BioSTEAM model
        refinery_attribute[i, 14] = TEA_result['VOC']  # viable cost $/L
        refinery_attribute[i, 15] = TEA_result['FOC']  # fixed cost $/L
        refinery_attribute[i, 17] = TEA_result['TCI']  # total investment
        loan = Loan(principal=TEA_result['TCI'], interest=config.ref_interest_rate[0],
                    term=config.refinery_life_span)
        interest = float(12 * loan._monthly_payment) - TEA_result['TCI'] / config.refinery_life_span
        refinery_attribute[i, 18] = interest  # total investment

    corn_ref_farmer_dist_matrix = np.loadtxt('./data/corn_ref_farmer_dist_matrix.csv', delimiter=',', skiprows=0)
    ref_list = gen_agents.gen_refinery(N_ref, refinery_attribute, corn_ref_farmer_dist_matrix, np.zeros(N_ref))
    for i in range(N_ref):
        ref_list[i].Attributes['start_year'] = 0
        ref_list[i].Attributes['aver_dist'] = 55
    ref_list_backup = copy.deepcopy(ref_list)

    # start simple looping
    # year = 0
    bagasse_avaiable = 0  # assuming there is no bagasse in the market at the begining of simulation
    feed_prices = copy.deepcopy(config.external_provided_prices[0, 7:])
    product_prices_pred = copy.deepcopy(config.external_provided_prices[0, :7])
    price_adj = np.zeros(7)  # the price adjusting for cellulosic refinery products
    price_adj[0] = config.endogenous_ethanol_price[0] - config.external_provided_prices[0, 0]
    feed_prices_hist = [copy.deepcopy(feed_prices)]
    product_prices_pred_hist = [copy.deepcopy(product_prices_pred)]
    # feed_prices_hist = [copy.deepcopy(feed_prices)]
    price_adj_hist = [copy.deepcopy(price_adj)]
    invest_cost_his = np.zeros((config.simu_horizon, 4))
    fix_prod_cost_his = np.zeros((config.simu_horizon, 4))
    via_prod_cost_his = np.zeros((config.simu_horizon, 4))
    ref_inv_cost_adj_his = params.ref_inv_cost_adj_his  # the investment cost adjustment of learning by doing
    ref_pro_cost_adj_his = params.ref_pro_cost_adj_his  # the production cost adjustment of learning by doing
    BCHP_invest_cost_hist = []
    BCHP_pro_cost_hist = []
    BCHP_invest_cost_hist.append(copy.deepcopy(config.ref_cost_table[6,1]))
    BCHP_pro_cost_hist.append(copy.deepcopy(config.ref_cost_table[6,5]))
    IRR_adj_factor=[0]

    trans_costs_his = []
    storage_costs_his = []
    trans_costs_his.append(copy.deepcopy(config.trans_costs))
    storage_costs_his.append(copy.deepcopy(config.storage_costs))

    for temp_agent in farmer_agent_list:
        temp_agent.cal_adopt_priority()
        temp_agent.cal_PT_imp_env()

    # year=0
    # gov_agent.cal_CWC_price(config.gov_related_prices[year,1],config.gov_related_prices[year,0])
    # p_cell_ethanol_temp = gov_agent.cal_cell_ethanol_price(3.78541*config.external_provided_prices[year,0],0) # in $/gallon
    # p_cell_ethanol_temp = p_cell_ethanol_temp  ############################ check ##################
    #
    # # eleventh step is to update consumer's willingness to pay
    # consumer_agent.cal_willingness_to_pay()
    # p_cell_consumer = consumer_agent.cal_ethanol_price(3.78541*config.external_provided_prices[year, 0])
    # p_cell_ethanol_temp = max(p_cell_ethanol_temp,p_cell_consumer) # in $/gallon
    # p_adj_for_cell_ethanol = p_cell_ethanol_temp/3.78541 - config.external_provided_prices[year, 0]  # adjust the cellulosic ethanol price after policy intervension
    # price_adj = np.asarray([p_adj_for_cell_ethanol,p_adj_for_cell_ethanol*config.ethanol_equivilents[1],0,0,0]) # price adjust for cellulosic refinery

    for year in range(params.simu_horizon):
        print(year)
        # step zero is to update the policy information to each farmer agent
        for temp_farmer in farmer_agent_list:
            TMDL_eligible_temp = (config.patch_N_loads_mean[temp_farmer.Attributes['patch_ID'], 0] > gov.TMDL_N_limits[
                gov_agent.States['N_limit_ID'][-1]]) + 0
            temp_farmer.States['TMDL_eligible'].append(TMDL_eligible_temp)
            temp_farmer.States['BCAP_eligible'].append(np.zeros(temp_farmer.Attributes['patch_ID'].__len__(), int))
            for j in range(ref_list.__len__()):
                if (ref_list[j].Attributes['dist_farmer'][temp_farmer.ID] <= config.patch_influence_range / 3) & (
                        ref_list[j].Attributes['tech_type'] > 1):
                    temp_farmer.States['BCAP_eligible'][-1] = np.ones(temp_farmer.Attributes['patch_ID'].__len__(), int)
                    break

        # feed_prices = copy.deepcopy(config.external_provided_prices[year, 7:])
        mis_price_market, market_demand_watershed, market_supply_watershed = GME.update_market_feed_price(ref_list,
                                                                    farmer_agent_list,config.external_provided_prices[year, 9],product_prices_pred)
        feed_prices = copy.deepcopy(config.external_provided_prices[year, 7:])
        feed_prices[2:4] = copy.deepcopy(mis_price_market)

        total_cap_corn_ref = 0
        total_cap_cell_ref = 0
        for temp_ref_agent in ref_list:
            if temp_ref_agent.Attributes['tech_type'] == 1:
                total_cap_corn_ref += temp_ref_agent.Attributes['capacity'][-1]
            elif temp_ref_agent.Attributes['tech_type'] == 2:
                total_cap_cell_ref += temp_ref_agent.Attributes['capacity'][-1]

        can_refinery_attribute[can_refinery_attribute[:, 3] == 2, 5] += IRR_adj_factor[-1]  # update the minimum IRR based on RFS singal
        can_refinery_attribute[can_refinery_attribute[:, 5] > 0.35, 5] = 0.35
        can_refinery_attribute[(can_refinery_attribute[:, 5] < 0.15)&(can_refinery_attribute[:, 6]==1), 5] = 0.15
        can_refinery_attribute[(can_refinery_attribute[:, 5] < 0.15) & (can_refinery_attribute[:, 6] == 0), 5] = 0.15

        can_ref_list = gen_agents.gen_candidate_refinery(N_can_ref, can_refinery_attribute,
                                                         ref_farmer_dist_matrix)  # re-initiate candidate refinery every year
        GMP.initiate_community_new_time_step(community_list)
        occupied_feed = GMP.cal_occupied_feed(farmer_agent_list, ref_list)
        # can_delete_ID = []
        # first step is simple validity check for refinery feedstock availability and water availability, i.e., update candidate refinery locations
        j = 0  # candicate biorefineries
        while j < can_ref_list.__len__():
            feed_amount, feed_stock_enough, feed_available, aver_dist = GMP.quick_cal_ref_feed_amount(can_ref_list[j],
                                                                                                      farmer_agent_list,
                                                                                                      bagasse_avaiable,
                                                                                                      config.refinery_delta_LU_limit,
                                                                                                      occupied_feed)
            if feed_stock_enough == 0:
                # can_delete_ID.append(j)
                can_ref_list.pop(j)
                continue
            else:
                WU = can_ref_list[j].cal_water_use(feed_amount)
                WA = copy.deepcopy(community_list[can_ref_list[j].Attributes['community_ID']].States['water_avail'][-1])
                if WU > WA:
                    # can_delete_ID.append(j)
                    can_ref_list.pop(j)
                    continue
                else:
                    exist_cap = 0
                    for k in range(ref_list.__len__()):
                        if (can_ref_list[j].Attributes['loc_ID'] == ref_list[k].Attributes['loc_ID']) & \
                                (can_ref_list[j].Attributes['refinery_type'] == ref_list[k].Attributes['tech_type']):
                            exist_cap = exist_cap + ref_list[k].Attributes['capacity'][-1]
                    if exist_cap + can_ref_list[j].Attributes['capacity'] > config.max_ref_cap:
                        can_ref_list.pop(j)
                        continue
                    elif (can_ref_list[j].Attributes['refinery_type'] == 1) & (
                            can_ref_list[j].Attributes['capacity'] + total_cap_corn_ref > config.max_watersh_cap_corn):
                        can_ref_list.pop(j)
                        continue
                    elif (can_ref_list[j].Attributes['refinery_type'] == 2) & (
                            can_ref_list[j].Attributes['capacity'] + total_cap_cell_ref > config.max_watersh_cap_cell):
                        can_ref_list.pop(j)
                        continue
                    else:
                        can_ref_list[j].States['WU'].append(WU)
                        can_ref_list[j].States['feedstock_amount'].append(feed_amount)
                        can_ref_list[j].States['feed_stock_avail'].append(feed_available)
                        can_ref_list[j].States['aver_dist'].append(aver_dist)
                        j = j + 1

        j = 0  # candicate BCHPs
        temp_can_BCHP_list = copy.deepcopy(can_BCHP_list)
        while j < temp_can_BCHP_list.__len__():
            feed_amount, feed_stock_enough, feed_available, aver_dist = GMP.quick_cal_ref_feed_amount(
                temp_can_BCHP_list[j],
                farmer_agent_list, bagasse_avaiable, config.refinery_delta_LU_limit, occupied_feed)
            if feed_stock_enough == 0:
                temp_can_BCHP_list.pop(j)
                continue
            else:
                temp_can_BCHP_list[j].States['WU'].append(0)
                temp_can_BCHP_list[j].States['feedstock_amount'].append(feed_amount)
                temp_can_BCHP_list[j].States['feed_stock_avail'].append(feed_available)
                temp_can_BCHP_list[j].States['aver_dist'].append(aver_dist)
                j = j + 1

        j = 0  # candicate cofire
        temp_can_cofire_list = copy.deepcopy(can_cofire_list)
        while j < temp_can_cofire_list.__len__():
            feed_amount, feed_stock_enough, feed_available, aver_dist = GMP.quick_cal_ref_feed_amount(
                temp_can_cofire_list[j],
                farmer_agent_list,
                bagasse_avaiable,
                config.refinery_delta_LU_limit,
                occupied_feed)
            if feed_stock_enough == 0:
                temp_can_cofire_list.pop(j)
                continue
            else:
                temp_can_cofire_list[j].States['WU'].append(0)
                temp_can_cofire_list[j].States['feedstock_amount'].append(feed_amount)
                temp_can_cofire_list[j].States['feed_stock_avail'].append(feed_available)
                temp_can_cofire_list[j].States['aver_dist'].append(aver_dist)
                j = j + 1
        # for j in np.flip(can_delete_ID,0):
        #     can_ref_list.pop(j)

        # second step is to select the candidate refinery to invest
        patch_BEM = GME.cal_BEM_patch(farmer_agent_list, config.N_patch, 6, np.delete(feed_prices, 2),
                                      config.empirical_risks, year)
        patch_BEM = np.insert(patch_BEM, 2, config.stover_harvest_cost,axis=1)  # the breakeven price of corn stover is the harvesting cost
        patch_PM = GMP.cal_PM_for_ref(farmer_agent_list, config.N_patch, 6)
        patch_PM = np.insert(patch_PM, 2, patch_PM[:, 0] * config.stover_harvest_ratio, axis=1)
        j = 0  # candidate refineries
        while j < can_ref_list.__len__():
            ref_type = copy.deepcopy(can_ref_list[j].Attributes['refinery_type'])
            ref_plan_feed_prices = feed_prices + config.trans_costs * \
                                   (config.patch_influence_range - config.patch_influence_range)  # the feedstock price for refinery planning stage
            for k in range(10):
                # if the available feedstock is enough to support the refinery, then there is no need to check the supply curve
                # if k==5 (bagasse), also no need to check supply curve
                if (can_ref_list[j].States['feedstock_amount'][-1][k] <= can_ref_list[j].States['feed_stock_avail'][-1][
                    k]) | (k == 5):
                    continue
                else:  # otherwise, the refinery need to use the supply curve to calculate the new feedstock price to support its production
                    if k == 2:
                        crop_ID = 2  # 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
                        # 5 for sorghum, 6 for cane
                    elif k < 2:
                        crop_ID = GMP.feed_ID_to_land_use(k) - 1
                    else:
                        crop_ID = GMP.feed_ID_to_land_use(k)
                    supply_list = can_ref_list[j].cal_supply_curve(crop_ID, patch_BEM, patch_PM, farmer_agent_list,
                                                                   config.N_patch)
                    demand = can_ref_list[j].States['feedstock_amount'][-1][k] - \
                             can_ref_list[j].States['feed_stock_avail'][-1][k]
                    demand = demand * (demand > 0.01)  # if demand is close to 0, make it 0
                    price, supply = can_ref_list[j].check_supply_curve(supply_list, demand)
                    if k > 7:
                        ref_plan_feed_prices[k - 2] = copy.deepcopy(price)
                    else:
                        ref_plan_feed_prices[k] = copy.deepcopy(price)
            # invest_cost_components = np.asarray([config.ref_fix_invest_costs[ref_type-1], config.ref_via_invest_costs[ref_type-1],config.ref_base_cap[ref_type-1]])
            # production_costs = np.asarray([config.ref_fix_oper_costs[ref_type-1], config.ref_via_oper_costs[ref_type-1]])
            NPV, IRR, invest_cost = can_ref_list[j].cal_NPV(ref_plan_feed_prices, product_prices_pred, price_adj,
                                                            config.trans_costs, config.storage_costs,
                                                            gov.ref_subsidys[can_ref_list[j].Attributes['refinery_type'] - 1, :],
                                                            gov.ref_taxs[can_ref_list[j].Attributes['refinery_type'] - 1, :],
                                                            ref_inv_cost_adj_his, ref_pro_cost_adj_his, year)
            if (IRR < can_ref_list[j].Attributes['IRR_min']) | (np.isnan(IRR)) | (NPV <= 0):
                can_ref_list.pop(j)
                continue
            else:
                # check community approval
                community_ID = copy.deepcopy(can_ref_list[j].Attributes['community_ID'])
                delta_revenue, delta_LU, delta_PI, WU = community_list[community_ID].pred_lU_change(can_ref_list[j],
                                                                                                    farmer_agent_list,
                                                                                                    feed_prices,
                                                                                                    ref_plan_feed_prices,
                                                                                                    config.average_crop_yields)
                temp = np.where((config.ref_job_table[:, 0] == can_ref_list[j].Attributes['capacity']) & (
                            config.ref_job_table[:, 1] == can_ref_list[j].Attributes['refinery_type']))[0]
                N_job = int(config.ref_job_table[temp, 2])
                w = community_list[community_ID].cal_willingess(
                    community_list[community_ID].States['revenue'][-1] + delta_revenue,
                    N_job, WU, delta_LU, can_ref_list[j].Attributes['capacity'])
                if w < community_list[community_ID].Attributes['accept_threshold']:
                    can_ref_list.pop(j)
                    community_list[community_ID].States['denial'].append(1)
                    continue
                else:
                    j = j + 1

        j = 0  # candidate BCHP
        while j < temp_can_BCHP_list.__len__():
            ref_type = copy.deepcopy(temp_can_BCHP_list[j].Attributes['refinery_type'])
            ref_plan_feed_prices = feed_prices + config.trans_costs * \
                                   (config.patch_influence_range - config.patch_influence_range)  # the feedstock price for refinery planning stage
            for k in range(10):
                # if the available feedstock is enough to support the refinery, then there is no need to check the supply curve
                # if k==5 (bagasse), also no need to check supply curve
                if (temp_can_BCHP_list[j].States['feedstock_amount'][-1][k] <=
                    temp_can_BCHP_list[j].States['feed_stock_avail'][-1][k]) | (k == 5):
                    continue
                else:  # otherwise, the refinery need to use the supply curve to calculate the new feedstock price to support its production
                    if k == 2:
                        crop_ID = 2  # 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
                        # 5 for sorghum, 6 for cane
                    elif k < 2:
                        crop_ID = GMP.feed_ID_to_land_use(k) - 1
                    else:
                        crop_ID = GMP.feed_ID_to_land_use(k)
                    supply_list = temp_can_BCHP_list[j].cal_supply_curve(crop_ID, patch_BEM, patch_PM,
                                                                         farmer_agent_list,
                                                                         config.N_patch)
                    demand = temp_can_BCHP_list[j].States['feedstock_amount'][-1][k] - \
                             temp_can_BCHP_list[j].States['feed_stock_avail'][-1][k]
                    demand = demand * (demand > 0.01)  # if demand is close to 0, make it 0
                    price, supply = temp_can_BCHP_list[j].check_supply_curve(supply_list, demand)
                    if k > 7:
                        ref_plan_feed_prices[k - 2] = copy.deepcopy(price)
                    else:
                        ref_plan_feed_prices[k] = copy.deepcopy(price)
            # invest_cost_components = np.asarray([config.ref_fix_invest_costs[ref_type-1], config.ref_via_invest_costs[ref_type-1],config.ref_base_cap[ref_type-1]])
            # production_costs = np.asarray([config.ref_fix_oper_costs[ref_type-1], config.ref_via_oper_costs[ref_type-1]])
            NPV, IRR, invest_cost = temp_can_BCHP_list[j].cal_NPV(ref_plan_feed_prices, product_prices_pred, price_adj,
                                                                  config.trans_costs, config.storage_costs,
                                                                  gov.ref_subsidys[temp_can_BCHP_list[j].Attributes['refinery_type'] - 1,:],
                                                                  gov.ref_taxs[temp_can_BCHP_list[j].Attributes['refinery_type'] - 1,:],
                                                                  ref_inv_cost_adj_his, ref_pro_cost_adj_his, year)
            if (IRR < temp_can_BCHP_list[j].Attributes['IRR_min']) | (np.isnan(IRR)) | (NPV <= 0):
                temp_can_BCHP_list.pop(j)
                continue
            j = j + 1

        j = 0  # candidate cofire
        while j < temp_can_cofire_list.__len__():
            ref_type = copy.deepcopy(temp_can_cofire_list[j].Attributes['refinery_type'])
            ref_plan_feed_prices = feed_prices + config.trans_costs * \
                                   (config.patch_influence_range - config.patch_influence_range)  # the feedstock price for refinery planning stage
            for k in range(10):
                # if the available feedstock is enough to support the refinery, then there is no need to check the supply curve
                # if k==5 (bagasse), also no need to check supply curve
                if (temp_can_cofire_list[j].States['feedstock_amount'][-1][k] <=
                    temp_can_cofire_list[j].States['feed_stock_avail'][-1][
                        k]) | (k == 5):
                    continue
                else:  # otherwise, the refinery need to use the supply curve to calculate the new feedstock price to support its production
                    if k == 2:
                        crop_ID = 2  # 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
                        # 5 for sorghum, 6 for cane
                    elif k < 2:
                        crop_ID = GMP.feed_ID_to_land_use(k) - 1
                    else:
                        crop_ID = GMP.feed_ID_to_land_use(k)
                    supply_list = temp_can_cofire_list[j].cal_supply_curve(crop_ID, patch_BEM, patch_PM,
                                                                           farmer_agent_list,
                                                                           config.N_patch)
                    demand = temp_can_cofire_list[j].States['feedstock_amount'][-1][k] - \
                             temp_can_cofire_list[j].States['feed_stock_avail'][-1][k]
                    demand = demand * (demand > 0.01)  # if demand is close to 0, make it 0
                    price, supply = temp_can_cofire_list[j].check_supply_curve(supply_list, demand)
                    if k > 7:
                        ref_plan_feed_prices[k - 2] = copy.deepcopy(price)
                    else:
                        ref_plan_feed_prices[k] = copy.deepcopy(price)
            # invest_cost_components = np.asarray([config.ref_fix_invest_costs[ref_type-1], config.ref_via_invest_costs[ref_type-1],config.ref_base_cap[ref_type-1]])
            # production_costs = np.asarray([config.ref_fix_oper_costs[ref_type-1], config.ref_via_oper_costs[ref_type-1]])
            NPV, IRR, invest_cost = temp_can_cofire_list[j].cal_NPV(ref_plan_feed_prices, product_prices_pred,
                                                                    price_adj,
                                                                    config.trans_costs, config.storage_costs,
                                                                    gov.ref_subsidys[temp_can_cofire_list[j].Attributes[
                                                                                         'refinery_type'] - 1, :],
                                                                    gov.ref_taxs[temp_can_cofire_list[j].Attributes[
                                                                                     'refinery_type'] - 1, :],
                                                                    ref_inv_cost_adj_his, ref_pro_cost_adj_his, year)
            if (IRR < temp_can_cofire_list[j].Attributes['IRR_min']) | (np.isnan(IRR)) | (NPV <= 0):
                temp_can_cofire_list.pop(j)
                continue
            j = j + 1

        # third step is to select the top 1 candidate refinery plans and transfer them to the list of REAL refineries
        if can_ref_list.__len__() > 0:  # if there still remains candidate refinery, choose the one with top 2 IRR
            max_NPVs = -1 * np.ones(config.N_can_ref_locs)  # initiate the maximum IRR values for each possible candidate refinery location
            best_can_ref_IDs = -1 * np.ones(config.N_can_ref_locs)  # initiate the ID of best refinery for each possible candidate refinery location
            for j in range(can_ref_list.__len__()):
                if can_ref_list[j].States['NPV'][-1] > max_NPVs[can_ref_list[j].Attributes['loc_ID']]:
                    max_NPVs[can_ref_list[j].Attributes['loc_ID']] = can_ref_list[j].States['NPV'][-1]
                    best_can_ref_IDs[can_ref_list[j].Attributes['loc_ID']] = int(j)
            temp_max_NPVs = copy.deepcopy(max_NPVs)
            max_NPVs = copy.deepcopy(max_NPVs[np.where(max_NPVs > 0)])
            best_can_ref_IDs = copy.deepcopy(best_can_ref_IDs[np.where(temp_max_NPVs > 0)].astype(int))
            sort_ID = np.argsort(max_NPVs)
            max_NPVs = copy.deepcopy(max_NPVs[sort_ID])
            best_can_ref_IDs = copy.deepcopy(best_can_ref_IDs[sort_ID])
            if max_NPVs.__len__() <= 2:  # if there is less than 2 candidate refinery qualifies, build all
                new_refs = can_ref_list
            elif max_NPVs.__len__() > 2:  # otherwise, build the refinerys with top NPVs
                new_refs = [can_ref_list[jj] for jj in best_can_ref_IDs[-2:]]
            else:
                new_refs = []

            for ref_agent in new_refs:
                loc_ID = copy.deepcopy(ref_agent.Attributes['loc_ID'])
                planned_feed_amount = copy.deepcopy(ref_agent.States['feedstock_amount'][-1])
                capacity = copy.deepcopy(ref_agent.Attributes['capacity'])
                tech_type = int(ref_agent.Attributes['refinery_type'])
                invest_amount = copy.deepcopy(ref_agent.States['invest_cost'][-1])
                interest_payment = copy.deepcopy(ref_agent.States['interest_payment'][-1])
                fix_cost = copy.deepcopy(ref_agent.States['fix_cost'][-1])
                via_cost = copy.deepcopy(ref_agent.States['via_cost'][-1])
                is_co_op = copy.deepcopy(ref_agent.Attributes['co-op'])
                bagasse_amount = copy.deepcopy(planned_feed_amount[5])
                community_ID = copy.deepcopy(ref_agent.Attributes['community_ID'])
                WU = copy.deepcopy(ref_agent.States['WU'][-1])
                community_list[community_ID].Temp['WU'].append(WU)
                aver_dist = copy.deepcopy(ref_agent.States['aver_dist'][-1])
                tempi = 0
                for tempj in ref_list:
                    if (loc_ID == tempj.Attributes['loc_ID']) & (is_co_op == tempj.Attributes['co-op']) & (
                            tech_type == tempj.Attributes['tech_type']):
                        attribute_temp = copy.deepcopy(tempj.Attributes)
                        attribute_temp['feedstock_amount'] = attribute_temp['feedstock_amount'] + planned_feed_amount
                        attribute_temp['fix_cost'] = (attribute_temp['capacity'][-1] * attribute_temp['fix_cost'] +
                                                      capacity * fix_cost) / (attribute_temp['capacity'][-1] + capacity)
                        attribute_temp['via_cost'] = (attribute_temp['capacity'][-1] * attribute_temp['via_cost'] +
                                                      capacity * via_cost) / (attribute_temp['capacity'][-1] + capacity)
                        # attribute_temp['capacity'].append(attribute_temp['capacity'][-1] + capacity)
                        tempj.Temp['temp_cap'].append(capacity)  # temporally save the new capacity and then add it to the attribute at the end of time step
                        attribute_temp['invest'] = attribute_temp['invest'] + invest_amount
                        attribute_temp['interest_payment'].append(interest_payment)
                        attribute_temp['bagasse_amount'] = attribute_temp['bagasse_amount'] + bagasse_amount
                        state_temp = copy.deepcopy(tempj.States)
                        # state_temp['production_year'].append(-1)
                        tempj.Attributes = copy.deepcopy(attribute_temp)
                        tempj.States = copy.deepcopy(state_temp)
                        tempi = 1
                    else:
                        continue

                if tempi == 0:  # if the new capacity is not for expand capacity of existing refineries
                    attribute_temp = np.zeros((1, 20))
                    attribute_temp[0, 0] = ref_list_backup.__len__()
                    attribute_temp[0, 1] = copy.deepcopy(loc_ID)
                    attribute_temp[0, 2:12] = copy.deepcopy(planned_feed_amount)
                    attribute_temp[0, 12] = copy.deepcopy(capacity)
                    attribute_temp[0, 13] = copy.deepcopy(tech_type)
                    attribute_temp[0, 14] = copy.deepcopy(via_cost)
                    attribute_temp[0, 15] = copy.deepcopy(fix_cost)
                    attribute_temp[0, 16] = 0.2
                    attribute_temp[0, 17] = copy.deepcopy(invest_amount)
                    attribute_temp[0, 18] = copy.deepcopy(interest_payment)
                    attribute_temp[0, 19] = copy.deepcopy(is_co_op)
                    ref_list = gen_agents.add_new_refinery(ref_list, 1, attribute_temp,
                                                           ref_agent.Attributes['dist_farmer'],
                                                           [bagasse_amount])
                    ref_list[-1].Attributes['start_year'] = year
                    ref_list[-1].Attributes['aver_dist'] = aver_dist.max()
                    ref_list_backup.append(ref_list[-1])

        if temp_can_BCHP_list.__len__() > 0:  # if there still remains candidate BCHP, choose the one with top 2 NPV
            max_NPVs = -1 * np.ones(N_can_BCHP)  # initiate the maximum IRR values for each possible candidate BCHP
            best_can_ref_IDs = -1 * np.ones(N_can_BCHP)  # initiate the ID of best refinery for each possible candidate BCHP
            for j in range(temp_can_BCHP_list.__len__()):
                max_NPVs[j] = temp_can_BCHP_list[j].States['NPV'][-1]
                best_can_ref_IDs[j] = int(j)
            temp_max_NPVs = copy.deepcopy(max_NPVs)
            max_NPVs = copy.deepcopy(max_NPVs[np.where(max_NPVs > 0)])
            best_can_ref_IDs = copy.deepcopy(best_can_ref_IDs[np.where(temp_max_NPVs > 0)].astype(int))
            sort_ID = np.argsort(max_NPVs)
            max_NPVs = copy.deepcopy(max_NPVs[sort_ID])
            best_can_ref_IDs = copy.deepcopy(best_can_ref_IDs[sort_ID])
            if max_NPVs.__len__() <= 2:  # if there is less than 2 candidate BCHP qualifies, build all
                new_refs = temp_can_BCHP_list
            elif max_NPVs.__len__() > 2:  # otherwise, build the 2 BCHPs with top NPVs
                new_refs = [temp_can_BCHP_list[jj] for jj in best_can_ref_IDs[-2:]]
            else:
                new_refs = []

            for ref_agent in new_refs:
                can_BCHP_remove_ID = \
                np.where(can_BCHP_IDs == ref_agent.Attributes['loc_ID'] - config.N_can_ref_locs)[0][0]
                can_BCHP_IDs = np.delete(can_BCHP_IDs, can_BCHP_remove_ID)
                can_BCHP_list.pop(can_BCHP_remove_ID)
                planned_feed_amount = copy.deepcopy(ref_agent.States['feedstock_amount'][-1])
                bagasse_amount = copy.deepcopy(planned_feed_amount[5])
                community_ID = copy.deepcopy(ref_agent.Attributes['community_ID'])
                WU = copy.deepcopy(ref_agent.States['WU'][-1])
                community_list[community_ID].Temp['WU'].append(WU)
                attribute_temp = np.zeros((1, 20))
                attribute_temp[0, 0] = ref_list_backup.__len__()
                attribute_temp[0, 1] = copy.deepcopy(ref_agent.Attributes['loc_ID'])
                attribute_temp[0, 2:12] = copy.deepcopy(ref_agent.States['feedstock_amount'][-1])
                attribute_temp[0, 12] = copy.deepcopy(ref_agent.Attributes['capacity'])
                attribute_temp[0, 13] = copy.deepcopy(ref_agent.Attributes['refinery_type'])
                attribute_temp[0, 14] = copy.deepcopy(ref_agent.States['via_cost'][-1])
                attribute_temp[0, 15] = copy.deepcopy(ref_agent.States['fix_cost'][-1])
                attribute_temp[0, 16] = 0.2
                attribute_temp[0, 17] = copy.deepcopy(ref_agent.States['invest_cost'][-1])
                attribute_temp[0, 18] = copy.deepcopy(ref_agent.States['interest_payment'][-1])
                attribute_temp[0, 19] = copy.deepcopy(ref_agent.Attributes['co-op'])
                ref_list = gen_agents.add_new_refinery(ref_list, 1, attribute_temp, ref_agent.Attributes['dist_farmer'],
                                                       [bagasse_amount])
                ref_list[-1].Attributes['start_year'] = year
                ref_list[-1].Attributes['aver_dist'] = ref_agent.States['aver_dist'][-1].max()
                ref_list_backup.append(ref_list[-1])

        if temp_can_cofire_list.__len__() > 0:  # if there still remains candidate cofiring, choose the all
            max_NPVs = -1 * np.ones(N_can_cofire)  # initiate the maximum IRR values for each possible candidate BCHP
            best_can_ref_IDs = -1 * np.ones(N_can_cofire)  # initiate the ID of best refinery for each possible candidate BCHP
            for j in range(temp_can_cofire_list.__len__()):
                max_NPVs[j] = temp_can_cofire_list[j].States['NPV'][-1]
                best_can_ref_IDs[j] = int(j)
            temp_max_NPVs = copy.deepcopy(max_NPVs)
            max_NPVs = copy.deepcopy(max_NPVs[np.where(max_NPVs > 0)])
            best_can_ref_IDs = copy.deepcopy(best_can_ref_IDs[np.where(temp_max_NPVs > 0)].astype(int))
            sort_ID = np.argsort(max_NPVs)
            max_NPVs = copy.deepcopy(max_NPVs[sort_ID])
            best_can_ref_IDs = copy.deepcopy(best_can_ref_IDs[sort_ID])
            if max_NPVs.__len__() <= 2:  # if there is less than 2 candidate BCHP qualifies, build all
                new_refs = temp_can_cofire_list
            elif max_NPVs.__len__() > 2:  # otherwise, build the 2 BCHPs with top NPVs
                new_refs = [temp_can_cofire_list[jj] for jj in best_can_ref_IDs[-2:]]
            else:
                new_refs = []

            for ref_agent in new_refs:
                can_cofire_remove_ID = \
                np.where(can_cofire_IDs == ref_agent.Attributes['loc_ID'] - config.N_can_ref_locs - config.N_can_BCHP)[
                    0][0]
                can_cofire_IDs = np.delete(can_cofire_IDs, can_cofire_remove_ID)
                can_cofire_list.pop(can_cofire_remove_ID)
                planned_feed_amount = copy.deepcopy(ref_agent.States['feedstock_amount'][-1])
                bagasse_amount = copy.deepcopy(planned_feed_amount[5])
                community_ID = copy.deepcopy(ref_agent.Attributes['community_ID'])
                WU = copy.deepcopy(ref_agent.States['WU'][-1])
                community_list[community_ID].Temp['WU'].append(WU)
                attribute_temp = np.zeros((1, 20))
                attribute_temp[0, 0] = ref_list_backup.__len__()
                attribute_temp[0, 1] = copy.deepcopy(ref_agent.Attributes['loc_ID'])
                attribute_temp[0, 2:12] = copy.deepcopy(ref_agent.States['feedstock_amount'][-1])
                attribute_temp[0, 12] = copy.deepcopy(ref_agent.Attributes['capacity'])
                attribute_temp[0, 13] = copy.deepcopy(ref_agent.Attributes['refinery_type'])
                attribute_temp[0, 14] = copy.deepcopy(ref_agent.States['via_cost'][-1])
                attribute_temp[0, 15] = copy.deepcopy(ref_agent.States['fix_cost'][-1])
                attribute_temp[0, 16] = 0.2
                attribute_temp[0, 17] = copy.deepcopy(ref_agent.States['invest_cost'][-1])
                attribute_temp[0, 18] = copy.deepcopy(ref_agent.States['interest_payment'][-1])
                attribute_temp[0, 19] = copy.deepcopy(ref_agent.Attributes['co-op'])
                ref_list = gen_agents.add_new_refinery(ref_list, 1, attribute_temp, ref_agent.Attributes['dist_farmer'],
                                                       [bagasse_amount])
                ref_list[-1].Attributes['start_year'] = year
                ref_list[-1].Attributes['aver_dist'] = ref_agent.States['aver_dist'][-1].max()
                ref_list_backup.append(ref_list[-1])

        # fourth step is for the REAL refineries to negotiate contract with farmers
        PBE = np.zeros((8, ref_list.__len__()))
        GMP.initiate_farmer_new_time_step(farmer_agent_list,feed_prices[2])  # initiate the land use decisions for all farmers at the begining of a year
        market_demand = np.zeros(3)
        market_supply = np.zeros(3)
        for j in range(3):
            market_demand[j] = GMP.check_feed_demand(ref_list, j + 1)
            market_supply[j] = GMP.check_feed_supply(farmer_agent_list, feed_prices, j + 1)

        feed_prices_ref = feed_prices + config.trans_costs * (config.patch_influence_range - config.patch_influence_range)
        for j in range(ref_list.__len__()):
            # check if the contracted feedstock is enough to fill the capacity
            temp_ref_agent = ref_list[j]
            contracted_farmer_ID, contracted_patch_ID, contracted_patch_amount, \
            contracted_patch_dist, contracted_patch_price = temp_ref_agent.check_contract_continuity(farmer_agent_list)
            # contracted_patch_amount = copy.deepcopy(temp_ref_agent.States['contracted_patch_amount'][-1])
            capacity = copy.deepcopy(temp_ref_agent.Attributes['capacity'][-1])
            tech_type = copy.deepcopy(temp_ref_agent.Attributes['tech_type'])
            PBE[:, j] = temp_ref_agent.cal_PBE(product_prices_pred, config.trans_costs, config.storage_costs)

            # dist_matrix_temp = temp_ref_agent.Attributes['dist_farmer']
            # ID_farmer_within_range = np.where(dist_matrix_temp <= config.patch_influence_range)[0]

            if tech_type == 1:  # corn refinery does not make contract
                get_new_contracts = 0
            elif contracted_patch_amount.sum() == 0:  # if there is no contracted amount
                get_new_contracts = 1
            elif GMP.cal_ref_production_amount(contracted_patch_amount.sum(0),tech_type) < capacity:
                get_new_contracts = 1
            else:
                get_new_contracts = 0

            if get_new_contracts == 1:
                # PBE[:,j] = temp_ref_agent.cal_PBE(product_prices_pred,config.trans_costs,config.storage_costs)
                # for k in range(ID_farmer_within_range.__len__()): # check each farmer if they are going to make contract
                if tech_type == 1:
                    temp_contract_crop_ID = 0
                elif (tech_type == np.asarray([2, 5, 6, 7])).sum() > 0:
                    temp_contract_crop_ID = 1
                else:
                    temp_contract_crop_ID = 2
                market_demand[temp_contract_crop_ID], market_supply[temp_contract_crop_ID] = \
                    temp_ref_agent.make_contracts(farmer_agent_list, ref_list, feed_prices_ref, config.trans_costs,
                                                  config.storage_costs,
                                                  PBE[:, j], market_demand[temp_contract_crop_ID],
                                                  market_supply[temp_contract_crop_ID],
                                                  community_list, year)
            else:
                temp_ref_agent.States['contracted_patch_amount'].append(contracted_patch_amount)
                temp_ref_agent.States['contracted_patch_price'].append(contracted_patch_price)
                temp_ref_agent.States['contracted_patch_dist'].append(contracted_patch_dist)
                temp_ref_agent.States['contracted_farmer_ID'].append(contracted_farmer_ID)
                temp_ref_agent.States['contracted_patch_ID'].append(contracted_patch_ID)

        # fifth step is for farmers to identify their land use
        #   calculate the attitude of all farmers before they influence each other
        for loop_farmer_agent in farmer_agent_list:
            is_contract = loop_farmer_agent.States['contract'][-1] - \
                          2 * (loop_farmer_agent.States['peren_age'][-1] * (
                        loop_farmer_agent.States['contract'][-1] > 0) >= GMP.cal_contract_length(
                loop_farmer_agent.States['land_use'][-1])) + \
                          2 * (loop_farmer_agent.Temp[
                                   'contract_land_use'] > 2)  # first remove the contract with perennial age exceeds the life span, then add up new contracts
            opt_land_use = loop_farmer_agent.land_use_decision_bn(is_contract, community_list, year)
            loop_farmer_agent.Temp['opt_land_use'] = copy.deepcopy(opt_land_use)
            loop_farmer_agent.compile_farmer_contract(opt_land_use)

        # sixth step is for the environmental model to grow crop, calculate N load and streamflow
        # is_flood, is_drought = GMP.cal_flood_drought(config.Prcp_data)
        is_flood, is_drought = (config.Prcp_data[year] == 1), (config.Prcp_data[year] == 2)
        for j in range(N_farmer):
            farmer_agent_temp = farmer_agent_list[j]
            N_patch_temp = farmer_agent_temp.Attributes['patch_ID'].__len__()
            yield_patch_temp = np.zeros(N_patch_temp)
            N_release_temp = np.zeros(N_patch_temp)
            ferti_use_temp = np.zeros(N_patch_temp)
            peren_age_temp = np.zeros(N_patch_temp)
            C_sequest_temp = np.zeros(N_patch_temp)
            for k in range(N_patch_temp):
                patch_ID = copy.deepcopy(farmer_agent_temp.Attributes['patch_ID'][k])
                slope = copy.deepcopy(farmer_agent_temp.Attributes['patch_slope'][k])
                land_use_b = copy.deepcopy(farmer_agent_temp.States['land_use'][-2][k])
                land_use_n = copy.deepcopy(farmer_agent_temp.States['land_use'][-1][k])
                perennial_age = copy.deepcopy(farmer_agent_temp.States['peren_age'][-1][k])
                if farmer_agent_temp.Temp['peren_age_refresh'][k] == 1:
                    perennial_age = -2
                is_stocha = 1
                output = GMP.look_up_table_crop_no_physical_model(patch_ID, is_flood, is_drought, slope, land_use_b,
                                                                  land_use_n, 0, perennial_age, year, is_stocha)
                yield_patch_temp[k] = copy.deepcopy(output['yield'])
                N_release_temp[k] = copy.deepcopy(output['N_release'])
                C_sequest_temp[k] = copy.deepcopy(output['C_sequest'])
                ferti_use_temp[k] = copy.deepcopy(output['ferti_use'])
                peren_age_temp[k] = copy.deepcopy(output['peren_age'])
                if land_use_n != 1:
                    farmer_agent_temp.Temp['stover_available_for_sale'][k] = 0
                if (land_use_n == 7) | (land_use_n == 8):
                    farmer_agent_temp.Temp['patch_available_for_sale'][k] = 0
                if farmer_agent_temp.States['contract'][-1][k] == 1:
                    farmer_agent_temp.Temp['patch_available_for_sale'][k] = 0

            farmer_agent_temp.States['yield'].append(yield_patch_temp)
            farmer_agent_temp.States['N_release'].append(N_release_temp)
            farmer_agent_temp.States['C_sequest'].append(C_sequest_temp)
            farmer_agent_temp.States['ferti_use'].append(ferti_use_temp)
            farmer_agent_temp.States['peren_age'].append(peren_age_temp.astype(int))

        # seventh step is refinery buy/sell their feedstock
        N_ref_temp = ref_list.__len__()
        ref_feed_buy = np.zeros((10, N_ref_temp))
        ref_feed_sell = np.zeros((10, N_ref_temp))
        contracted_feedstock_amount = GMP.cal_contracted_feedstock_amount(farmer_agent_list, ref_list)
        for j in range(N_ref_temp):
            ref_agent_temp = ref_list[j]
            feed_buy_temp, feed_sell_temp = ref_agent_temp.feed_management_after_contract(
                contracted_feedstock_amount[j, :])
            ref_feed_buy[:, j] = copy.deepcopy(feed_buy_temp)
            ref_feed_sell[:, j] = copy.deepcopy(feed_sell_temp)

        feed_prices_out_of_boundary = copy.deepcopy(config.external_provided_prices[year, 7:])
        feed_prices_out_of_boundary = np.append(feed_prices_out_of_boundary, feed_prices_out_of_boundary[-2:])
        GMP.initiate_ref_new_time_step(ref_list)
        GME.match_feed_demand_supply(ref_list, farmer_agent_list, PBE, ref_feed_buy, ref_feed_sell,
                                     config.external_provided_prices[year, 7:])

        # eighth step is refinery produce biofuel and update profits
        feed_prices_hist.append(feed_prices)
        j = 0
        total_feedstock_amount = np.zeros(10)
        while j < ref_list.__len__():  # update refinery profits
            if ref_list[j].Attributes['tech_type'] > 1:
                product_prices = config.external_provided_prices[year,
                                 :7] + price_adj  # adjust the price if it is a cellulosic refinery
            else:
                product_prices = copy.deepcopy(config.external_provided_prices[year, :7])
            total_feedstock_amount_temp, product_amount, profit = ref_list[j].cal_profit(farmer_agent_list,
                                                                                         product_prices,
                                                                                         feed_prices,
                                                                                         feed_prices_out_of_boundary[:8],
                                                                                         config.trans_costs,
                                                                                         config.storage_costs)
            total_feedstock_amount += total_feedstock_amount_temp
            ref_list[j].States['profit'].append(profit)
            if ref_list[j].ID >= ref_list_backup.__len__():
                ref_list_backup.append(ref_list[j])  # add new refinery to the backup list
            else:  # update the states of the backup list
                ref_list_backup[ref_list[j].ID] = copy.deepcopy(ref_list[j])
            close, N_years = ref_list[j].cal_stop_production()
            # refinery also update their capacity at the end of a year
            if ref_list[j].Temp['temp_cap'].__len__() > 0:
                ref_list[j].Attributes['capacity'].append(
                    ref_list[j].Attributes['capacity'][-1] + ref_list[j].Temp['temp_cap'][0])
                ref_list[j].Temp['temp_cap'] = []
                ref_list[j].States['production_year'][-1] = 0
            else:
                ref_list[j].Attributes['capacity'].append(ref_list[j].Attributes['capacity'][-1])

            if close == 1:  # shutdown the loss-making refinery
                ref_list.pop(j)
            else:
                j += 1

        for temp_farmer in farmer_agent_list:  # update farmer's revenue
            # temp_farmer.States['price_prior'].append(price_posterior)
            revenue_temp = temp_farmer.States['yield'][-1] * temp_farmer.Temp['patch_received_prices'] * \
                           temp_farmer.Attributes['patch_areas']
            revenue_temp += (temp_farmer.States['land_use'][-1] == 3) * (
                        config.mis_carbon_credit * gov.carbon_price + config.mis_N_credit * gov.nitrogen_price) + \
                            (temp_farmer.States['land_use'][-1] == 4) * (
                                        config.mis_carbon_credit * gov.carbon_price + + config.mis_N_credit * gov.nitrogen_price)
            revenue_temp = revenue_temp.sum()
            temp_farmer.States['revenue'].append(revenue_temp)
            p_flood_temp = temp_farmer.forecasts_climate(temp_farmer.States['climate_forecasts'][-1][0],
                                                         (config.Prcp_data[year] == 1) + 0)
            p_drought_temp = temp_farmer.forecasts_climate(temp_farmer.States['climate_forecasts'][-1][1],
                                                           (config.Prcp_data[year] == 2) + 0)
            temp_farmer.States['climate_forecasts'].append([p_flood_temp, p_drought_temp, 0.1])
            #
            # price_posterior = temp_farmer.forecasts_price(temp_farmer.States['price_prior'][-1],
            #                                               config.external_provided_prices[year, [7, 8, 9, 10, 13, 14]])
            # temp_farmer.States['price_post'].append(price_posterior)
            # temp_farmer.States['price_prior'].append(price_posterior)
            temp_farmer.States['price_post'].append(feed_prices[np.asarray([0,1,2,3,5,6])])
            temp_farmer.States['price_prior'].append(feed_prices[np.asarray([0,1,2,3,5,6])])
        # ninth step is to update community's attitude and WU
        N_load_total_temp = 0
        for j in range(N_community):
            attitude, N_load = community_list[j].cal_attitude(farmer_agent_list)
            N_load_total_temp += N_load
            community_list[j].States['N_release'].append(N_load)
            revenue_commu_temp = community_list[j].cal_revenue_in_community(farmer_agent_list)
            community_list[j].States['revenue'].append(revenue_commu_temp)
            WU_temp = np.asarray(community_list[j].Temp['WU'])
            community_list[j].States['water_avail'].append(community_list[j].States['water_avail'][-1] - WU_temp.sum())

        # tenth step is to update government policies
        gov_agent.update_TMDL(N_load_total_temp, year)  # update TMDL management rule
        V_CE_temp = 0  # update total cellulosic ethanol equivilent production
        for ref_temp in ref_list:
            if ref_temp.Attributes['tech_type'] == 1:
                continue
            else:
                V_CE_temp += (ref_temp.States['biofuel_production'][-1] * config.ethanol_equivilents).sum()
        V_CE_temp = V_CE_temp / 10 ** 6
        gov_agent.States['RFS_volume'][year], RFS_signal = gov_agent.update_RFS(V_CE_temp,
                                                                                gov.RFS_volume[year - 1],
                                                                                gov_agent.States['RFS_volume'][year],
                                                                                config.maintain_RFS)
        gov.RFS_signal.append(RFS_signal)
        IRR_adj_factor.append(gov_agent.cal_IRR_adj_factor(gov.RFS_signal))
        gov_agent.cal_CWC_price(config.gov_related_prices[year, 1], config.gov_related_prices[year, 0],
                                V_CE_temp, gov_agent.States['RFS_volume'][year - 1])
        p_cell_ethanol_temp = gov_agent.cal_cell_ethanol_price(3.78541 * config.external_provided_prices[year, 0],
                                                               V_CE_temp)  # in $/gallon
        p_cell_ethanol_temp = p_cell_ethanol_temp  ############################ check ##################

        # eleventh step is to update consumer's willingness to pay
        consumer_agent.cal_willingness_to_pay(year)
        p_cell_consumer = consumer_agent.cal_ethanol_price(3.78541 * config.external_provided_prices[year, 0])
        p_cell_ethanol_temp = max(p_cell_ethanol_temp, p_cell_consumer)  # in $/gallon
        p_adj_for_cell_ethanol = p_cell_ethanol_temp / 3.78541 - config.external_provided_prices[
            year, 0]  # adjust the cellulosic ethanol price after policy intervension
        price_adj = np.asarray([p_adj_for_cell_ethanol, p_adj_for_cell_ethanol * config.ethanol_equivilents[1], 0, 0,
                                0])  # price adjust for cellulosic refinery
        price_adj[0] = max(price_adj[0],
                           config.endogenous_ethanol_price[year + 1] - config.external_provided_prices[year + 1, 0])
        price_adj = np.append(price_adj, [0., 0.])
        price_adj_hist.append(copy.deepcopy(price_adj))

        # twelveth step is to update price forecasts
        for j in range(config.external_provided_prices.shape[1]):
            if j < 7:
                product_prices_pred[j] = GME.forecasts_price(product_prices_pred[j],
                                                             config.external_provided_prices[year + 1, j],
                                                             config.price_update_rate)
            # else:
            #     feed_prices[j - 7] = GME.forecasts_price(feed_prices[j - 7],
            #                                                   config.external_provided_prices[year + 1, j],
            #                                                   config.price_update_rate)
        product_prices_pred_hist.append(copy.deepcopy(product_prices_pred))
        # feed_prices_hist.append(copy.deepcopy(feed_prices))
        # thirtyth step is to update the investment cost and supply chain costs according to learning by doing
        invest_cost_his[year, 0] = config.ref_fix_invest_costs[0]
        invest_cost_his[year, 1] = config.ref_fix_invest_costs[1]
        invest_cost_his[year, 2] = config.ref_fix_invest_costs[2]
        invest_cost_his[year, 3] = config.ref_fix_invest_costs[3]

        fix_prod_cost_his[year, 0] = config.ref_fix_oper_costs[0]
        fix_prod_cost_his[year, 1] = config.ref_fix_oper_costs[1]
        fix_prod_cost_his[year, 2] = config.ref_fix_oper_costs[2]
        fix_prod_cost_his[year, 3] = config.ref_fix_oper_costs[3]

        via_prod_cost_his[year, 0] = config.ref_via_oper_costs[0]
        via_prod_cost_his[year, 1] = config.ref_via_oper_costs[1]
        via_prod_cost_his[year, 2] = config.ref_via_oper_costs[2]
        via_prod_cost_his[year, 3] = config.ref_via_oper_costs[3]

        if params.is_ref_learn_by_do == 1: # if the refinery production cost need to be updated based on learning by doing
            total_cap = np.zeros(4)
            for j in range(4):
                for k in range(ref_list.__len__()):
                    if ref_list[k].Attributes['tech_type'] == (j + 1):
                        total_cap[j] += ref_list[k].Attributes['capacity'][-1]
                if j > 0:  # investment of corn refinery will not be updated
                    ref_inv_cost_adj_his[year + 1, j] = GME.cal_learn_by_do(1, config.learn_by_do_rate, total_cap[j])
                    ref_pro_cost_adj_his[year + 1, j] = GME.cal_learn_by_do(1, config.learn_by_do_rate, total_cap[j])
                    # config.ref_via_oper_costs[j] = GME.cal_learn_by_do(via_prod_cost_his[0, j], config.learn_by_do_rate,total_cap[j])

        N_BCHP = 0
        for k in range(ref_list.__len__()): # update the costs related to BCHPs
            if ref_list[k].Attributes['tech_type'] == 7:
                N_BCHP += 1
        config.ref_cost_table[6,1] = BCHP_invest_cost_hist[0] * GME.cal_learn_by_do_BCHP(1,config.learn_by_do_rate,N_BCHP)
        config.ref_cost_table[6,5] = BCHP_pro_cost_hist[0] * GME.cal_learn_by_do_BCHP(1, config.learn_by_do_rate, N_BCHP)
        BCHP_invest_cost_hist.append(copy.deepcopy(config.ref_cost_table[6,1]))
        BCHP_pro_cost_hist.append(copy.deepcopy(config.ref_cost_table[6, 5]))

        config.trans_costs[2:] = trans_costs_his[0][2:] * GME.cal_learn_by_do_supply_chain(1, config.learn_by_do_rate, total_feedstock_amount[2:6].sum())
        config.storage_costs[2:] = storage_costs_his[0][2:] * GME.cal_learn_by_do_supply_chain(1, config.learn_by_do_rate, total_feedstock_amount[2:6].sum())
        trans_costs_his.append(copy.deepcopy(config.trans_costs))
        storage_costs_his.append(copy.deepcopy(config.storage_costs))

        farmer_agent_list,non_type_I_farmer = GMP.change_farmer_type(farmer_agent_list, non_type_I_farmer, params.farm_type_change_prob)

    ABM_parameters = {
        'folder': params.folder,
        'result_file': params.result_file,
        'simu_horizon': params.simu_horizon,
        'land_rent': params.land_rent,
        'marginal_land_rent_adj': params.marginal_land_rent_adj,
        'enhanced_neighbor_impact': params.enhanced_neighbor_impact,
        'learn_by_do_rate': params.learn_by_do_rate,
        'base_cap_for_learn_by_do': params.base_cap_for_learn_by_do,
        'base_feed_amount_for_learn_by_do': params.base_feed_amount_for_learn_by_do,
        'base_BCHP_cap_for_learn_by_do': params.base_BCHP_cap_for_learn_by_do,
        'allowable_defecit': params.allowable_defecit,
        'ini_WP': params.ini_WP,
        'IRW': params.IRW,
        'max_WP': params.max_WP,
        'price_update_rate': params.price_update_rate,
        'maintain_RFS': params.maintain_RFS,
        'CRP_subsidy': params.CRP_subsidy,
        'CRP_relax': params.CRP_relax,
        'TMDL_subsidy': params.TMDL_subsidy,
        'BCAP_subsidy': params.BCAP_subsidy,
        'BCAP_cost_share': params.BCAP_cost_share,
        'tax_deduction': params.tax_deduction,
        'tax_rate': params.tax_rate,
        'carbon_price': params.carbon_price,
        'nitrogen_price': params.nitrogen_price,
        'gov_ref_tax_subsidy': gov.gov_ref_tax_subsidy,
        'external_provided_prices': config.external_provided_prices,
        'RFS_signal': gov.RFS_signal
    }

    # define the name of the directory to be created
    path = "/results"

    try:
        os.makedirs(params.folder + path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

    import shelve
    my_shelf = shelve.open(params.result_file, 'n')  # 'n' for new

    for key in dir():
        try:
            my_shelf[key] = locals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    my_shelf.close()


if __name__ == '__main__':

    import imp

    # Scenarios for methodology
    # Scenario status quo (insight S1)
    imp.reload(params)
    params.result_file = './results/ABM_result_MD_SQ.out'
    params.BCAP_subsidy = 0
    params.BCAP_cost_share = 0
    imp.reload(config)
    imp.reload(gov)
    gov.gov_ref_tax_subsidy[4:7, 2] = 0
    gov.gov_ref_tax_subsidy[4:7, 3] = 0
    gov.ref_subsidys[4:7, 1] = 0
    gov.ref_subsidys[4:7, 2] = 0
    main()
    
