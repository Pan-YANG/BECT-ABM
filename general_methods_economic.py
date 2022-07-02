import agents
import config
import government as gov
import gen_agents
import numpy as np
import general_methods_physical as GMP
import warnings
from scipy.optimize import fsolve
import copy


def forecasts_price(prior, obs, update_rate):

    return (obs ** update_rate) * (prior ** (1 - update_rate))

def cal_learn_by_do(ini_cost,learn_rate,accum_amount):
    rela_amount = accum_amount / config.base_cap_for_learn_by_do
    rela_amount = max(1,rela_amount)

    return ini_cost * (rela_amount**learn_rate)

def cal_learn_by_do_supply_chain(ini_cost,learn_rate,accum_amount):
    rela_amount = accum_amount / config.base_feed_amount_for_learn_by_do
    rela_amount = max(1, rela_amount)

    return ini_cost * (rela_amount ** learn_rate)

def cal_learn_by_do_BCHP(ini_cost,learn_rate,accum_amount):
    rela_amount = accum_amount / config.base_BCHP_cap_for_learn_by_do
    rela_amount = max(1, rela_amount)

    return ini_cost * (rela_amount ** learn_rate)

def cal_PA(discount_rate,N):
    # function to calculate the NPV of uniform annual cash flow, P/A
    # N: the time horizon
    return ((1+discount_rate)**N-1)/(discount_rate*(1+discount_rate)**N)

def cal_AP(discount_rate,N):
    # function to calculate the annual cash flow of a upfront investment, A/P
    # N: the time horizon
    return discount_rate*(1+discount_rate)**N/((1+discount_rate)**N-1)

def cal_profit_exp_patch(patch_ID,land_use_b,land_use_n,slope,peren_age,policy_eligibilities,soil_loss,prices,discount_rate,risks,year):
    # common function to calculate the expected profit of each land use
    # patch_ID: the ID for land patch
    # land_use_b, land_use_n: the previous and new land uses, for land_use_n, 30 stand for the condition that land_use_b==3
    #   and new decision is to start new mis installation; 40 stand for the condition that land_use_b==4 and new decision
    #   is to start new switch installation
    # slope: the slope of land patch
    # peren_age: the age of perennial grass
    # policy_eligibilities: an array showing the eglibility for all policies [BCAP,TMDL,CRP,environmental_concern]
    # soil_loss: the economic loss associated with regular crop
    # discount_rate: the time preference factor for farmer
    # prices: the forecasted prices, [corn, soy, mis, switch, sorghum, cane]
    # risks: an array showing the different risk conditions, [is_flood, is_drought, is_failed]
    # profit: output of the function, for new installation of perennial grass, the profit is calculated as the expected annual cash flow (ACF)

    BCAP_eligible = copy.deepcopy(policy_eligibilities[0])
    TMDL_eligible = copy.deepcopy(policy_eligibilities[1])
    CRP_eligible = copy.deepcopy(policy_eligibilities[2])
    is_environmental = copy.deepcopy(policy_eligibilities[3])

    is_flood = copy.deepcopy(risks[0])
    is_drought = copy.deepcopy(risks[1])
    is_failed = copy.deepcopy(risks[2])

    if land_use_n>10:
        land_use_n_adj = int(land_use_n/10)
        peren_age_adj = 3 # if the decision is to start new perennial grass installation, then calculate the yield at maximum production (at year 4)
    else:
        land_use_n_adj = copy.deepcopy(land_use_n)
        peren_age_adj = max(peren_age,3)

    is_stocha = 0
    output = GMP.look_up_table_crop_no_physical_model(patch_ID,is_flood,is_drought,slope,land_use_b,land_use_n_adj,0,peren_age_adj,1,is_stocha)
    yield_patch = copy.deepcopy(output['yield'])
    ferti_use = copy.deepcopy(output['ferti_use'])
    if ((year % 2) == 1) & (land_use_n_adj == 2): # for corn and soybean rotation, every two years, the price of corn should be taken
        price = copy.deepcopy(prices[0])
    else:
        price = copy.deepcopy(prices[land_use_n_adj - 1])

    fix_cost = copy.deepcopy(config.farmer_cost_table[0, land_use_n_adj-1])
    via_cost = copy.deepcopy(config.farmer_cost_table[1, land_use_n_adj-1])
    ini_cost = copy.deepcopy(config.farmer_cost_table[2, land_use_n_adj-1])

    if land_use_n == 7:
        revenue = TMDL_eligible * gov.TMDL_subsidy
        cost = 0
        profit = (revenue - cost) + is_environmental * soil_loss
    elif land_use_n == 8:
        revenue = gov.CRP_subsidy + TMDL_eligible * gov.TMDL_subsidy
        cost = 0
        profit = (revenue - cost) + is_environmental * soil_loss - config.land_rent * CRP_eligible * config.marginal_land_rent_adj
    elif land_use_n == 1:
        revenue = yield_patch * price
        cost = fix_cost + via_cost * yield_patch + ferti_use * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj
        if config.stover_harvest_cost<prices[2]: # if corn stover harvesting is profitable, the revenue and cost should be updated accordingly
            revenue = revenue + yield_patch * config.stover_harvest_ratio * prices[2]
            cost = cost + yield_patch * config.stover_harvest_ratio * config.stover_harvest_cost
        profit = (revenue - cost)
    elif land_use_n == 2:
        output_rotation_0 = GMP.look_up_table_crop_no_physical_model(patch_ID, is_flood, is_drought, slope, land_use_b,
                                                          land_use_n_adj, 0, peren_age_adj, 1, is_stocha) # estimate the yield of corn and soybean respectively for rotation
        output_rotation_1 = GMP.look_up_table_crop_no_physical_model(patch_ID, is_flood, is_drought, slope, land_use_b,
                                                                     land_use_n_adj, 1, peren_age_adj, 1, is_stocha)
        yield_patch_rotation_0 = copy.deepcopy(output_rotation_0['yield']) # soybean
        yield_patch_rotation_1 = copy.deepcopy(output_rotation_1['yield']) # corn
        revenue = (yield_patch_rotation_0 * prices[1]+yield_patch_rotation_1 * prices[0])/2
        cost_rotation_0 = fix_cost + via_cost * yield_patch_rotation_0 + ferti_use * config.ferti_price + \
                          config.land_rent * CRP_eligible * config.marginal_land_rent_adj
        cost_rotation_1 = config.farmer_cost_table[0, 1] + config.farmer_cost_table[1, 1] * yield_patch_rotation_1 + \
                          config.fertilizer_table[1] * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj
        cost = (cost_rotation_0+cost_rotation_1)/2
        profit = (revenue - cost)
    elif (land_use_n >= 3) & (land_use_n <= 6): # if continue grow energy crop, receive subsidies
        revenue = yield_patch * price + BCAP_eligible * gov.BCAP_subsidy + TMDL_eligible * gov.TMDL_subsidy + \
                               CRP_eligible * gov.CRP_subsidy * gov.CRP_relax + \
                                config.mis_carbon_credit * gov.carbon_price + config.mis_N_credit * gov.nitrogen_price
        cost = fix_cost + via_cost * yield_patch + ferti_use * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj
        profit = (revenue - cost) + is_environmental * soil_loss
    elif land_use_n == 30: # if start new mis installation
        annual_revenue = yield_patch * price + BCAP_eligible * gov.BCAP_subsidy + TMDL_eligible * gov.TMDL_subsidy + \
                         CRP_eligible * gov.CRP_subsidy * gov.CRP_relax + config.mis_carbon_credit * gov.carbon_price + \
                         config.mis_N_credit * gov.nitrogen_price
        annual_cost = fix_cost + via_cost * yield_patch + ferti_use * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj

        half_revenue = config.perennial_yield_adj_table[1, 0] * yield_patch * price + BCAP_eligible * gov.BCAP_subsidy + \
                       TMDL_eligible * gov.TMDL_subsidy + CRP_eligible * gov.CRP_subsidy * gov.CRP_relax + \
                       config.mis_carbon_credit * gov.carbon_price + config.mis_N_credit * gov.nitrogen_price
        half_cost = fix_cost + config.perennial_yield_adj_table[1, 0] * via_cost * yield_patch + \
                    ferti_use * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj
        if is_failed == 0: # if the installation does not fail
            cash_flow = np.zeros(15)
            cash_flow[0] = -ini_cost * (1-BCAP_eligible * gov.BCAP_cost_share)
            cash_flow[1] = (half_revenue - half_cost) / (1+discount_rate)
            cash_flow[2:] = annual_revenue - annual_cost
            NPV = np.npv(discount_rate,cash_flow)
            profit = np.pmt(discount_rate,15,-NPV,0) + is_environmental * soil_loss
        else:
            cash_flow = np.zeros(16)
            cash_flow[0] = -ini_cost * (1 - BCAP_eligible * gov.BCAP_cost_share)
            cash_flow[1] = -ini_cost * (1-BCAP_eligible * gov.BCAP_cost_share) / (1+discount_rate)
            cash_flow[2] = (half_revenue - half_cost) / (1 + discount_rate)**2
            cash_flow[3:] = annual_revenue - annual_cost
            NPV = np.npv(discount_rate, cash_flow)
            profit = np.pmt(discount_rate, 15, -NPV, 0) + is_environmental * soil_loss
    elif land_use_n == 40: # if start new swtich installation
        annual_revenue = yield_patch * price + BCAP_eligible * gov.BCAP_subsidy + TMDL_eligible * gov.TMDL_subsidy + \
                         CRP_eligible * gov.CRP_subsidy * gov.CRP_relax + config.mis_carbon_credit * gov.carbon_price + \
                         config.mis_N_credit * gov.nitrogen_price
        annual_cost = fix_cost + via_cost * yield_patch + ferti_use * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj

        half_revenue = config.perennial_yield_adj_table[1, 0] * yield_patch * price + BCAP_eligible * gov.BCAP_subsidy + \
                       TMDL_eligible * gov.TMDL_subsidy + CRP_eligible * gov.CRP_subsidy * gov.CRP_relax + \
                       config.mis_carbon_credit * gov.carbon_price + config.mis_N_credit * gov.nitrogen_price
        half_cost = fix_cost + config.perennial_yield_adj_table[1, 0] * via_cost * yield_patch + \
                    ferti_use * config.ferti_price + config.land_rent * CRP_eligible * config.marginal_land_rent_adj
        if is_failed == 0: # if the installation does not fail
            cash_flow = np.zeros(10)
            cash_flow[0] = -ini_cost * (1 - BCAP_eligible * gov.BCAP_cost_share)
            cash_flow[1] = (half_revenue - half_cost) / (1 + discount_rate)
            cash_flow[2:] = annual_revenue - annual_cost
            NPV = np.npv(discount_rate, cash_flow)
            profit = np.pmt(discount_rate, 15, -NPV, 0) + is_environmental * soil_loss
        else:
            cash_flow = np.zeros(11)
            cash_flow[0] = -ini_cost * (1 - BCAP_eligible * gov.BCAP_cost_share)
            cash_flow[1] = -ini_cost * (1 - BCAP_eligible * gov.BCAP_cost_share) / (1 + discount_rate)
            cash_flow[2] = (half_revenue - half_cost) / (1 + discount_rate) ** 2
            cash_flow[3:] = annual_revenue - annual_cost
            NPV = np.npv(discount_rate, cash_flow)
            profit = np.pmt(discount_rate, 15, -NPV, 0) + is_environmental * soil_loss
    else:
        warnings.warn('Land use not identified!')

    return profit

def cal_farm_profit_ref_esti(patch_ID,land_use_b,land_use_n,slope,peren_age,policy_eligibilities,soil_loss,prices,discount_rate,risks,year):
    # calculate the refinery estimated farmer's profit on each patch
    # all parameters are the same as cal_profit_exp_patch, expect risks: [flood_prob,drought_prob,fail_prob]
    profit_best = cal_profit_exp_patch(patch_ID,land_use_b,land_use_n,slope,peren_age,policy_eligibilities,soil_loss,prices,discount_rate,[0,0,0],year)
    profit_flood = cal_profit_exp_patch(patch_ID,land_use_b,land_use_n,slope,peren_age,policy_eligibilities,soil_loss,prices,discount_rate,[1,0,0],year)
    profit_drought = cal_profit_exp_patch(patch_ID,land_use_b,land_use_n,slope,peren_age,policy_eligibilities,soil_loss,prices,discount_rate,[0,1,0],year)
    profit_fail = cal_profit_exp_patch(patch_ID,land_use_b,land_use_n,slope,peren_age,policy_eligibilities,soil_loss,prices,discount_rate,[0,0,1],year)
    profit = (1-risks[0]-risks[1]-risks[2]) * profit_best + profit_flood * risks[0] + profit_drought * risks[1] + profit_fail * risks[2]
    return profit

def cal_BEM_patch(farmer_list,N_patch,N_crop,market_prices,empirical_risks,year):
    # function to calculate the break even price matrix for each crop (column) and and land patch (row)
    # farmer_list: a list of all farmer agents
    # N_patch: number of land patches
    # N_crop: number of crop types
    # market_prices: the current market prices of all crops, [corn,soy,mis,switch,sorghum,cane]
    # empirical_risks: the risks estimated from historical average values [P_flood, P_drought, P_fail]
    patch_attribute = np.loadtxt('./data/patch_attributes.csv', delimiter=',', skiprows=1) # column 3 contains the marginal land information
    current_profit_patch = np.zeros((N_patch,N_crop+2))
    current_profit_patch[:,-2] = 0
    current_profit_patch[:,-1] = copy.deepcopy(gov.CRP_subsidy-config.land_rent * patch_attribute[:,3] * config.marginal_land_rent_adj)
    delta_profit_patch = np.zeros((N_patch,N_crop)) # the additional profit available if price is increase by 1
    patch_BEM = np.zeros((N_patch,N_crop)) # patch and crop specific break even prices
    market_prices = np.delete(market_prices,5)

    for i in range(N_crop):
        crop_id = i+1
        for farm in farmer_list:
            farmer_ID = copy.deepcopy(farm.ID)
            patch_IDs = copy.deepcopy(farm.Attributes['patch_ID'])
            N_patch_temp = patch_IDs.__len__()
            Peren_ages = copy.deepcopy(farm.States['peren_age'][-1])
            land_uses_b = copy.deepcopy(farm.States['land_use'][-1])
            slopes = copy.deepcopy(farm.Attributes['patch_slope'])
            TMDL_eligible = copy.deepcopy(farm.States['TMDL_eligible'][-1])  # 1 for eligible, 0 for not
            BCAP_eligible = copy.deepcopy(farm.States['BCAP_eligible'][-1])
            CRP_eligible = copy.deepcopy(farm.Attributes['patch_CRP_eligible'])
            soil_losses = copy.deepcopy(farm.Attributes['soil_loss'])
            for j in range(N_patch_temp):
                land_use_n = copy.deepcopy(crop_id)
                policy_eligibilities = [BCAP_eligible[j], TMDL_eligible[j], CRP_eligible[j], 0]
                if policy_eligibilities[2] == 0:
                    current_profit_patch[patch_IDs[j], -1] = -10**8

                if crop_id == 3:
                    if (Peren_ages[j] >= 15)|(land_uses_b[j]!=3):
                        land_use_n = 30
                if crop_id == 4:
                    if (Peren_ages[j] >= 10)|(land_uses_b[j]!=4):
                        land_use_n = 40

                profit = cal_farm_profit_ref_esti(patch_IDs[j],land_uses_b[j],land_use_n,slopes[j],Peren_ages[j],
                                                  policy_eligibilities,soil_losses[j],market_prices,
                                                  config.refinery_esti_farmer_discount_rate,empirical_risks,year)
                profit_more = cal_farm_profit_ref_esti(patch_IDs[j],land_uses_b[j],land_use_n,slopes[j],Peren_ages[j],
                                                  policy_eligibilities,soil_losses[j],market_prices+1,
                                                  config.refinery_esti_farmer_discount_rate,empirical_risks,year)
                current_profit_patch[patch_IDs[j],crop_id-1] = copy.deepcopy(profit)
                delta_profit_patch[patch_IDs[j],crop_id-1] = profit_more - profit

    for i in range(N_crop):
        for j in range(N_patch):
            original_profit = copy.deepcopy(current_profit_patch[j,i])
            max_profit = current_profit_patch[j,:].max()
            patch_BEM[j,i] = GMP.divide(max_profit - original_profit,delta_profit_patch[j,i]) + market_prices[i]

    # patch_BEM[:,2] = config.stover_harvest_cost # the breakeven price of corn stover is its harvesting cost

    return patch_BEM

def cal_bargain_price(LB,UB,market_demand,market_supply):
    # function to simulate the bargaining process between refinery and farmer
    # LB: lower bound of bidding, which is the larger one of PBA and PM
    # UB: upper bound of bidding, which is the PBE
    # market_demand: the total demand in market
    # market_supply: the total supply in market

    market_tightness = market_demand / (market_demand + market_supply)

    fn = lambda x: LB + (UB-LB)**(market_tightness * x) - UB + (UB-LB)**((1-market_tightness) * x)
    bid_step_1 = fsolve(fn,np.random.rand())
    bid_step_2 = fsolve(fn, np.random.rand())
    bid_step_3 = fsolve(fn, np.random.rand())
    bid_step_4 = fsolve(fn, np.random.rand())
    bid_step_5 = fsolve(fn, np.random.rand())
    ID_best = np.argmin(np.abs(np.asarray([fn(bid_step_1),fn(bid_step_2),fn(bid_step_3),fn(bid_step_4),fn(bid_step_5)])))
    bid_step = [bid_step_1,bid_step_2,bid_step_3,bid_step_4,bid_step_5][ID_best]

    P_bargain = LB + (UB-LB)**(market_tightness * bid_step)

    return P_bargain

def update_market_feed_price(ref_list,farmer_agent_list,feed_price_external,product_prices_pred):
    # function to calculate the market feedstock price based on supply and demend
    # feed_price_external: externally provided feedstock price
    # the product price for refinery
    N_ref = ref_list.__len__()
    PBE = []
    market_demand = []
    market_supply = 0
    for temp_farmer_agent in farmer_agent_list:
        for j in range(temp_farmer_agent.Attributes['patch_ID'].size):
            patch_ID = temp_farmer_agent.Attributes['patch_ID'][j]
            if temp_farmer_agent.States['land_use'][-1][j] == 1:
                market_supply += config.patch_yield_table_mean[patch_ID,1] * config.stover_harvest_ratio \
                                * temp_farmer_agent.Attributes['patch_areas'][j]
            elif temp_farmer_agent.States['land_use'][-1][j] == 3:
                market_supply += config.patch_yield_table_mean[patch_ID, 3] * \
                                 temp_farmer_agent.Attributes['patch_areas'][j]
    for i in range(N_ref):
        if ref_list[i].Attributes['tech_type'] > 1:
            PBE.append(ref_list[i].cal_PBE(product_prices_pred, config.trans_costs, config.storage_costs)[2])
            market_demand.append(ref_list[i].Attributes['feedstock_amount'][2] -
                                 ref_list[i].States['contracted_patch_amount'][-1].sum())
            market_supply -= ref_list[i].States['contracted_patch_amount'][-1].sum()
    LB = max(feed_price_external,config.stover_harvest_cost)
    if PBE.__len__()==0:
        UB = LB+0.000001
    else:
        UB = (np.asarray(PBE) * np.asarray(market_demand)).sum() / np.asarray(market_demand).sum()
    market_demand = np.asarray(market_demand).sum()
    if UB < LB:
        UB=LB+0.000001
    market_supply = max(0,market_supply)
    feed_price = cal_bargain_price(LB, UB, market_demand, market_supply)
    return feed_price, market_demand, market_supply

def feed_price_to_land_use_price(feed_prices):
    # function to convert feedstock price to received price of land use, assuming all the products from the land patch are sold to the same buyer
    land_use_price = np.zeros(8)
    land_use_price[0] = feed_prices[0] + feed_prices[2] * config.stover_harvest_ratio
    land_use_price[1] = feed_prices[1]
    land_use_price[2] = feed_prices[3]
    land_use_price[3] = feed_prices[4]
    land_use_price[4] = feed_prices[6]
    land_use_price[5] = feed_prices[7]
    return land_use_price

def is_match_land_use_tech_type(land_use,tech_type):
    if (tech_type == 1) & (np.isin(land_use,[1])):
        is_match = 1
    elif (np.isin(tech_type,[2,5,6,7])) & (np.isin(land_use,[1,3,4])):
        is_match = 1
    elif (tech_type == 3) & (np.isin(land_use,[5,6])):
        is_match = 1
    elif (tech_type == 4) & (np.isin(land_use,[5,6])):
        is_match = 1
    else:
        is_match = 0
    return is_match

def check_feed_price_affordability(land_use,tech_type,feed_prices,dist,PBE):
    # function to check if PBE <= feed_price+trans_cost
    feed_ID = GMP.land_use_to_feed_ID(land_use,tech_type)
    feed_price = copy.deepcopy(feed_prices[feed_ID])
    PBE = np.append(PBE,PBE[-2:])
    PBE = copy.deepcopy(PBE[feed_ID])
    is_affordable = (PBE + config.patch_influence_range * config.trans_costs[feed_ID] >= feed_price + dist * config.trans_costs[feed_ID]) + 0
    return is_affordable

def check_look_up_table(look_up_table,input):
    ID = np.where(look_up_table[0,:] == input)
    output = copy.deepcopy(look_up_table[1,ID[0]])
    return output

def match_feed_demand_supply(ref_list,farmer_list,PBE,ref_feed_buy,ref_feed_sell,feed_prices):
    # function to match the demand and supply of feedstocks
    N_farmer = farmer_list.__len__()
    N_ref = ref_list.__len__()
    feed_prices_within_boundary = feed_prices + config.trans_costs * (config.patch_influence_range - config.patch_influence_range)

    if ref_feed_buy.sum() == 0:
        for i in range(N_ref):
            ref_list[i].States['purchased_feedstock'].append([np.zeros((1,10)), np.zeros(10)])
            ref_list[i].States['purchased_feed_dist'].append(np.zeros(1))
            ref_list[i].States['sold_feedstock'].append(np.zeros((10, 2)))
            ref_list[i].States['sold_feedstock'][-1][:,1] = ref_feed_sell[:,i]
        for i in range(N_farmer):
            ID_no_contract = np.where(farmer_list[i].States['contract'][-1] == -1)[0]
            N_no_contract = ID_no_contract.__len__()
            for j in range(N_no_contract):
                land_use = farmer_list[i].States['land_use'][-1][ID_no_contract[j]]
                prices = feed_price_to_land_use_price(feed_prices)
                farmer_list[i].Temp['patch_received_prices'][ID_no_contract[j]] = prices[land_use-1]

    else:
        buyer_ID = np.where(ref_feed_buy.sum(0) > 0)[0]
        ref_feed_buy = ref_feed_buy[:,buyer_ID]
        N_buyer = buyer_ID.size
        buyer_loc_ID = np.zeros(N_buyer,int)
        tech_types = np.zeros(N_buyer,int)
        for i in range(N_buyer):
            buyer_loc_ID[i] = ref_list[buyer_ID[i]].Attributes['loc_ID']
            tech_types[i] = ref_list[buyer_ID[i]].Attributes['tech_type']
        dist_seller_buyer = np.empty((0,N_buyer))
        feed_available_ID = np.empty(0)
        for temp_farmer in farmer_list:
            non_contract_land_use = copy.deepcopy(temp_farmer.States['land_use'][-1][temp_farmer.States['contract'][-1]==-1])
            if non_contract_land_use.size == 0:
                continue
            elif np.isin(non_contract_land_use,[1,2,3,4,5,6]).sum() == 0: # all lands are in fallow or CRP
                continue
            else:
                feed_available_ID = np.append(feed_available_ID,temp_farmer.ID)
                temp_idxs = (tech_types < 5) * buyer_loc_ID * 6 + (tech_types > 5) * buyer_loc_ID
                temp_dist_matrix = np.concatenate((config.ref_farmer_dist_matrix,config.BCHP_farmer_dist_matrix,config.cofire_farmer_dist_matrix),axis=0)
                dist_temp = copy.deepcopy(temp_dist_matrix[:,temp_farmer.ID][temp_idxs]) # adjust the buyer_loc_ID to cope with the config.ref_farmer_dist_matrix demension
                dist_seller_buyer = np.vstack((dist_seller_buyer,dist_temp))

        for i in range(N_ref):
            feed_available_ID = np.append(feed_available_ID,i+N_farmer)
            dist_temp = copy.deepcopy(config.ref_ref_dist_matrix[:, ref_list[i].Attributes['loc_ID']][buyer_loc_ID])
            dist_seller_buyer = np.vstack((dist_seller_buyer, dist_temp))

        shortest_dist_ID = np.zeros(dist_seller_buyer.shape)
        total_demand = np.zeros(N_buyer)
        tech_types = np.zeros(N_buyer)
        for i in range(N_buyer):
            shortest_dist_ID[:,i] = np.argsort(dist_seller_buyer[:,i])
            tech_types[i] = copy.deepcopy(ref_list[buyer_ID[i]].Attributes['tech_type'])
            total_demand[i] = GMP.convert_demand_array_to_single_number(ref_feed_buy[:,i],tech_types[i])

        for i in range(N_buyer):
            tech_type = copy.deepcopy(ref_list[buyer_ID[i]].Attributes['tech_type'])
            brought_feed_temp = np.empty((0, 10))
            feed_dist_temp = np.empty(0)
            j = 0
            while (total_demand[i] > 0) & (j < feed_available_ID.size):
                dist_based_ID_temp = int(shortest_dist_ID[j, i])
                sell_ID = int(feed_available_ID[dist_based_ID_temp])
                if sell_ID < N_farmer:
                    N_patch_temp = farmer_list[sell_ID].Attributes['patch_ID'].__len__()
                    for k in range(N_patch_temp):
                        if (farmer_list[sell_ID].Temp['patch_available_for_sale'][k] == 0) & (farmer_list[sell_ID].Temp['stover_available_for_sale'][k] == 0):
                            # j = j + 1
                            continue
                        elif is_match_land_use_tech_type(farmer_list[sell_ID].States['land_use'][-1][k],tech_type) == 0:
                            # j = j + 1
                            continue
                        else:
                            land_use_temp = copy.deepcopy(farmer_list[sell_ID].States['land_use'][-1][k])
                            if check_feed_price_affordability(land_use_temp, tech_type, feed_prices_within_boundary,dist_seller_buyer[dist_based_ID_temp, i], PBE[:,buyer_ID[i]]) == 0:
                                # j = j + 1
                                continue
                            elif (land_use_temp == 1) & (np.isin(tech_type,[2,5,6,7]))&(farmer_list[sell_ID].Temp['stover_available_for_sale'][k] == 0):
                                continue
                            elif (land_use_temp == 1) & (np.isin(tech_type,[2,5,6,7]))&(farmer_list[sell_ID].Temp['stover_available_for_sale'][k] > 0):
                                crop_yield_temp = copy.deepcopy(config.stover_harvest_ratio * farmer_list[sell_ID].States['yield'][-1][k])
                                farmer_list[sell_ID].Temp['stover_available_for_sale'][k] = 0
                            else:
                                crop_yield_temp = copy.deepcopy(farmer_list[sell_ID].States['yield'][-1][k])
                                farmer_list[sell_ID].Temp['patch_available_for_sale'][k] = 0
                            area_temp = copy.deepcopy(farmer_list[sell_ID].Attributes['patch_areas'][k])
                            sell_amount, feed_ID_temp = GMP.convert_production_to_supply(land_use_temp,crop_yield_temp,area_temp,tech_type)
                            total_demand[i] = total_demand[i] - sell_amount
                            temp = np.zeros(10)
                            temp[feed_ID_temp] = crop_yield_temp * area_temp
                            brought_feed_temp = np.vstack((brought_feed_temp,temp))
                            feed_dist_temp = np.append(feed_dist_temp,dist_seller_buyer[dist_based_ID_temp,i])
                            if feed_ID_temp == 2:
                                price_temp = copy.deepcopy(feed_prices_within_boundary[feed_ID_temp])
                            else:
                                price_temp = copy.deepcopy(feed_prices_within_boundary[feed_ID_temp])
                            farmer_list[sell_ID].Temp['patch_received_prices'][k] += price_temp
                    j = j+1
                else:
                    ref_sell_temp = copy.deepcopy(ref_feed_sell[:,sell_ID-N_farmer])
                    sell_amount = GMP.convert_demand_array_to_single_number(ref_sell_temp,tech_type)
                    if sell_amount == 0:
                        j = j + 1
                        continue
                    else:
                        total_demand[i] = total_demand[i] - sell_amount
                        temp = np.zeros(10)
                        if tech_type == 1:
                            temp[0] = copy.deepcopy(ref_sell_temp[0])
                            ref_sell_temp[0] = 0
                        elif tech_type == 2:
                            temp[2:5] = copy.deepcopy(ref_sell_temp[2:5])
                            ref_sell_temp[2:5] = 0
                        elif tech_type == 3:
                            temp[6:8] = copy.deepcopy(ref_sell_temp[6:8])
                            ref_sell_temp[6:8] = 0
                        else:
                            temp[8:10] = copy.deepcopy(ref_sell_temp[8:10])
                            ref_sell_temp[8:10] = 0
                        if temp.sum() == 0:
                            j = j + 1
                            continue
                        else:
                            brought_feed_temp = np.vstack((brought_feed_temp, temp))
                            feed_dist_temp = np.append(feed_dist_temp, dist_seller_buyer[dist_based_ID_temp, i])
                            ref_feed_sell[:, sell_ID-N_farmer] = copy.deepcopy(ref_sell_temp)
                            sold_feed_temp=np.zeros((10,2))
                            sold_feed_temp[:,0] = copy.deepcopy(temp)
                            ref_list[sell_ID - N_farmer].States['sold_feedstock'].append(sold_feed_temp)
                        j = j + 1

            if brought_feed_temp.size == 0:
                brought_feed_temp = np.zeros((1, 10))
                feed_dist_temp = np.zeros(1)
            ref_list[buyer_ID[i]].Temp['purchased_feedstock'] = brought_feed_temp
            ref_list[buyer_ID[i]].States['purchased_feed_dist'].append(feed_dist_temp)

        for i in range(N_ref):
            if (i == buyer_ID).sum() == 0:
                ref_list[i].States['purchased_feedstock'].append([np.zeros((1,10)), np.zeros(10)])
                ref_list[i].States['purchased_feed_dist'].append(np.zeros(1))
                # the refinery does not need to buy feedstock
            elif total_demand[np.argwhere(i==buyer_ID)[0][0]] <= 0:
                # ref_ID_temp = np.argwhere(i==buyer_ID)[0][0]
                temp_amount_out = np.zeros(10)
                temp_amount_in = copy.deepcopy(ref_list[i].Temp['purchased_feedstock'])
                temp = [temp_amount_in, temp_amount_out]
                ref_list[i].States['purchased_feedstock'].append(temp)
            else:
                if ref_list[i].Attributes['tech_type'] == 1:
                    feed_ID_temp = 0
                elif ref_list[i].Attributes['tech_type'] == 2:
                    feed_ID_temp = 3
                else:
                    feed_ID_temp = 6

                # ref_ID_temp = np.argwhere(i == buyer_ID)[0][0]
                if feed_prices[feed_ID_temp] + config.trans_costs[feed_ID_temp] * config.system_boundary_radius > PBE[feed_ID_temp,i]:
                    temp_amount_out = np.zeros(10)
                    temp_amount_in = copy.deepcopy(ref_list[i].Temp['purchased_feedstock'])
                    temp = [temp_amount_in, temp_amount_out]
                    ref_list[i].States['purchased_feedstock'].append(temp)
                    # the feedstock outside the watershed is too expensive
                else:
                    temp_amount_out = np.zeros(10)
                    temp_amount_out[feed_ID_temp] = copy.deepcopy(total_demand[np.argwhere(i==buyer_ID)[0][0]])
                    temp_amount_in = copy.deepcopy(ref_list[i].Temp['purchased_feedstock'])
                    temp = [temp_amount_in ,temp_amount_out]
                    ref_list[i].States['purchased_feedstock'].append(temp)

            if ref_feed_sell[:,i].sum() == 0:
                sold_feed_temp = np.zeros((10, 2))
                ref_list[i].States['sold_feedstock'].append(sold_feed_temp)
                # if there is no available feedstock for sell
            else:
                sold_feed_temp = np.zeros((10, 2))
                sold_feed_temp[:, 1] = copy.deepcopy(ref_feed_sell[:,i])
                ref_list[i].States['sold_feedstock'].append(sold_feed_temp)

        loop_up_table_temp = np.asarray([[1,2,3,4,5,6],[0,1,3,4,6,7]])
        for i in range(N_farmer):
            N_patch = farmer_list[i].Attributes['patch_ID'].size
            for j in range(N_patch):
                if farmer_list[i].Temp['patch_available_for_sale'][j] == 0:
                    pass
                else:
                    farmer_list[i].Temp['patch_received_prices'][j] += feed_prices[check_look_up_table(loop_up_table_temp, farmer_list[i].States['land_use'][-1][j])]

                if farmer_list[i].Temp['stover_available_for_sale'][j] == 0:
                    pass
                else:
                    farmer_list[i].Temp['patch_received_prices'][j] += feed_prices[2] * config.stover_harvest_ratio


        # temp_var_for_stop_condition = np.ones(N_buyer)
        # while (total_demand.sum()>0) & (temp_var_for_stop_condition.sum()==0):
        #     buyer_ID_temp = np.random.choice(np.where(temp_var_for_stop_condition==1)[0],1)
        #     j = 0
        #     while j < feed_available_ID.size:
        #         dist_based_ID_temp = shortest_dist_ID[j,buyer_ID_temp]
        #         sell_ID = feed_available_ID[dist_based_ID_temp]
        #         brought_feed_temp = np.empty((0,10))
        #         feed_dist_temp = np.empty(0)
        #         if sell_ID < N_farmer:
        #             N_patch_temp = farmer_list[sell_ID].Attributes['patch_ID'].__len__()
        #             for k in range(N_patch_temp):
        #                 if farmer_list[sell_ID].Temp['patch_available_for_sale'][k] == 0:
        #                     continue
        #                 elif farmer_list[sell_ID].States['contract'][-1][k] ==1:
        #                     farmer_list[sell_ID].Temp['patch_available_for_sale'][k] = 0
        #                 elif (farmer_list[sell_ID].States['land_use'][-1][k] == 7) | (farmer_list[sell_ID].States['land_use'][-1][k] == 8):
        #                     farmer_list[sell_ID].Temp['patch_available_for_sale'][k] = 0
        #                 elif is_match_land_use_tech_type(farmer_list[sell_ID].States['land_use'][-1][k],
        #                                              ref_list[buyer_ID[buyer_ID_temp]].Attributes['tech_type'] == 0):
        #                     continue
        #                 elif (farmer_list[sell_ID].Temp['stover_available_for_sale'][k] == 0) & (ref_list[buyer_ID[buyer_ID_temp]].Attributes['tech_type'] == 1):
        #                     continue
        #                 else:
        #                     land_use_temp = farmer_list[sell_ID].States['land_use'][-1][k]
        #                     tech_type_temp = ref_list[buyer_ID[buyer_ID_temp]].Attributes['tech_type']
        #                     if (land_use_temp == 1) & (tech_type_temp == 2):
        #                         crop_yield_temp = config.stover_harvest_ratio * farmer_list[sell_ID].States['yield'][-1][k]
        #                         farmer_list[sell_ID].Temp['stover_available_for_sale'][k] = 0
        #                     else:
        #                         crop_yield_temp = farmer_list[sell_ID].States['yield'][-1][k]
        #                         farmer_list[sell_ID].Temp['patch_available_for_sale'][k] = 0
        #                     area_temp = farmer_list[sell_ID].Attributes['patch_areas'][-1][k]
        #                     sell_amount, feed_ID_temp = GMP.convert_production_to_supply(land_use_temp,crop_yield_temp,area_temp,tech_type_temp)
        #                     total_demand[buyer_ID_temp] = total_demand[buyer_ID_temp] - sell_amount
        #                     temp = np.zeros(10)
        #                     temp[feed_ID_temp] = crop_yield_temp * area_temp
        #                     brought_feed_temp = np.vstack((brought_feed_temp,temp))
        #                     feed_dist_temp = np.append(feed_dist_temp,dist_seller_buyer[dist_based_ID_temp,buyer_ID_temp])








