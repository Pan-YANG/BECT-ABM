import numpy as np
import scipy.optimize as optimize
import general_methods_physical as GMP
import general_methods_economic as GME
import config
import government as gov
import warnings
from mortgage import Loan
import copy
import pandas as pd
from scipy.stats import rankdata

survey_data_loc = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/BN_rule/'
survey_data = pd.read_csv(survey_data_loc+'BN_data_start1_adj.csv')
bn_df=pd.read_csv(survey_data_loc+'BN_df.csv') # import the MC samples for BN inferencing

class Farmer:
    def __init__(self, ID, Attributes, States, Temp):

        self.ID = ID
        self.Attributes = Attributes
        self.States = States
        self.Temp = Temp

    def forecasts_price(self,prior,obs):

        update_rate = self.Attributes['price_update_rate']
        return (obs**update_rate) * (prior**(1-update_rate))

    def forecasts_climate(self,prior,obs):

        update_rate = self.Attributes['flood_update_rate']

        part_1 = update_rate*prior*obs/(update_rate*prior+(1-update_rate)*(1-prior))
        part_2 = (1-update_rate)*prior*(1-obs)/((1-update_rate)*prior+update_rate*(1-prior))

        return part_1 + part_2

    def cal_peer_ec(self,farmer_list):
        # function to calculate the percentage of neighbors already growing perennial grass
        neighbor_ID = self.Attributes['neighbors']
        N_neighbor = neighbor_ID.size
        adopted = 0
        for i in range(N_neighbor):
            if ((farmer_list[neighbor_ID[i]].States['land_use'][-1] > 2) & (farmer_list[neighbor_ID[i]].States['land_use'][-1] < 7)).sum()>0:
                adopted += 1

        peer_ec_num = adopted/N_neighbor
        if peer_ec_num<0.1:
            peer_ec=1
        elif peer_ec_num<0.2:
            peer_ec=2
        else:
            peer_ec=3
        self.States['peer_ec'].append(peer_ec)

    def cal_PT_imp_env(self):
        # calculate the probability table of imp_env
        df_temp = bn_df[(bn_df['farm_size'] == self.Attributes['farm_size']) &
                        (bn_df['info_use'] == self.Attributes['info_use']) &
                        (bn_df['benefit'] == self.Attributes['benefit']) &
                        (bn_df['concern'] == self.Attributes['concern']) &
                        (bn_df['lql'] == self.Attributes['lql'])]
        PT_imp_env=np.zeros(3)
        PT_imp_env[0] = (df_temp['imp_env']==1).sum()
        PT_imp_env[1] = (df_temp['imp_env'] == 2).sum()
        PT_imp_env[2] = (df_temp['imp_env'] == 3).sum()
        PT_imp_env = GMP.divide(PT_imp_env,PT_imp_env.sum())
        self.Attributes['PT_imp_env'] = PT_imp_env

    def update_environ_sen(self,ini_sen, min_sen, max_sen, theta, AT_commu, sign_AT):
        # function to update farmer's environmental sensitivity
        if sign_AT == 1: # if AT_commu increases
            environ_sen = ini_sen + theta * AT_commu * (1 - ini_sen/max_sen)
        elif sign_AT == 0: # if AT_commu decreases
            environ_sen = ini_sen - theta * AT_commu * (1- min_sen/ini_sen)
        else:
            warnings.warn('sign_AT can only be 0 or 1')

        return min(max(environ_sen,0),max_sen)

    def update_crop_no_physical_model(self,prcp):
        # function to update the actual yield, N release, flooding, drought condition of each land patch
        # prcp: the annual precipitation
        # yield: the actual yield of each land patch
        # N_release: the N release of each land patch

        is_flood = prcp >= 1000  # annual precipitation larger than 1000 will result in flooding
        is_drought = prcp <= 200  # annual precipitation smaller than 200 will result in drought

        patch_IDs = self.Attributes['patch_ID']
        N_patch = patch_IDs.size
        land_uses_b = self.States['land_use'][-2] # the land use in previous year
        land_uses_n = self.States['land_use'][-1] # the land use in current year
        Peren_age = self.States['peren_age'][-1] # the age of perennial grass in current year, this function update the age of perennial grass
        # the age of perennial grass could be -2 (if not growing perennial grass), -1 (if installation failed), or 0-20 (if installation is successful)
        slopes = self.Attributes['patch_slope']

        yield_patch = np.zeros(N_patch)
        N_release = np.zeros(N_patch)
        Peren_age_new = copy.deepcopy(Peren_age) # the perennial grass age will repeat itself if not growing anything (i.e., equal to -2)
        ferti_use = np.zeros(N_patch)

        is_stocha = 1
        year = self.States['land_use'].__len__()

        for idx in range(N_patch):
            output = GMP.look_up_table_crop_no_physical_model(patch_IDs[idx],is_flood,is_drought,slopes[idx],land_uses_b[idx],
                                                              land_uses_n[idx],0,Peren_age[idx],year,is_stocha)
            yield_patch[idx] = output['yield']
            N_release[idx] = output['N_release']
            Peren_age_new[idx] = output['peren_age']
            ferti_use[idx] = output['ferti_use']

        self.States['yield'].append(yield_patch)
        self.States['N_release'].append(N_release)
        self.States['peren_age'].append(Peren_age_new.astype(int))
        self.States['ferti_use'].append(ferti_use)

    def update_profit(self):
        # function to update the actual profit after the land use decision

        # identify the crops with positive acreages for high and low quality lands

        patch_IDs = self.Attributes['patch_ID']
        N_patch = patch_IDs.size
        land_uses = self.States['land_use'][-1]  # the land use in current year
        yields_patch = self.States['yield'][-1]
        prices = self.States['price_received'][-1]
        Peren_ages = self.States['peren_age'][-2] # since the peren_ages have already been updated in update_crop_no_physical_model,
        ferti_uses = self.States['ferti_use'][-1]

        TMDL_eligible = self.States['TMDL_eligible'][-1] # 1 for eligible, 0 for not
        BCAP_eligible = self.States['BCAP_eligible'][-1]
        CRP_eligible = self.Attributes['patch_CRP_eligible']
        # we need to retrieve the peren_age before update to calculate the installation cost
        revenue = np.zeros(N_patch)
        cost = np.zeros(N_patch)

        for idx in range(N_patch):
            land_use = int(land_uses[idx])
            if land_use == 7:
                revenue[idx] = TMDL_eligible[idx] * gov.TMDL_subsidy
                cost[idx] = 0
            elif land_use == 8:
                revenue[idx] = gov.CRP_subsidy + TMDL_eligible[idx] * gov.TMDL_subsidy
                cost[idx] = config.land_rent * self.Attributes['patch_CRP_eligible'][idx] * config.marginal_land_rent_adj
            else:
                yield_patch = yields_patch[idx]
                price = prices[land_use-1]
                fix_cost = config.farmer_cost_table[0,land_use]
                via_cost = config.farmer_cost_table[1,land_use]
                ini_cost = config.farmer_cost_table[2,land_use]
                if (Peren_ages[idx] == -1) | (Peren_ages[idx]==0):
                    is_install = 1
                else:
                    is_install = 0
                # account for the BCAP, TMDL, and possible CRP subsidy when calculating the revenue
                revenue[idx] = yield_patch * price + BCAP_eligible[idx] * gov.BCAP_subsidy * ((land_use>=3) & (land_use<=6)) + \
                               TMDL_eligible[idx] * gov.TMDL_subsidy * ((land_use>=3) & (land_use<=8)) + \
                               CRP_eligible[idx] * gov.CRP_subsidy * gov.CRP_relax * ((land_use>=3) & (land_use<=6))
                # account for the BCAP cost share when calculating the cost
                cost[idx] = fix_cost + via_cost * yield_patch + ferti_uses[idx] * config.ferti_price + ini_cost * is_install * (1-BCAP_eligible[idx] * gov.BCAP_cost_share) + \
                            config.land_rent * self.Attributes['patch_CRP_eligible'][idx] * config.marginal_land_rent_adj
                if (land_use == 1) & (config.stover_harvest_cost<prices[2]): # if corn stover harvesting is profitable, the revenue and cost should be updated accordingly
                    revenue[idx] = revenue[idx] + yield_patch * config.stover_harvest_ratio * prices[2]
                    cost[idx] = cost[idx] + yield_patch * config.stover_harvest_ratio * config.stover_harvest_cost

        profit = (revenue - cost)*self.Attributes['patch_areas']
        self.States['profit'].append(profit)

    def cal_utility_annual_crop(self,land_use,patch_ID,land_use_b,climates,slope,area,prices,peren_age,policy_eligibilities,soil_loss,
                                discount_rate,risk_aver,loss_aver,year):
        # function to calculate the utility of annual crop
        # see cal_max_potential_U_crop for the variable definitions

        profit_normal = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, peren_age, policy_eligibilities,soil_loss,
                                                      prices, discount_rate, [0, 0, 0],year)
        profit_flood = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, peren_age, policy_eligibilities,soil_loss,
                                                     prices, discount_rate, [1, 0, 0],year)
        profit_drought = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, peren_age, policy_eligibilities,soil_loss,
                                                       prices, discount_rate, [0, 1, 0],year)
        if profit_normal < 0:
            U_C_N = -(1 - climates[0] - climates[1]) * (-profit_normal * area) ** risk_aver
        else:
            U_C_N = (1 - climates[0] - climates[1]) * (profit_normal * area) ** risk_aver  # utility of crop under normal condition

        if profit_flood < 0:  # utility of crop under flood condition
            U_C_F = -climates[0] * loss_aver * (-profit_flood * area) ** risk_aver  # consider loss aversion
        else:
            U_C_F = climates[0] * (profit_flood * area) ** risk_aver

        if profit_drought < 0:  # utility of crop under drought condition
            U_C_D = -climates[1] * loss_aver * (-profit_drought * area) ** risk_aver
        else:
            U_C_D = climates[1] * (profit_drought * area) ** risk_aver
        Utility = U_C_N + U_C_F + U_C_D  # utitlity of crop

        return Utility

    def cal_utility_perennial(self,land_use,patch_ID,land_use_b,slope,area,prices,peren_age,policy_eligibilities,soil_loss,
                                discount_rate,risk_aver,loss_aver,year):

        if land_use == 30: # identify the installation failure probability
            profit_normal = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, -2*np.ones(peren_age.shape,int),
                                                     policy_eligibilities, soil_loss,
                                                     prices, discount_rate, [0, 0, 0],year)
            profit_fail = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, -2*np.ones(peren_age.shape,int),
                                                   policy_eligibilities, soil_loss,
                                                   prices, discount_rate, [0, 0, 1],year)
            p_fail = copy.deepcopy(config.install_fail_rate_mis)
        elif land_use == 40:
            profit_normal = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, -2*np.ones(peren_age.shape,int),
                                                     policy_eligibilities, soil_loss,
                                                     prices, discount_rate, [0, 0, 0],year)
            profit_fail = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, -2*np.ones(peren_age.shape,int),
                                                   policy_eligibilities, soil_loss,
                                                   prices, discount_rate, [0, 0, 1],year)
            p_fail = copy.deepcopy(config.install_fail_rate_switch)
        else:
            profit_normal = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, peren_age,
                                                     policy_eligibilities, soil_loss,
                                                     prices, discount_rate, [0, 0, 0],year)
            profit_fail = GME.cal_profit_exp_patch(patch_ID, land_use_b, land_use, slope, peren_age,
                                                   policy_eligibilities, soil_loss,
                                                   prices, discount_rate, [0, 0, 1],year)
            p_fail = 0

        if profit_normal < 0:
            U_P_N = -(1-p_fail) * loss_aver * (-profit_normal * area) ** risk_aver
        else:
            U_P_N = (1-p_fail) * (profit_normal * area) ** risk_aver

        if profit_fail < 0:
            U_P_F = -p_fail * loss_aver * (-profit_fail * area) ** risk_aver
        else:
            U_P_F = p_fail * (profit_fail * area) ** risk_aver

        Utility = U_P_F + U_P_N # utitlity of grass

        return Utility

    def cal_max_potential_U_crop(self,idx,land_use_b,climates,prices,peren_age,policy_eligibilities,year):
        # function to calculate the maximum utility of crop for a particular land patch under planning stage
        # idx: the land patch indice for the particular farmer
        # land_use_b: the previous year land use
        # climates: farmer's estimate of flooding and drought probabilities
        # peren_age: the age of perennial grass, if any
        # policy_eligiblities: eligibilities for BCAP, TMDL,and CRP
        # output: a list showing the best crop and the utility of that crop

        patch_ID = self.Attributes['patch_ID'][idx]
        slope = self.Attributes['patch_slope'][idx]
        area = self.Attributes['patch_areas'][idx]
        discount_rate = self.Attributes['discount_factor']
        risk_aver = self.Attributes['risk_factor']
        loss_aver = self.Attributes['loss_factor']

        U_corn = self.cal_utility_annual_crop(1,patch_ID,land_use_b,climates,slope,area,prices,peren_age,policy_eligibilities,0, # for regular crop, no soil_loss in calculation
                                discount_rate,risk_aver,loss_aver,year) # utitlity of corn
        U_soy = self.cal_utility_annual_crop(2,patch_ID,land_use_b,climates,slope,area,prices,peren_age,policy_eligibilities,0, # for regular crop, no soil_loss in calculation
                                discount_rate,risk_aver,loss_aver,year) # utility of soy

        if U_corn > U_soy:
            best_crop = 1
            U_max_crop = copy.deepcopy(U_corn)
        else:
            best_crop = 2
            U_max_crop = copy.deepcopy(U_soy)

        return best_crop,U_max_crop

    def cal_max_potential_U_peren(self,idx,land_use_b,climates,prices,peren_age,policy_eligibilities,contract_feed_type,
                                  contract_max_price,year):
        # is_environmental is a binary variable representing if environmental concern is included

        patch_ID = self.Attributes['patch_ID'][idx]
        slope = self.Attributes['patch_slope'][idx]
        area = self.Attributes['patch_areas'][idx]
        discount_rate = self.Attributes['discount_factor']
        risk_aver = self.Attributes['risk_factor']
        loss_aver = self.Attributes['loss_factor']
        soil_loss = self.Attributes['soil_loss'][idx] # soil erosion loss for regular crop will be added to the profit of perennial grass

        is_environmental = policy_eligibilities[3]

        if is_environmental == 0:
            soil_loss = 0 # if environmental aspect is not considered, then soil erosion causes no loss

        U_peren = -10**6*np.ones(6) # the utilities of all perennial grass options, land uses: [3,4,5,6,30,40]

        if land_use_b == 3: # this is assuming farmer not making contract with refinery
            profit = GME.cal_profit_exp_patch(patch_ID, land_use_b, 3, slope, peren_age,
                                                         policy_eligibilities, soil_loss, # since this function is only used for economic evaluation, soil_loss is not taken into account.
                                                         prices, discount_rate, [0, 0, 0],year)
            if profit < 0:
                U_peren[0] = -(-profit * area) ** risk_aver
            else:
                U_peren[0] = (profit * area) ** risk_aver  # utility of crop under normal condition
            U_peren[1] = -10**6 # when previous land use is mis, next land use of 4 is not fessible
        elif land_use_b == 4:
            profit = GME.cal_profit_exp_patch(patch_ID, land_use_b, 4, slope, peren_age,
                                              policy_eligibilities,soil_loss, # since this function is only used for economic evaluation, soil_loss is not taken into account.
                                              prices, discount_rate, [0, 0, 0],year)
            U_peren[0] = -10**6  # when previous land use is mis, next land use of 4 is not fessible
            if profit < 0:
                U_peren[1] = -(-profit * area) ** risk_aver
            else:
                U_peren[1] = (profit * area) ** risk_aver  # utility of crop under normal condition

        if contract_feed_type == 1: # mis
            prices_new = copy.deepcopy(prices)
            prices_new[2] = contract_max_price
            prices_new[3] = contract_max_price # mis and swtich share the same price
            U_peren[4] = self.cal_utility_perennial(30, patch_ID, land_use_b, slope, area, prices_new, -2*np.ones(peren_age.shape,int),
                                                    policy_eligibilities,soil_loss,
                                                    discount_rate, risk_aver, loss_aver,year)  # utility of new mis install
            U_peren[5] = self.cal_utility_perennial(40, patch_ID, land_use_b, slope, area, prices_new, -2*np.ones(peren_age.shape,int),
                                                    policy_eligibilities,soil_loss,
                                                    discount_rate, risk_aver,
                                                    loss_aver,year)  # utility of new switch install
        elif contract_feed_type == 2: # sorghum
            prices_new = copy.deepcopy(prices)
            prices_new[4] = contract_max_price
            U_peren[2] = self.cal_utility_annual_crop(5,patch_ID,land_use_b,climates,slope,area,prices_new,peren_age,policy_eligibilities,soil_loss,
                                discount_rate,risk_aver,loss_aver,year) # utitlity of sorghum
        elif contract_feed_type == 3: # cane
            prices_new = copy.deepcopy(prices)
            prices_new[5] = contract_max_price
            U_peren[3] = self.cal_utility_annual_crop(6, patch_ID, land_use_b, climates, slope, area, prices_new, peren_age,
                                                  policy_eligibilities,soil_loss,discount_rate, risk_aver, loss_aver,year)  # utitlity of cane
        elif contract_feed_type == -1: # if there is no contract
            U_peren[2] = self.cal_utility_annual_crop(5, patch_ID, land_use_b, climates, slope, area, prices,
                                                      peren_age, policy_eligibilities,soil_loss,
                                                      discount_rate, risk_aver, loss_aver,year)  # utitlity of sorghum
            U_peren[3] = self.cal_utility_annual_crop(6, patch_ID, land_use_b, climates, slope, area, prices,
                                                      peren_age,
                                                      policy_eligibilities,soil_loss, discount_rate, risk_aver,
                                                      loss_aver,year)  # utitlity of cane
            U_peren[4] = self.cal_utility_perennial(30, patch_ID, land_use_b, slope, area, prices, -2*np.ones(peren_age.shape,int),
                                                    policy_eligibilities,soil_loss,
                                                    discount_rate, risk_aver, loss_aver,year)  # utility of new mis install
            U_peren[5] = self.cal_utility_perennial(40, patch_ID, land_use_b, slope, area, prices, -2*np.ones(peren_age.shape,int),
                                                    policy_eligibilities,soil_loss,
                                                    discount_rate, risk_aver,
                                                    loss_aver,year)  # utility of new switch install

        max_U = np.max(U_peren)
        argmax_U = np.argmax(U_peren)

        return argmax_U,max_U

    def cal_max_potential_profit_peren(self,prices,contract_feed_type,contract_max_price,year):
        # function to calculate the maximum profit among available perennial grasses, and the type of perennial grass should be selected

        patch_ID = self.Attributes['patch_ID']
        slope = self.Attributes['patch_slope']
        area = self.Attributes['patch_areas']
        discount_rate = self.Attributes['discount_factor']
        TMDL_eligible = self.States['TMDL_eligible'][-1]  # 1 for eligible, 0 for not
        BCAP_eligible = self.States['BCAP_eligible'][-1]
        CRP_eligible = self.Attributes['patch_CRP_eligible']
        soil_loss = self.Attributes['soil_loss']
        peer_ec = self.States['peer_ec'][-1]
        if peer_ec == 1:
            environ_factor = config.enhanced_neighbor_impact[0]
        elif peer_ec == 2:
            environ_factor = config.enhanced_neighbor_impact[1]
        else:
            environ_factor = config.enhanced_neighbor_impact[2]

        if contract_feed_type == 1:
            prices[2:4] = contract_max_price
        elif contract_feed_type == 2:
            prices[4] = contract_max_price
        elif contract_feed_type == 3:
            prices[5] = contract_max_price

        profit_peren = -2000*np.ones(4) # the utilities of all perennial grass options, land uses: [3,4,5,6]

        for i in range(patch_ID.size):
            policy_eligibilities = [BCAP_eligible[i], TMDL_eligible[i], CRP_eligible[i], environ_factor]
            profit_peren[0] = max(profit_peren[0],GME.cal_profit_exp_patch(patch_ID[i], 3, 30, slope, 0,
                                                        policy_eligibilities, soil_loss[i],
                                                        prices, discount_rate, [0, 0, 0], year))
            profit_peren[1] = max(profit_peren[1],GME.cal_profit_exp_patch(patch_ID[i], 4, 40, slope, 0,
                                                        policy_eligibilities, soil_loss[i],
                                                        prices, discount_rate, [0, 0, 0], year))
            profit_peren[2] = max(profit_peren[2],GME.cal_profit_exp_patch(patch_ID[i], 5, 5, slope, 0,
                                                        policy_eligibilities, soil_loss[i],
                                                        prices, discount_rate, [0, 0, 0], year))
            profit_peren[3] = max(profit_peren[3],GME.cal_profit_exp_patch(patch_ID[i], 6, 6, slope, 0,
                                                        policy_eligibilities, soil_loss[i],
                                                        prices, discount_rate, [0, 0, 0], year))
            # profit_peren[0] += GME.cal_profit_exp_patch(patch_ID[i], 3, 30, slope, 0,
            #                                              policy_eligibilities, soil_loss[i],
            #                                              prices, discount_rate, [0, 0, 0],year) * area[i]
            # profit_peren[1] += GME.cal_profit_exp_patch(patch_ID[i], 4, 40, slope, 0,
            #                                             policy_eligibilities, soil_loss[i],
            #                                             prices, discount_rate, [0, 0, 0],year) * area[i]
            # profit_peren[2] += GME.cal_profit_exp_patch(patch_ID[i], 5, 5, slope, 0,
            #                                             policy_eligibilities, soil_loss[i],
            #                                             prices, discount_rate, [0, 0, 0],year) * area[i]
            # profit_peren[3] += GME.cal_profit_exp_patch(patch_ID[i], 6, 6, slope, 0,
            #                                             policy_eligibilities, soil_loss[i],
            #                                             prices, discount_rate, [0, 0, 0],year) * area[i]

        # profit_peren = profit_peren/area.sum()

        max_profit = np.max(profit_peren) + config.mis_carbon_credit * gov.carbon_price + config.mis_N_credit * gov.nitrogen_price
        argmax_profit = np.argmax(profit_peren)

        return max_profit,argmax_profit+3

    def cal_adopt_priority(self,):
        # function to identify the priority of adopting perennial grass based on profitability
        patch_ids = self.Attributes['patch_ID']
        crop_yield_table = config.patch_yield_table_mean[patch_ids,:]
        corn_yields = crop_yield_table[:,1]
        mis_yields = crop_yield_table[:,3]
        mis_to_corn = mis_yields/corn_yields
        adopt_priority = patch_ids.size - rankdata(mis_to_corn,method='ordinal')
        self.Attributes['adopt_priority'] = adopt_priority

    def identify_contract_patches(self,community_list,contract_feed_type,contract_max_price,year):
        # function to identify the land patches that farmer will make contract
        # contract_feed_type: the feedstock type that will be contracted, 1 for mis,swtich, or stover, 2 for sorghum, 3 for cane
        # contract_max_price: the maximum price that refinery could offer
        # output is the array showing if the farmer is willing to make contract for the land patches
        patch_IDs = self.Attributes['patch_ID']
        N_patch = patch_IDs.size

        exist_contract = self.States['contract'][-1] # the existing contract, -1 represent no contract
        temp_contract = self.Temp['contract_land_use'] > 2
        # exist_contract_adj = exist_contract + 2 * temp_contract # temp_contract is the contracts made during the negotiation step, before contract state is updated

        land_uses_b = self.States['land_use'][-1]  # the land use in previous year; at this time the land use has not been updated
        prices = copy.deepcopy(self.States['price_prior'][-1])  # farmer's forecast of crop prices
        climates = self.States['climate_forecasts'][-1]  # farmer's forecast of climate
        Peren_ages = self.States['peren_age'][-1]  # the age of perennial grass in the previous year
        TMDL_eligible = self.States['TMDL_eligible'][-1]  # 1 for eligible, 0 for not
        BCAP_eligible = self.States['BCAP_eligible'][-1]
        CRP_eligible = self.Attributes['patch_CRP_eligible']
        peer_ec = self.States['peer_ec'][-1]
        if peer_ec == 1:
            environ_factor = config.enhanced_neighbor_impact[0]
        elif peer_ec == 2:
            environ_factor = config.enhanced_neighbor_impact[1]
        else:
            environ_factor = config.enhanced_neighbor_impact[2]

        is_contract = np.zeros(N_patch,int)
        opt_land_use = np.zeros(N_patch,int) # this is only to calculate the optimal perennial grass

        AT_commu = community_list[self.Attributes['community_ID']].States['attitude'][-1]
        if community_list[self.Attributes['community_ID']].States['attitude'].__len__() == 1:
            sign_AT = 1
        elif community_list[self.Attributes['community_ID']].States['attitude'][-1] >= community_list[self.Attributes['community_ID']].States['attitude'][-2]:
            sign_AT = 1
        else:
            sign_AT = 0

        SC_Rev_mean_num, opt_peren = self.cal_max_potential_profit_peren(prices,contract_feed_type,contract_max_price,year)
        if SC_Rev_mean_num < 150*2.47:
            SC_Rev_mean = 1
        elif SC_Rev_mean_num < 250*2.47:
            SC_Rev_mean = 2
        else:
            SC_Rev_mean = 3

        peren_lookup_table = [3, 4, 5, 6, 30, 40]

        opt_peren = opt_peren*10
        self.bn_inference(AT_commu, sign_AT, 2, SC_Rev_mean, year)

        land_use_exist = self.States['land_use'][-1]  # the existing land use
        is_peren_exist = ((land_use_exist > 2) & (land_use_exist < 7)) | (land_use_exist > 10)
        area_peren_exist = self.Attributes['patch_areas'][is_peren_exist].sum() / self.Attributes['patch_areas'].sum()

        # check willingness to adopt perennial grass based on BN
        is_adopt = self.States['is_adopt'][-1]
        if SC_Rev_mean_num<0:
            is_adopt = 0
        SC_ratio_discrete = self.States['SC_Ratio'][-1]
        if (SC_ratio_discrete == 1) | (is_adopt == 0):
            SC_ratio = 0
        elif SC_ratio_discrete == 2:
            SC_ratio = 0.05
        elif SC_ratio_discrete == 3:
            SC_ratio = 0.1
        elif SC_ratio_discrete == 4:
            SC_ratio = 0.15
        else:
            SC_ratio = 0.3

        for i in self.Attributes['adopt_priority']:
            if (exist_contract[i]>0) & (Peren_ages[i]<GMP.cal_contract_length(land_uses_b[i])):
                is_contract[i] = exist_contract[i]
                opt_land_use[i] = land_uses_b[i]
            elif temp_contract[i]>0:
                is_contract[i] = 1
                opt_land_use[i] = self.Temp['contract_land_use'][i]
                continue
            else:
                land_use_b = land_uses_b[i]
                policy_eligibilities = [BCAP_eligible[i], TMDL_eligible[i], CRP_eligible[i],environ_factor]
                best_crop, U_max_crop = self.cal_max_potential_U_crop(i, land_use_b, climates, prices,
                                                                      Peren_ages[i],
                                                                      policy_eligibilities,year)

                argmax_U_peren, max_U_peren = self.cal_max_potential_U_peren(i, land_use_b, climates, prices,
                                                                             Peren_ages[i], policy_eligibilities,
                                                                             contract_feed_type, contract_max_price,year)

                is_contract[i] = 0
                opt_land_use[i] = best_crop
                if (SC_ratio>area_peren_exist)&(max_U_peren>0):
                    is_contract[i] = 1
                    opt_land_use[i] = opt_peren
                    area_peren_exist += self.Attributes['patch_areas'][i]/self.Attributes['patch_areas'].sum()

                if self.Attributes['type'] in [1,2]: # Type I and II farmers allow adoption of perennial crop when it is profitable, the other two types do not
                    if (argmax_U_peren > 1) & (max_U_peren > 10.*U_max_crop):
                        is_contract[i] = 1
                        opt_land_use[i] = peren_lookup_table[int(argmax_U_peren)]
                    elif max_U_peren > 10.*U_max_crop:
                        opt_land_use[i] = peren_lookup_table[int(argmax_U_peren)]

        return is_contract, opt_land_use

    def cal_utility_perennial_series(self,ID_contract,land_uses,price,year):
        # function to calculate the total Utitlity value of land patches for the function cal_peren_break_even_price

        N = ID_contract.size
        land_uses_b = self.States['land_use'][-1]  # the land use in previous year; at this time the land use has not been updated
        Peren_ages = self.States['peren_age'][-1]  # the age of perennial grass in the previous year
        TMDL_eligible = self.States['TMDL_eligible'][-1]  # 1 for eligible, 0 for not
        BCAP_eligible = self.States['BCAP_eligible'][-1]
        CRP_eligible = self.Attributes['patch_CRP_eligible']
        slopes = self.Attributes['patch_slope']
        areas = self.Attributes['patch_areas']
        discount_rate = self.Attributes['discount_factor']
        risk_aver = self.Attributes['risk_factor']
        loss_aver = self.Attributes['loss_factor']
        soil_loss = self.Attributes['soil_loss']
        peer_ec = self.States['peer_ec'][-1]
        if peer_ec == 1:
            environ_factor = config.enhanced_neighbor_impact[0]
        elif peer_ec == 2:
            environ_factor = config.enhanced_neighbor_impact[1]
        else:
            environ_factor = config.enhanced_neighbor_impact[2]

        prices = np.repeat(price,6) # use the contract price

        Utility = np.zeros(N)

        for idx in range(N):
            policy_eligibilities = [BCAP_eligible[ID_contract[idx]], TMDL_eligible[ID_contract[idx]],
                                    CRP_eligible[ID_contract[idx]],environ_factor] # since this is only for the contract estimation, soil_loss is not taken into account
            Utility[idx] = self.cal_utility_perennial(land_uses[idx], ID_contract[idx], land_uses_b[ID_contract[idx]], slopes[ID_contract[idx]],
                                                      areas[ID_contract[idx]], prices, Peren_ages[ID_contract[idx]],policy_eligibilities,soil_loss, # since this is only for the contract estimation, soil_loss is not taken into account
                                                    discount_rate, risk_aver, loss_aver,year)

        return Utility.sum()

    def cal_peren_break_even_price(self,is_contract,opt_land_uses,contract_max_price,year):
        # function to calculate the break even price of selected land patches

        N_contract = (opt_land_uses[is_contract==1]>4).sum() # only account for the land patches that new installation will be implemented
        ID_contract = np.argwhere(opt_land_uses[is_contract==1]>4)
        opt_land_use = opt_land_uses[ID_contract]
        opt_land_use = opt_land_use.astype(int)
        prices_current = copy.deepcopy(self.States['price_prior'][-1]) # the current market prices of all crops
        land_uses_b = self.States['land_use'][-1]  # the land use in previous year; at this time the land use has not been updated
        climates = self.States['climate_forecasts'][-1]  # farmer's forecast of climate
        Peren_ages = self.States['peren_age'][-1]  # the age of perennial grass in the previous year
        TMDL_eligible = self.States['TMDL_eligible'][-1]  # 1 for eligible, 0 for not
        BCAP_eligible = self.States['BCAP_eligible'][-1]
        CRP_eligible = self.Attributes['patch_CRP_eligible']

        U_max_crop = np.zeros(N_contract)

        for idx in range(N_contract):
            policy_eligibilities = [BCAP_eligible[ID_contract[idx]], TMDL_eligible[ID_contract[idx]], CRP_eligible[ID_contract[idx]],0] # since this is only for the contract estimation, soil_loss is not taken into account
            best_crop, U_max_crop[idx] = self.cal_max_potential_U_crop(idx, land_uses_b[ID_contract[idx]], climates, prices_current, Peren_ages[ID_contract[idx]], policy_eligibilities,year)

        U_crop_max = U_max_crop.sum()

        func = lambda price: (self.cal_utility_perennial_series(ID_contract,opt_land_use,price,year) - U_crop_max)**2

        break_even_price = optimize.least_squares(func,contract_max_price/2,bounds=(0,contract_max_price))['x'][0]

        return break_even_price

    def cal_min_contract_price(self,contract_feed_type,contract_max_price,market_feedstock_price,year):
        # function to calculate the minimum contract price accepted by farmer

        patch_IDs = self.Attributes['patch_ID']
        N_patch = patch_IDs.size

        land_uses_b = self.States['land_use'][-1]  # the land use in previous year; at this time the land use has not been updated
        prices = copy.deepcopy(self.States['price_prior'][-1]) # farmer's forecast of crop prices
        climates = self.States['climate_forecasts'][-1] # farmer's forecast of climate
        Peren_ages = self.States['peren_age'][-1] # the age of perennial grass in the previous year
        slopes = self.Attributes['patch_slope']
        areas = self.Attributes['patch_areas']

        is_contract, opt_land_use = self.identify_contract_patches(contract_feed_type,contract_max_price,year)
        is_all_continue_current_grass = (opt_land_use[is_contract==1]<5).prod() # if all the new contract will be made with lands already growing perennial grass

        if is_all_continue_current_grass == 1: # if all the new contract will be made with lands already growing perennial grass
            min_contract_price = market_feedstock_price # use the market price
        else:
            min_contract_price = self.cal_peren_break_even_price(is_contract,opt_land_use,contract_max_price,year) # if there will be new installation, calculate the breakeven price

        min_contract_price = max(min_contract_price,market_feedstock_price) # the minimum price acceptable for farmer is the maximum among the breakeven and market price

        # return the minimum acceptable price, the land patches accepting contract, and the optimal crop grown on those contracted land
        return min_contract_price, is_contract, opt_land_use

    def land_use_decision_econo(self,is_contract,year):
        # function to identify the land use based on pure economic benefits
        # the output opt_land_use will not be used directly as the final land use decision, if the farmer is no growing energy crop,
        #   and he/she is not planning to grow energy crop based on economic reason, he will decide if he is going to grow
        #   energy crop in one of his land patch based on environmental reason
        # the output U_peren_minus_crop stores the information to identify the land that is most likely to grow perennial grass

        patch_IDs = copy.deepcopy(self.Attributes['patch_ID'])
        N_patch = copy.deepcopy(patch_IDs.size)

        prices = copy.deepcopy(self.States['price_prior'][-1])  # the predicted market prices of all crops
        land_uses_b = copy.deepcopy(self.States['land_use'][-1])  # the land use in previous year; at this time the land use has not been updated
        climates = copy.deepcopy(self.States['climate_forecasts'][-1])  # farmer's forecast of climate
        Peren_ages = copy.deepcopy(self.States['peren_age'][-1])  # the age of perennial grass in the previous year
        TMDL_eligible = copy.deepcopy(self.States['TMDL_eligible'][-1])  # 1 for eligible, 0 for not
        BCAP_eligible = copy.deepcopy(self.States['BCAP_eligible'][-1])
        CRP_eligible = copy.deepcopy(self.Attributes['patch_CRP_eligible'])
        risk_aver = copy.deepcopy(self.Attributes['risk_factor'])
        loss_aver = self.Attributes['loss_factor']
        peer_ec = self.States['peer_ec'][-1]
        if peer_ec == 1:
            environ_factor = config.enhanced_neighbor_impact[0]
        elif peer_ec == 2:
            environ_factor = config.enhanced_neighbor_impact[1]
        else:
            environ_factor = config.enhanced_neighbor_impact[2]

        contract_feed_type = -1 # assuming no contract
        contract_max_price = 0
        peren_code = [3, 4, 5, 6, 30, 40]

        opt_land_use = np.zeros(N_patch)
        best_crop = np.zeros(N_patch)
        U_max_crop = np.zeros(N_patch)
        best_peren = np.zeros(N_patch)
        U_max_peren = np.zeros(N_patch)

        # first calculate the utility of each land patch
        for idx in range(N_patch):
            if (Peren_ages[idx] >= GMP.cal_contract_length(land_uses_b[idx])): # !!!! if it is not already growing perennial grass
                policy_eligibilities = [BCAP_eligible[idx],TMDL_eligible[idx],CRP_eligible[idx],environ_factor] # since this is only for economic reasoning, soil_loss is not taken into account
                best_crop[idx], U_max_crop[idx] = self.cal_max_potential_U_crop(idx, land_uses_b[idx], climates, prices, Peren_ages[idx], policy_eligibilities,year)
                best_peren[idx], U_max_peren[idx] = self.cal_max_potential_U_peren(idx, land_uses_b[idx], climates, prices, Peren_ages[idx], policy_eligibilities,
                                      contract_feed_type, contract_max_price,year)
                U_fallow = (0+policy_eligibilities[1]*gov.TMDL_subsidy)**risk_aver
                if policy_eligibilities[2]>0:
                    if (0 + gov.CRP_subsidy + policy_eligibilities[1]*gov.TMDL_subsidy -
                        config.land_rent* self.Attributes['patch_CRP_eligible'][idx] * config.marginal_land_rent_adj) > 0:
                        U_CRP = (0 + gov.CRP_subsidy + policy_eligibilities[1]*gov.TMDL_subsidy -
                                 config.land_rent * self.Attributes['patch_CRP_eligible'][idx] * config.marginal_land_rent_adj) ** risk_aver
                    else:
                        U_CRP = -loss_aver*(-(0 + gov.CRP_subsidy + policy_eligibilities[1] * gov.TMDL_subsidy -
                                              config.land_rent * self.Attributes['patch_CRP_eligible'][idx] * config.marginal_land_rent_adj)) ** risk_aver
                else:
                    U_CRP = -10**6

                if self.Attributes['type'] in [1,2]: # Type I and Type II farmers allow adoption of perennial crop if it is profitable, the other two Types do not
                    best_ID = np.argmax([U_max_crop[idx],U_max_peren[idx]/10.,U_fallow,U_CRP])
                    if best_ID == 0:
                        opt_land_use[idx] = best_crop[idx]
                    elif best_ID == 1:
                        opt_land_use[idx] = peren_code[int(best_peren[idx])]
                    elif best_ID == 2:
                        opt_land_use[idx] = 7
                    elif best_ID == 3:
                        opt_land_use[idx] = 8
                else:
                    best_ID = np.argmax([U_max_crop[idx], U_fallow, U_CRP])
                    if best_ID == 0:
                        opt_land_use[idx] = best_crop[idx]
                    elif best_ID == 1:
                        opt_land_use[idx] = 7
                    elif best_ID == 2:
                        opt_land_use[idx] = 8

            else:
                opt_land_use[idx] = land_uses_b[idx]
        # U_peren_minus_crop = U_max_peren - U_max_crop # this variable is used to identify the land that is most likely to grow perennial grass
        # return opt_land_use, U_peren_minus_crop
        return opt_land_use

    def land_use_decision_bn(self,is_contract,community_list,year):
        # function for farmers to make decision using BN
        opt_land_use = self.land_use_decision_econo(is_contract,year)
        already_negeotiated = self.Temp['already_negeotiated']
        if already_negeotiated == 0: # if the farmer has never been asked to consider a contract
            AT_commu = community_list[self.Attributes['community_ID']].States['attitude'][-1]
            if community_list[self.Attributes['community_ID']].States['attitude'].__len__() == 1:
                sign_AT = 1
            elif community_list[self.Attributes['community_ID']].States['attitude'][-1] >= \
                    community_list[self.Attributes['community_ID']].States['attitude'][-2]:
                sign_AT = 1
            else:
                sign_AT = 0
            prices = copy.deepcopy(self.States['price_prior'][-1])  # farmer's forecast of crop prices

            SC_Rev_mean_num, opt_peren = self.cal_max_potential_profit_peren(prices,0,0,year)
            opt_peren = opt_peren * 10
            if SC_Rev_mean_num < 150 * 2.47:
                SC_Rev_mean = 1
            elif SC_Rev_mean_num < 250 * 2.47:
                SC_Rev_mean = 2
            else:
                SC_Rev_mean = 3

            self.bn_inference(AT_commu, sign_AT, 1, SC_Rev_mean, year) # do inferencing based on BN

            land_use_exist = self.States['land_use'][-1]  # the existing land use
            is_peren_exist = ((land_use_exist > 2) & (land_use_exist < 7)) | (land_use_exist > 10)
            area_peren_exist = self.Attributes['patch_areas'][is_peren_exist].sum() / self.Attributes[
                'patch_areas'].sum()

            # check willingness to adopt perennial grass based on BN
            is_adopt = self.States['is_adopt'][-1]
            if SC_Rev_mean_num < 0:
                is_adopt = 0
            SC_ratio_discrete = self.States['SC_Ratio'][-1]
            if (SC_ratio_discrete == 1) | (is_adopt == 0):
                SC_ratio = 0
            elif SC_ratio_discrete == 2:
                SC_ratio = 0.05
            elif SC_ratio_discrete == 3:
                SC_ratio = 0.1
            elif SC_ratio_discrete == 4:
                SC_ratio = 0.15
            else:
                SC_ratio = 0.3

            if is_adopt == 1:
                if SC_ratio > area_peren_exist:
                    adopt_patch_id = GMP.identify_adopt_id(SC_ratio - area_peren_exist,
                                                           self.Attributes['adopt_priority'],
                                                           self.Attributes['patch_areas'])
                    opt_land_use[adopt_patch_id] = opt_peren
        return opt_land_use

    def update_attitude_self(self,opt_land_use,U_peren_minus_crop,AT_commu, sign_AT,year):
        # function to update the farmer's attitude toward perennial grass
        # in the function, farmer's sensitivity to environment is also updated
        # also, the perennial grass that is most likely to be grown is provided as an second output

        prices = copy.deepcopy(self.States['price_prior'][-1])  # the predicted market prices of all crops
        land_uses_b = self.States['land_use'][-1]  # the land use in previous year; at this time the land use has not been updated
        climates = self.States['climate_forecasts'][-1]  # farmer's forecast of climate
        Peren_ages = self.States['peren_age'][-1]  # the age of perennial grass in the previous year
        TMDL_eligible = self.States['TMDL_eligible'][-1]  # 1 for eligible, 0 for not
        BCAP_eligible = self.States['BCAP_eligible'][-1]
        CRP_eligible = self.Attributes['patch_CRP_eligible']

        min_sen = self.Attributes['min_sensi_environ']
        max_sen = self.Attributes['max_sensi_environ']
        ini_sen = self.States['environ_sen'][-1]
        theta = self.Attributes['sensi_community']

        environ_sen = self.update_environ_sen(ini_sen, min_sen, max_sen, theta, AT_commu, sign_AT)
        self.States['environ_sen'].append(environ_sen) # update farmer's sensitivity to environment

        tao = self.Attributes['learning_rate']
        ini_AT = self.States['attitude'][-1] # initial attitude to perennial grass

        if ((opt_land_use <7) & (opt_land_use > 8)).sum() == 0:  # is all land are fallow or CRP
            most_likely_peren_ID = np.argmax(U_peren_minus_crop)
        else:
            U_peren_minus_crop[opt_land_use == 7] = -10 ** 10  # mask out fallow and CRP land
            U_peren_minus_crop[opt_land_use == 8] = -10 ** 10
            most_likely_peren_ID = np.argmax(U_peren_minus_crop)

        policy_eligibilities = [BCAP_eligible[most_likely_peren_ID],TMDL_eligible[most_likely_peren_ID],
                                CRP_eligible[most_likely_peren_ID],environ_sen]  # soil_loss is considered for updating attitude
        argmax_PG, U_PG = self.cal_max_potential_U_peren(most_likely_peren_ID, land_uses_b[most_likely_peren_ID], climates, prices,
                                              Peren_ages[most_likely_peren_ID], policy_eligibilities, -1, 0,year)
        argmax_C, U_C = self.cal_max_potential_U_crop(most_likely_peren_ID,land_uses_b[most_likely_peren_ID],climates,prices,
                                            Peren_ages[most_likely_peren_ID],policy_eligibilities,year)
        U_A = U_PG - U_C  # U_A = U_PG - U_C
        opt_peren = [30,40,5,6,30,40][int(argmax_PG)]  # the perennial grass with the maximum U_A

        if U_A >= 0: # function to update the ante attitude toward perennial grass
            AT = ini_AT + U_A * tao * (1 - ini_AT)
        else:
            AT = ini_AT + U_A * tao * ini_AT

        AT = min(max(AT,0),1)

        return AT, opt_peren, most_likely_peren_ID

    def update_AT_neighbor(self,AT_neighbors):

        # ID = self.ID
        neigh_weight = self.Attributes['neigh_weight']
        neighbors = self.Attributes['neighbors']
        AT = np.dot(neigh_weight,AT_neighbors[neighbors]) # final attitude is the weighted sum of all neighbor attitudes (including own attitude)

        self.States['attitude'].append(AT)  # update farmer's attitude
        return AT

    def bn_inference(self,AT_commu,sign_AT,SC_Contract,SC_Rev_mean,year):
        # farmer's land use decision using bayesian network
        ini_sen = copy.deepcopy(self.States['environ_sen'][-1])
        theta = self.Attributes['sensi_community']
        environ_sen = self.update_environ_sen(ini_sen, 0, 1, theta, AT_commu, sign_AT)
        if self.States['environ_sen'].__len__() < year+2:
            self.States['environ_sen'].append(environ_sen)  # update farmer's sensitivity to environment
        else:
            self.States['environ_sen'][-1] = environ_sen

        if self.States['imp_env'].__len__() < year+2:
            if environ_sen<0.33:
                self.States['imp_env'].append(3)
            elif environ_sen<0.67:
                self.States['imp_env'].append(2)
            else:
                self.States['imp_env'].append(1)
        else:
            if environ_sen<0.33:
                self.States['imp_env'][-1] = 3
            elif environ_sen<0.67:
                self.States['imp_env'][-1] = 2
            else:
                self.States['imp_env'][-1] = 1

        # inference the probability of adoption
        df_temp = bn_df[(bn_df['farm_size']==self.Attributes['farm_size'])&
                                                  (bn_df['info_use']==self.Attributes['info_use']) &
                                                  (bn_df['peer_ec']==self.States['peer_ec'][-1])&
                                                  (bn_df['benefit'] == self.Attributes['benefit']) &
                                                  (bn_df['concern'] == self.Attributes['concern']) &
                                                  (bn_df['imp_env'] == self.States['imp_env'][-1]) &
                                                  (bn_df['lql'] == self.Attributes['lql']) &
                                                  (bn_df['SC_Contract'] == SC_Contract) &
                                                  (bn_df['SC_Rev_mean'] == SC_Rev_mean)]

        if df_temp.shape[0]<10: # if the conditional sample size is too small, then use the unconditional adoption probability
            p_adopt = bn_df['SC_Will'].mean()-1
            if p_adopt>0.5:
                if self.States['is_adopt'].__len__() < year+2:
                    self.States['is_adopt'].append(1)
                else:
                    self.States['is_adopt'][-1] = 1
            else:
                if self.States['is_adopt'].__len__() < year + 2:
                    self.States['is_adopt'].append(0)
                else:
                    self.States['is_adopt'][-1] = 0
            if self.States['SC_Will'].__len__() < year + 2:
                self.States['SC_Will'].append(p_adopt)
                self.States['SC_Ratio'].append(bn_df.sample()['SC_Ratio'].max())
                self.States['max_fam'].append(bn_df['max_fam'].mean())
            else:
                self.States['SC_Will'][-1]=p_adopt
                self.States['SC_Ratio'][-1]=bn_df.sample()['SC_Ratio'].max()
                self.States['max_fam'][-1]=bn_df['max_fam'].mean()
        else: # otherwise use the conditional adoption probability
            p_adopt = df_temp['SC_Will'].mean() - 1
            if p_adopt > 0.5:
                if self.States['is_adopt'].__len__() < year + 2:
                    self.States['is_adopt'].append(1)
                else:
                    self.States['is_adopt'][-1]=1
            else:
                if self.States['is_adopt'].__len__() < year + 2:
                    self.States['is_adopt'].append(0)
                else:
                    self.States['is_adopt'][-1] = 0
            if self.States['SC_Will'].__len__() < year + 2:
                if np.isin(self.States['land_use'][-1], [3, 4]).sum() > 0 + 0: # if already growing perennial grass, the willingness will be last least the same as previous year
                    self.States['SC_Will'].append(max(p_adopt,self.States['SC_Will'][-1]))
                else:
                    self.States['SC_Will'].append(p_adopt)
                self.States['SC_Ratio'].append(df_temp.sample()['SC_Ratio'].max())
                self.States['max_fam'].append(df_temp['max_fam'].mean())
            else:
                if np.isin(self.States['land_use'][-1], [3, 4]).sum() > 0 + 0: # if already growing perennial grass, the willingness will be last least the same as previous year
                    self.States['SC_Will'][-1] = max(p_adopt,self.States['SC_Will'][-1])
                else:
                    self.States['SC_Will'][-1]=p_adopt
                self.States['SC_Ratio'][-1]=df_temp.sample()['SC_Ratio'].max()
                self.States['max_fam'][-1]=df_temp['max_fam'].mean()
        self.Temp['already_negeotiated'] = 1

    # def land_use_decision(self):
    #
    #     opt_land_use = self.Temp['opt_land_use']
    #     opt_peren = self.Temp['opt_peren']
    #
    #     land_use_exist = self.States['land_use'][-1] # the existing land use
    #     is_peren_exist = ((land_use_exist > 2) & (land_use_exist < 7)) | (land_use_exist > 10)
    #     area_peren_exist = self.Attributes['patch_areas'][is_peren_exist].sum()/self.Attributes['patch_areas'].sum()
    #
    #     # check willingness to adopt perennial grass based on BN
    #     is_adopt = self.States['is_adopt'][-1]
    #     SC_ratio_discrete = self.States['SC_Ratio'][-1]
    #     if SC_ratio_discrete==1:
    #         SC_ratio = 0
    #     elif SC_ratio_discrete==2:
    #         SC_ratio = 0.05
    #     elif SC_ratio_discrete==3:
    #         SC_ratio = 0.1
    #     elif SC_ratio_discrete==4:
    #         SC_ratio = 0.15
    #     else:
    #         SC_ratio=0.3
    #
    #     if is_adopt == 1:
    #         if SC_ratio > is_adopt:
    #             adopt_patch_id = GMP.identify_adopt_id(SC_ratio-area_peren_exist,
    #                                                    self.Attributes['adopt_priority'],self.Attributes['patch_areas'])
    #             opt_land_use[adopt_patch_id] = opt_peren
    #
    #     return opt_land_use

    def compile_farmer_contract(self,opt_land_use):
        # function to compile all contract related information for a farmer, and return the land patches that are under a contract during the year
        contract_land_use = self.Temp['contract_land_use']
        previous_contract = self.States['contract'][-1]
        peren_age = self.States['peren_age'][-1]
        previous_land_use = self.States['land_use'][-1]
        contract_previous = (previous_contract > 0) * (peren_age < GMP.cal_contract_length(previous_land_use))
        contract_new = contract_land_use > 2
        N_patch = previous_land_use.__len__()
        # contract = 2 * (contract_previous + contract_new) - 1
        contract = np.zeros(contract_previous.shape[0],int)
        for i in range(N_patch):
            if (contract_previous[i] ==1) | (contract_new[i] == 1):
                contract[i] = 1
            else:
                contract[i] = -1

        for i in range(N_patch):
            if contract_new[i] == 1:
                if contract_land_use[i]>10:
                    opt_land_use[i] = int(contract_land_use[i]/10)
                    self.Temp['peren_age_refresh'][i] = int(1)
                else:
                    opt_land_use[i] = int(contract_land_use[i])
            elif opt_land_use[i]>10:
                opt_land_use[i] = int(opt_land_use[i]/10)
                self.Temp['peren_age_refresh'][i] = int(1)

        self.States['contract'].append(contract)
        self.States['land_use'].append(opt_land_use.astype(int))


class Can_ref:

    def __init__(self, ID, Attributes, States):

        self.ID = ID
        self.Attributes = Attributes
        self.States = States

    def cal_supply_curve(self,feed_type,patch_BEM,patch_PM,farmer_list,N_patch):
        # function to calculate the local supply curve around the refinery
        # feed_type: the feedstock type, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #            5 for sorghum, 6 for cane
        # patch_BEM: patch specific break even matrix, a matrix showing the minimum acceptable price for farmer to grow certain crop for each land patch
        # patch_PM: a matrix showing the expected production (in MT) of each crop (column) for each land patch (row)
        # farmer_list: list of all farmer agents
        # N_patch: the total number of land patches considered in this model
        dist_farmer = self.Attributes['dist_farmer']
        patch_dist = GMP.farm_dist_to_patch_dist(dist_farmer,farmer_list,N_patch)
        patch_within_range = patch_dist <= config.patch_influence_range  # all patches within the range of influence will be considered as potential feedstock sources
        BE_list = patch_BEM[patch_within_range,feed_type] # a list of break even prices
        PM_list = patch_PM[patch_within_range,feed_type] # a list of production
        supply_list = np.asarray([BE_list,PM_list]).T
        supply_list = supply_list[np.argsort(supply_list[:,0])] # sort the supply list based on the price
        supply_list[:,1] = np.cumsum(supply_list[:,1]) # calculate the cumulative sum for the production, i.e., the total supply under specific price
        return supply_list  # supply_list is the supply curve

    def check_supply_curve(self,supply_curve,demand):
        # function to calculate the appropriate feedstock price based on the supply curve and demand
        # supply_curve: 2 column array showing the relationship between price (left) and supply (right)
        # demand: the demand of specific feedstock
        ID = np.where(supply_curve[:,1]>demand) # check the ID where supply is higher than demand
        if ID[0].__len__() == 0: # if the maximum possible supply is lower than demand
            price = supply_curve[-1,0] # price will be maximum price in supply curve
            supply = supply_curve[-1,1] # supply will be the maximum supply in supply curve
        else:
            price = supply_curve[ID,0].min() # otherwise, price will be the minimum price to meet the demand
            supply = copy.deepcopy(demand)

        if demand==0:
            price=0

        return price, supply

    def cal_NPV(self, feed_prices, product_prices, price_adj, trans_costs, storage_costs, subsidys,taxs,ref_inv_cost_adj_his,ref_pro_cost_adj_his,year):
        # function to calculate the net present value of a refinery investment
        # feed_prices: prices of feedstocks, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # product_prices: prices of refinery price, 0 for ethanol, 1 for biodisel, 2 for bagasse, 3 for DDGS, 4 for glycerol
        # price_adj: the adjustment of price for cellulosic biofuels
        # invest_costs: costs for investment, 0 for viable investment cost, 1 for base fixed investment cost, 2 for base capacity
        # production_costs: costs for biofuel production, 0 for fixed production cost, 1 for viable production cost
        # trans_costs: transportation costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # storage_costs: storage costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # subsidys: subsidies for refinery, 0 for cost share (in $), 1 for cost share (in %), 2 for production subsidy (in $/L ethanol equivilent)
        # taxs: taxes for refinery, 0 for tax deduction, 1 for tax rate
        # ref_inv_cost_adj_his: the investment cost adjustment of learning by doing
        # ref_pro_cost_adj_his: the production cost adjustment of learning by doing
        K = self.Attributes['capacity']
        feedstock_amount = self.States['feedstock_amount'][-1] # feedstock_amount: amount of biomass used (in MT),
        feedstock_amount = feedstock_amount * (feedstock_amount>0) * (1 - config.mis_trans_loss - config.mis_storage_loss)
        tech_type = self.Attributes['refinery_type']
        aver_dist = self.States['aver_dist'][-1]
        # invest_costs = invest_costs[:,tech_type-1]
        # production_costs = production_costs[:,tech_type-1]
        # 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        # 5 for bagasse, 6 for sorghum_oil, 7 for lipidcane_oil, 8 for sorghum_joint, 9 for lipidcane_joint

        if tech_type == 2:
            product_prices = product_prices + price_adj
            cornstr_ratio = feedstock_amount[2]/(feedstock_amount[2]+feedstock_amount[3])
            TEA_result = GMP.TEA_model_2(cornstover_fraction=cornstr_ratio,plant_capacity=K,
                                         price_cornstover=feed_prices[2],price_miscanthus=feed_prices[3],price_ethanol=product_prices[0])
            aver_dist_final = (aver_dist[2] * feedstock_amount[2] + aver_dist[3] * feedstock_amount[3])/(feedstock_amount[2]+feedstock_amount[3])
        elif tech_type == 1:
            aver_dist_final = aver_dist[0]
            TEA_result = GMP.TEA_model_1(plant_capacity=K,price_corn=feed_prices[0], price_DDGS=product_prices[3],
                                         price_ethanol=product_prices[0])
        elif tech_type == 5: # cofiring with <5% biomass
            TEA_result = {}
            TEA_result['TCI'] = 0
            TEA_result['VOC'] = config.ref_cost_table[4,5]
            TEA_result['FOC'] = config.ref_cost_table[4,4]
            aver_dist_final = (aver_dist[2] * feedstock_amount[2] + aver_dist[3] * feedstock_amount[3]) / (
                        feedstock_amount[2] + feedstock_amount[3])
        elif tech_type == 6: # cofiring with <15% biomass
            TEA_result = {}
            TEA_result['TCI'] = (config.ref_cost_table[5, 1] + config.ref_cost_table[5, 2] * K) * 10**6
            TEA_result['VOC'] = config.ref_cost_table[5, 5]
            TEA_result['FOC'] = config.ref_cost_table[5, 4]
            aver_dist_final = (aver_dist[2] * feedstock_amount[2] + aver_dist[3] * feedstock_amount[3]) / (
                        feedstock_amount[2] + feedstock_amount[3])
        elif tech_type == 7: #BCHP
            TEA_result = {}
            TEA_result['TCI'] = (config.ref_cost_table[6, 1] + config.ref_cost_table[6, 2] * K) * 10**6
            TEA_result['VOC'] = config.ref_cost_table[6, 5]
            TEA_result['FOC'] = config.ref_cost_table[6, 4]
            aver_dist_final = (aver_dist[2] * feedstock_amount[2] + aver_dist[3] * feedstock_amount[3]) / (
                        feedstock_amount[2] + feedstock_amount[3])

        if tech_type < 5:
            productions = np.dot(feedstock_amount, config.refinery_product_yield_table)
            invest_cost = TEA_result['TCI'] * ref_inv_cost_adj_his[year, tech_type - 1]
            invest_cost_adj = invest_cost - subsidys[0] - subsidys[1] * invest_cost
            production_cost = (TEA_result['VOC'] * K + TEA_result['FOC'] * K) * ref_pro_cost_adj_his[year, tech_type - 1]
        else:
            productions = np.dot(feedstock_amount[2:6].sum(), config.biofacility_product_yield_table[tech_type-5,:])
            invest_cost = TEA_result['TCI']
            invest_cost_adj = invest_cost - subsidys[0] - subsidys[1] * invest_cost
            production_cost = (TEA_result['VOC'] * K + TEA_result['FOC'] * K)

        trans_amount = feedstock_amount[0:8]
        trans_amount[6] = trans_amount[6] + feedstock_amount[8] # add up the amounts of sorghum oil and sorghum joint
        trans_amount[7] = trans_amount[7] + feedstock_amount[9] # add up the amounts of cane oil and cane joint
        # trans_cost = np.dot(trans_amount,trans_costs*config.system_boundary_radius/2)
        trans_cost = np.dot(trans_amount, trans_costs * aver_dist_final)
        storage_cost = np.dot(trans_amount,storage_costs)
        feed_cost = np.dot(trans_amount,feed_prices)

        if tech_type < 5:
            product_amount = np.dot(feedstock_amount,config.refinery_product_yield_table)
        else:
            product_amount = np.dot(feedstock_amount[2:6].sum(), config.biofacility_product_yield_table[tech_type-5,:])
        sale_amount = np.dot(product_amount,product_prices)
        product_subsidy = np.dot(productions[0:2],config.ethanol_equivilents) * subsidys[2]
        if tech_type >= 5: # subsidy for bio-electricity
            product_subsidy += productions[5] * subsidys[2]

        loan = Loan(principal=invest_cost_adj, interest=config.ref_interest_rate[self.Attributes['co-op']], term=config.refinery_life_span)
        interest = float(12 * loan._monthly_payment) - invest_cost_adj/config.refinery_life_span
        tax = (sale_amount + product_subsidy - production_cost - trans_cost - storage_cost - feed_cost - interest - invest_cost_adj/config.refinery_life_span - taxs[0]) * taxs[1]

        annual_cash_flow = sale_amount + product_subsidy - production_cost - trans_cost - storage_cost - feed_cost - tax - interest
        cash_flow = np.ones(config.refinery_life_span) * annual_cash_flow
        cash_flow = np.concatenate((np.asarray([-invest_cost_adj * 0.5]),cash_flow))
        cash_flow = np.concatenate((np.asarray([-invest_cost_adj * 0.5]), cash_flow))

        NPV = np.npv(config.inflation_rate,cash_flow)
        IRR = np.irr(cash_flow)

        self.States['NPV'].append(NPV)
        self.States['IRR'].append(IRR)
        self.States['invest_cost'].append(invest_cost)
        self.States['invest_cost_adj'].append(invest_cost_adj)
        self.States['interest_payment'].append(interest)
        if tech_type < 5:
            self.States['fix_cost'].append(TEA_result['FOC'] * ref_pro_cost_adj_his[year,tech_type-1])
            self.States['via_cost'].append(TEA_result['VOC'] * ref_pro_cost_adj_his[year, tech_type - 1])
        else:
            self.States['fix_cost'].append(TEA_result['FOC'])
            self.States['via_cost'].append(TEA_result['VOC'])
        return NPV, IRR, invest_cost

    def cal_water_use(self, feedstock_amount):
        # function to calculate the water use for biofuel production
        # feedstock_amount: amount of biomass used (in MT), 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum_oil, 7 for lipidcane_oil, 8 for sorghum_joint, 9 for lipidcane_joint
        K = self.Attributes['capacity']
        WU = np.dot(feedstock_amount,config.refinery_water_use_table)/10**6
        # self.States['WU'].append(WU)
        return WU

class Community:
    def __init__(self, ID, Attributes, States, Temp):

        self.ID = ID
        self.Attributes = Attributes
        self.States = States
        self.Temp = Temp

    def ini_community(self,farmer_list):
        # function to initiate the community agent
        ID = self.ID
        N_loads_sum = np.zeros(8)
        areas_sum = 0
        for farm_agent in farmer_list:
            commu_ID = farm_agent.Attributes['community_ID']
            if commu_ID == ID:
                patch_ID = farm_agent.Attributes['patch_ID']
                for i in range(patch_ID.size):
                    N_loads_sum += config.patch_N_loads_mean[patch_ID[i],:] * farm_agent.Attributes['patch_areas'][i]
                    areas_sum += farm_agent.Attributes['patch_areas'][i]

        average_N_loads = N_loads_sum/areas_sum
        self.Attributes['average_N_loads'] = average_N_loads

    def cal_N_load_in_community(self,farmer_list):
        # function to calculate the total N release in the community
        # farmer_list: a list of farmer agents
        N_farmer = farmer_list.__len__()
        N_release = 0
        for i in range(N_farmer):
            farmer_agent = farmer_list[i]
            if farmer_agent.Attributes['community_ID'] == self.ID:
                N_release += (farmer_agent.States['N_release'][-1] * farmer_agent.Attributes['patch_areas']).sum()/1000
        return N_release

    def cal_revenue_in_community(self,farmer_list):
        # function to calculate the total actual revenue received by the farmers in the community
        N_farmer = farmer_list.__len__()
        revenue = 0
        for i in range(N_farmer):
            farmer_agent = farmer_list[i]
            if farmer_agent.Attributes['community_ID'] == self.ID:
                revenue += farmer_agent.States['revenue'][-1].sum()
        return revenue

    def cal_attitude(self,farmer_list):
        # function to calculate the environmental attitude of community
        # N_load: the current nitrogen release of the community
        N_load = copy.deepcopy(self.cal_N_load_in_community(farmer_list))
        attitude = copy.deepcopy(self.States['attitude'][-1])
        base_rate = copy.deepcopy(self.Attributes['base_increase_environ'])
        TMDL = copy.deepcopy(self.Attributes['N_limit'])
        att_max = copy.deepcopy(self.Attributes['max_attitude'])
        l = copy.deepcopy(self.Attributes['sensi_N'])
        if N_load > TMDL:
            attitude = attitude + attitude * l * (1-attitude/att_max) * (N_load/TMDL) + base_rate
        else:
            attitude = attitude + base_rate

        attitude = min(attitude,att_max)
        self.States['attitude'].append(attitude)
        return attitude, N_load

    def cal_willingess(self,revenue,N_job,WU,delta_LU,capacity):
        # function to calculate the willingness of community to accept proposed refinery
        # revenue: the additional revenue brought by refinery
        # N_job: the number of job offered by refinery
        # WU: water use of refinery
        # delta_LU: change of land use
        # capacity: refinery capacity
        w_r = self.Attributes['ratio_farmer'] * (1/(1 + np.exp(-revenue / self.States['revenue'][-1])) - 0.5)
        w_j = (1 - self.Attributes['ratio_farmer']) * (1/(1 + np.exp(-N_job / self.Attributes['max_job'])) - 0.5)
        w_cap = -1/(1 + np.exp(-capacity/self.Attributes['max_cap'])) + 0.5
        w_w = -1 / (1 + np.exp(-WU / self.States['water_avail'][-1])) + 0.5
        w_l = -1 / (1 + np.exp(-delta_LU / self.Attributes['max_LU'])) + 0.5

        w = w_r + w_j + w_cap + w_w + w_l
        return w

    def pred_lU_change(self,ref_agent,farmer_list,current_feed_prices,ref_feed_prices,normal_crop_yields):
        # function to calculate the estimated land use change withnin a community
        # ref_agent: the proposed refinery agent
        # farmer_list: the list of farmer agents
        # current_feed_prices: current prices of feedstocks, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # ref_feed_prices: prices of feedstocks if new refinery is built
        # normal_crop_yields: the regular crop yield for each crop, 0 for corn, 1 for soy, 2 for mis, 3 for swtich, 4 for sorghum, 5 for lipidcane
        current_feed_prices_temp = copy.deepcopy(current_feed_prices)
        ref_feed_prices_temp = copy.deepcopy(ref_feed_prices)
        farm_dist = copy.deepcopy(ref_agent.Attributes['dist_farmer'])
        supply_farm_IDs = farm_dist <= config.patch_influence_range
        farm_IDs = np.linspace(0,farm_dist.__len__()-1,farm_dist.__len__())
        farm_IDs = farm_IDs[supply_farm_IDs].astype(int)
        supply_area = 0
        supply_area_in_community = 0
        crop_area_community_current = np.zeros(8)
        crop_area_community_history = np.zeros(8)
        area_weighted_slope = 0
        for idx in range(farm_IDs.__len__()):
            farm_area = farmer_list[idx].Attributes['patch_areas'].sum()
            supply_area = supply_area + farm_area
            if farmer_list[idx].Attributes['community_ID'] == self.ID:
                supply_area_in_community = supply_area_in_community + farm_area
                area_weighted_slope = area_weighted_slope + (farmer_list[idx].Attributes['patch_areas'] * farmer_list[idx].Attributes['patch_slope']).sum()
                for jnd in range(farmer_list[idx].Attributes['patch_ID'].__len__()):
                    temp = farmer_list[idx].States['land_use'][0][jnd] == np.asarray([1,2,3,4,5,6,7,8]) # check the initial land use as the historical land use
                    crop_area_community_history = crop_area_community_history + temp * farmer_list[idx].Attributes['patch_areas'][jnd]
                    temp = farmer_list[idx].States['land_use'][-1][jnd] == np.asarray([1, 2, 3, 4, 5, 6,7,8])  # check the current land use
                    crop_area_community_current = crop_area_community_current + temp * farmer_list[idx].Attributes['patch_areas'][jnd]

        feed_from_community = ref_agent.States['feedstock_amount'][-1] * supply_area_in_community/supply_area # assume feedstocks are provided propotional to land area
        crop_amount = feed_from_community[0:8]
        crop_amount = np.delete(crop_amount,5,0) # delete bagasse as it is not provided from farmer
        crop_amount[5] = crop_amount[5] + feed_from_community[8]  # add up the amounts of sorghum oil and sorghum joint
        crop_amount[6] = crop_amount[6] + feed_from_community[9]  # add up the amounts of cane oil and cane joint
        crop_amount[0] = np.maximum(crop_amount[0],crop_amount[2]/config.stover_harvest_ratio) # the area for corn and corn stover will be the maximum of the two
        crop_amount = np.delete(crop_amount, 2, 0)  # delete corn stover

        feed_area_from_community = crop_amount / normal_crop_yields # the needed area for producing feedstocks
        feed_area_from_community = np.concatenate((feed_area_from_community,[0,0])) # attach the fallow and CRP to feed_area_from_community for ease of calculation
        reallocated_LU = GMP.coarse_reallocate_land_use(crop_area_community_history,feed_area_from_community) # the target land use after refinery is operated
        delta_LU = GMP.cal_delta_LU(crop_area_community_history,reallocated_LU)
        delta_LU = delta_LU/crop_area_community_history.sum()  ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!check if land use change is based on the initial land use or the current land use!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#####

        ref_feed_prices_temp[0] = 0.2 * ref_feed_prices_temp[2] + ref_feed_prices_temp[0]
        current_feed_prices_temp[0] = 0.2 * current_feed_prices_temp[2] + current_feed_prices_temp[0] # lump the corn stover price to corn price
        ref_feed_prices_temp = np.delete(ref_feed_prices_temp,[2,5])
        current_feed_prices_temp = np.delete(current_feed_prices_temp, [2, 5])# revenue for bagasse is not available for farmer
        delta_revenue = reallocated_LU[0:6] * ref_feed_prices_temp - crop_area_community_current[0:6] * current_feed_prices_temp
        delta_revenue = delta_revenue.sum()/10**6

        # average_slope = area_weighted_slope/supply_area_in_community
        # PI_current = GMP.coarse_cal_PI(crop_area_community_current,average_slope) * crop_area_community_current.sum()/1000
        # PI_new = GMP.coarse_cal_PI(reallocated_LU,average_slope) * crop_area_community_current.sum()/1000
        PI_current = np.dot(crop_area_community_current,self.Attributes['average_N_loads'])/1000
        PI_new = np.dot(reallocated_LU,self.Attributes['average_N_loads'])/1000
        delta_PI = PI_new - PI_current

        WU = ref_agent.States['WU'][-1]
        return delta_revenue, delta_LU, delta_PI, WU

class consumer:
    def __init__(self, ID, Attributes, States):

        self.ID = ID
        self.Attributes = Attributes
        self.States = States

    def cal_willingness_to_pay(self,year):
        # function to update the consumer's willingness to pay for bioethanol
        # WP = self.States['willingness_to_pay'][-1]
        # WP = WP + self.Attributes['IRW'] # each time step increase the WP by IRW
        # WP = min(WP,self.Attributes['max_WP']) # WP has a maximum value
        pmax = self.Attributes['max_WP']
        pmin = self.States['willingness_to_pay'][0]
        rou = self.Attributes['IRW']
        WP = pmax*pmin*np.exp(rou*year)/(pmax+pmin*(np.exp(rou*year)-1))
        self.States['willingness_to_pay'].append(WP)
        return WP

    def cal_ethanol_price(self,P_e):
        # function to update the consumer's payed price for bioethanol
        # P_e = self.States['ethanol_price'][-1]
        P_e = P_e + self.States['willingness_to_pay'] [-1]
        self.States['ethanol_price'].append(P_e)
        return P_e

class govern_agent:
    def __init__(self, ID, Attributes, States):

        self.ID = ID
        self.Attributes = Attributes
        self.States = States

    def update_RFS(self, V_ce, V_RFS, V_RFS_new,maintain_RFS):
        # function to adjust the RFS cellulosic mandate
        # if cellulosic biofuel production is 50% lower than mandate, the mandated amount will be
        # adjusted to 1.4 of cellulosic biofuel production
        # V_ce: the volume of cellulosic biofuel production
        # V_RFS: the volume of RFS mandate
        # V_RFS_new: the RFS mandate for the next year
        # maintain_RFS: a bool variable representing the government's determination about maintaining RFS
        if maintain_RFS>0:
            RFS_signal = 1  # a positive policy signal is generated if the RFS mandate is maintained
        elif V_ce < 0.5 * V_RFS:
            V_RFS_new = 1.4 * V_ce
            RFS_signal = -1  # a negative policy signal is generated if the RFS mandate is adjusted
        else:
            RFS_signal = 1  # a positive policy signal is generated if the RFS mandate is maintained
        return V_RFS_new, RFS_signal

    def cal_IRR_adj_factor(self,RFS_signal):
        # function to determine if the minimum IRRs for biorefinery will be adjusted
        # RFS_signal: a list of RFS adjustment history
        if RFS_signal[-1] == -1:
            IRR_adj_factor = 0.02
        elif (RFS_signal[-1] + RFS_signal[-2]) == 2:
            IRR_adj_factor = -0.02
        else:
            IRR_adj_factor = 0
        return IRR_adj_factor

    def cal_CWC_price(self,if_2009,p_gas,V_ce,V_RFS):
        # function to calculate the cellulosic wavier credit price
        # if_2009: the inflation factor wrt 2009
        # p_gas: the price of wholesale gasoline
        # V_ce: the volume of cellulosic biofuel production
        # V_RFS: the volume of RFS mandate
        # telta = self.Attributes['scaling_factor'] # the maximum range of government behavior factor
        # p_CWC = max(0.25*if_2009,3*if_2009-p_gas) * telta * (1+np.exp(-V_ce/V_RFS))
        p_CWC = max(0.25 * if_2009, 3 * if_2009 - p_gas)
        self.States['CWC_price'].append(p_CWC)
        return p_CWC

    def cal_cell_ethanol_price(self,p_ethanol,V_ce):
        # function to calculate the price of cellulosic ethanol as a result of RFS enforcement
        # p_gas: the price of wholesale gasoline
        # p_ethanol: the price of ethanol after BEPAM
        # V_ce: the total volume of cellulosic ethanol produced in the watershed
        CWC_adj_price = self.States['CWC_price'][-1]  # the price of cellulosic ethanol after CWC adjustment
        # CWC_adj_price = CWC_adj_price
        year = len(self.States['CWC_price'])
        if V_ce <= self.States['RFS_volume'][year-1]:
            p_cell_ethanol = max(p_ethanol,CWC_adj_price) # if the RFS mandate is not fulfilled, the cellulosic ethanol price will be the maximum of the CWC adjusted price and the original ethanol price
        else:
            p_cell_ethanol = copy.deepcopy(p_ethanol)

        self.States['RFS_adjusted_cell_ethanol_price'].append(p_cell_ethanol)
        return p_cell_ethanol

    # def update_TMDL(self,N_load):
    #     # function to update the slope limit for defining TMDL critical area for the next year
    #     # N_load: the current N loading
    #     if N_load>self.Attributes['TMDL']:
    #         self.States['slope_limit_ID'].append(min(self.States['slope_limit_ID'][-1]+1,5))
    #     else:
    #         self.States['slope_limit_ID'].append(self.States['slope_limit_ID'][-1])
    #
    #     slope_limit = self.Attributes['slope_limits'][self.States['slope_limit_ID'][-1]]
    #     return slope_limit

    def update_TMDL(self,N_load,year):
        # function to update the N limit for defining TMDL critical area for the next year
        # N_load: the current N loading
        if N_load>self.Attributes['TMDL'][year]:
            self.States['N_limit_ID'].append(min(self.States['N_limit_ID'][-1]+1,11))
        else:
            self.States['N_limit_ID'].append(self.States['N_limit_ID'][-1])

        N_limit = self.Attributes['TMDL_N_limits'][self.States['N_limit_ID'][-1]]
        return N_limit


class Refinery:
    def __init__(self, ID, Attributes, States, Temp):

        self.ID = ID
        self.Attributes = Attributes
        self.States = States
        self.Temp = Temp

    def cal_PBE(self,product_prices,trans_costs,storage_costs):
        # function to calculate the break even price for refinery
        # product_prices: prices of refinery price, 0 for ethanol, 1 for biodisel, 2 for bagasse, 3 for DDGS, 4 for glycerol
        # trans_costs: transportation costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # storage_costs: storage costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        fix_cost = copy.deepcopy(self.Attributes['fix_cost'])
        via_cost = copy.deepcopy(self.Attributes['via_cost'])
        subsidy_rate = copy.deepcopy(self.Attributes['subsidy'])
        tax_rate = copy.deepcopy(self.Attributes['tax_rate'])
        tax_reduction = copy.deepcopy(self.Attributes['tax_reduction'])
        taxs = [tax_reduction,tax_rate]
        K = copy.deepcopy(self.Attributes['capacity'][-1])
        tech_type = self.Attributes['tech_type']
        feedstock_amount = copy.deepcopy(self.Attributes['feedstock_amount'])
        PBE = np.zeros(feedstock_amount.__len__())
        aver_dist = copy.deepcopy(self.Attributes['aver_dist'])

        for i in range(PBE.__len__()):
            if GMP.feed_stock_tech_type_match(i,self.Attributes['tech_type']) == 0:
                PBE[i] = 0
            else:
                crop_ID = i
                if i > 7:
                    crop_ID = i - 2

                if tech_type < 2:
                    feed_weight = K / max(config.refinery_product_yield_table[i, 0:2])
                    product_amount = feed_weight * config.refinery_product_yield_table[i, :]
                elif tech_type < 5:
                    feed_weight = K / max(config.refinery_product_yield_table[i, 0:2])
                    feed_weight /= 1 - config.mis_trans_loss - config.mis_storage_loss
                    product_amount = feed_weight * config.refinery_product_yield_table[i, :] * (1 - config.mis_trans_loss - config.mis_storage_loss)
                elif (i>=2) & (i<=5):
                    feed_weight = K / config.biofacility_product_yield_table[tech_type-5, 5]
                    feed_weight /= 1 - config.mis_trans_loss - config.mis_storage_loss
                    product_amount = feed_weight * config.biofacility_product_yield_table[tech_type-5, :] * (1 - config.mis_trans_loss - config.mis_storage_loss)
                else:
                    product_amount = np.zeros(7)
                revenue = np.dot(product_amount, product_prices)
                trans_cost = feed_weight * trans_costs[crop_ID] * aver_dist
                production_cost = fix_cost * K + via_cost * K
                storage_cost = feed_weight * storage_costs[crop_ID]
                if tech_type < 5:
                    subsidy = np.dot(product_amount[0:2],config.ethanol_equivilents) * subsidy_rate
                else:
                    subsidy = product_amount[5] * subsidy_rate
                tax = (revenue+subsidy-trans_cost-production_cost-storage_cost-taxs[0])*taxs[1]
                PBE[i] = (revenue+subsidy-trans_cost-production_cost-storage_cost-tax)/feed_weight

        PBE[6] = max(PBE[6], PBE[8])
        PBE[7] = max(PBE[7],PBE[9])
        PBE = np.delete(PBE,[8,9]) # 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #             5 for bagasse, 6 for sorghum oil, 7 for lipidcane oil, 8 for sorghum joint, 9 for lipidcane joint
        return PBE

    def produce_biofuel(self,feedstock_amount):
        # function to calculate the biofuel and byproduct production based on the available feedstock
        # feedstock_amount:  amount of biomass used (in MT), 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus,
        # 4 for switchgrass, 5 for bagasse, 6 for sorghum_oil, 7 for lipidcane_oil, 8 for sorghum_joint, 9 for lipidcane_joint
        tech_type = self.Attributes['tech_type']
        if tech_type < 5:
            product_amount = np.dot(feedstock_amount, config.refinery_product_yield_table)
        else:
            product_amount = np.dot(feedstock_amount[2:6].sum(), config.biofacility_product_yield_table[tech_type-5,:])
        return product_amount # 0 fir ethanol, 1 for lipid, 2 for bagasse, 3 for DDGS, 4 for glycerol

    def feed_management_after_contract(self,contracted_feedstock_amount):
        # function to decide the amount of feedstocks to buy or sale based on the contracted feedstock the refinery receive
        # contracted_feedstock_amount: the actual feedstock provided by contracted farmer
        tech_type = self.Attributes['tech_type']
        biofuel_amount = self.produce_biofuel(contracted_feedstock_amount * (1 - config.mis_storage_loss - config.mis_trans_loss))
        if tech_type < 5:
            primary_product_ID = np.argmax(biofuel_amount[0:2] * config.ethanol_equivilents) # among ethaol and biodiesel, the one with larger ethanol equivilent is the primary product for refinery
            product_yield_table = config.refinery_product_yield_table
        else: # if the 'refinery' agent is a biofacility, then the primary product is set as electricity
            primary_product_ID = 5
            product_yield_table = np.zeros(config.refinery_product_yield_table.shape)
            product_yield_table[2:6,:] = config.biofacility_product_yield_table[tech_type-5,:]
        feed_buy = np.zeros(10)
        feed_sell = np.zeros(10)
        if self.States['production_year'][-1]<0:
            pass
        else:
            feedstock_amount_no_bagasse = copy.deepcopy(contracted_feedstock_amount)
            feedstock_amount_no_bagasse[5] = 0
            biofuel_amount_no_bagasse = self.produce_biofuel(feedstock_amount_no_bagasse * (1 - config.mis_storage_loss - config.mis_trans_loss)) # this variable of calculate for future manage of feedstock assuming no change of bagasse
            K = self.Attributes['capacity'][-1]
            if biofuel_amount[primary_product_ID] ==0:
                if self.Attributes['tech_type'] == 1:
                    feed_buy[0] = K / product_yield_table[0,0]
                elif self.Attributes['tech_type'] == 2:
                    feed_buy[3] = K / product_yield_table[3, 0]
                elif self.Attributes['tech_type'] == 3:
                    feed_buy[6] = K / product_yield_table[6, 1]
                elif self.Attributes['tech_type'] == 4:
                    feed_buy[8] = K / product_yield_table[8, 1]
                elif (self.Attributes['tech_type'] == np.asarray([5, 6, 7])).sum() > 0:
                    feed_buy[3] = K / product_yield_table[3, 5]
            elif biofuel_amount[primary_product_ID] < K:
                feed_buy = contracted_feedstock_amount * (K - biofuel_amount[primary_product_ID]) / biofuel_amount_no_bagasse[primary_product_ID]
                feed_buy[5] = 0 # buy feedstocks, but not bagasse
            else:
                feed_sell = contracted_feedstock_amount * (biofuel_amount[primary_product_ID] - K) / biofuel_amount_no_bagasse[primary_product_ID]
                feed_sell[5] = 0  # sell feedstocks, but not bagasse
            feed_buy /= 1 - config.mis_trans_loss - config.mis_storage_loss
        return feed_buy, feed_sell # 10 element array, column 0 for buy, column 1 for sale

    def cal_profit(self,farmer_list,product_prices,feed_prices_within_boundary,feed_prices_out_of_boundary,trans_costs,storage_costs):
        # function to calculate each year's profit of refinery
        # product_prices: prices of refinery price, 0 for ethanol, 1 for biodisel, 2 for bagasse, 3 for DDGS, 4 for glycerol
        #  feed_prices_within_boundary: prices of feedstocks within system boundary, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # feed_prices_out_of_boundary: prices of feedstocks out of system boundary, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # trans_costs: transportation costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane
        # storage_costs: storage costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass,
        #               5 for bagasse, 6 for sorghum, 7 for lipidcane

        contracted_patch_supply = copy.deepcopy(self.States['contracted_patch_supply'][-1])
        contracted_patch_price = copy.deepcopy(self.States['contracted_patch_price'][-1])
        contracted_patch_dist = copy.deepcopy(self.States['contracted_patch_dist'][-1])
        tech_type = self.Attributes['tech_type']

        if contracted_patch_supply.size==0:
            contracted_patch_supply=np.zeros((1,10))
            contracted_patch_price=np.zeros((1,10))
            contracted_patch_dist=np.zeros((1,10))
        else:
            contracted_patch_price = np.repeat(contracted_patch_price.reshape(contracted_patch_price.__len__(),1),contracted_patch_supply.shape[1],1)
            contracted_patch_dist = np.repeat(contracted_patch_dist.reshape(contracted_patch_dist.__len__(), 1),contracted_patch_supply.shape[1], 1)
        subsidy_rate = copy.deepcopy(self.Attributes['subsidy'])
        tax_rate = copy.deepcopy(self.Attributes['tax_rate'])
        tax_reduction = copy.deepcopy(self.Attributes['tax_reduction'])
        taxs = [tax_reduction, tax_rate]
        fix_cost = copy.deepcopy(self.Attributes['fix_cost'])
        via_cost = copy.deepcopy(self.Attributes['via_cost'])
        interest_payments = copy.deepcopy(self.Attributes['interest_payment'])
        interest_payment = 0
        # for ind in range(interest_payments.__len__()):
        #     if self.States['production_year'][ind] <= 20:
        #         interest_payment = interest_payment + interest_payments[ind]
        interest_payment = np.asarray(interest_payments).sum() ####  WILL NEED TO CHANGE THE CODE TO HAVE SEPERATE PRODUCTION YEAR LISTS FOR EACH NEW EXPANDED CAPACITIES

        feed_prices_within_boundary = np.append(feed_prices_within_boundary, feed_prices_within_boundary[6:8])
        feed_prices_out_of_boundary = np.append(feed_prices_out_of_boundary, feed_prices_out_of_boundary[6:8])
        trans_costs = np.append(trans_costs, trans_costs[6:8])
        storage_costs = np.append(storage_costs,storage_costs[6:8])  # expand the price array to the same length of feedstock arrary

        feed_contract_amount = contracted_patch_supply.sum(0) # feed_contract_amount is a 10 element 1d array
        purchased_feedstock = copy.deepcopy(self.States['purchased_feedstock'][-1]) # purchased_feedstock is a 10*2 matrix, column 0 is the feed from within boundary
        # temp = purchased_feedstock[0].sum(0)
        # purchased_feedstock = np.column_stack((temp,purchased_feedstock[1]))
        purchased_dist = copy.deepcopy(self.States['purchased_feed_dist'][-1])
        sold_feedstock = copy.deepcopy(self.States['sold_feedstock'][-1]) # sold_feedstock is a 10*2 matrix

        feed_contract_cost = (contracted_patch_supply * (contracted_patch_price +
                                                         np.repeat(storage_costs.reshape(1,storage_costs.__len__()),contracted_patch_supply.shape[0],0) +
                                                         contracted_patch_dist * np.repeat(trans_costs.reshape(1,trans_costs.__len__()),contracted_patch_supply.shape[0],0))).sum()
        feed_purchase_cost = (purchased_feedstock[0] * (np.repeat(feed_prices_within_boundary.reshape(1,feed_prices_within_boundary.size),purchased_feedstock[0].shape[0],0)+
                                                        np.repeat(purchased_dist.reshape(purchased_dist.size,1),purchased_feedstock[0].shape[1],1) *
                                                        np.repeat(trans_costs.reshape(1,trans_costs.__len__()),purchased_feedstock[0].shape[0],0) +
                                                        np.repeat(storage_costs.reshape(1, storage_costs.__len__()),purchased_feedstock[0].shape[0], 0))).sum()
        feed_purchase_cost = feed_purchase_cost + np.dot(purchased_feedstock[1], feed_prices_out_of_boundary + storage_costs + trans_costs * config.system_boundary_radius)
        feed_sale_revenue = np.dot(sold_feedstock[:, 0],feed_prices_within_boundary - storage_costs - trans_costs * config.patch_influence_range) + \
                             np.dot(sold_feedstock[:, 1],feed_prices_out_of_boundary - storage_costs - trans_costs * config.system_boundary_radius)

        total_feedstock_amount = feed_contract_amount + purchased_feedstock[0].sum(0) + purchased_feedstock[1] - sold_feedstock.sum(1)
        product_amount = self.produce_biofuel(total_feedstock_amount * (1 - config.mis_storage_loss - config.mis_trans_loss))

        if tech_type < 5:
            primary_product_ID = np.argmax(product_amount[0:2] * config.ethanol_equivilents)  # among ethaol and biodiesel, the one with larger ethanol equivilent is the primary product for refinery
        else:  # if the 'refinery' agent is a biofacility, then the primary product is set as electricity
            primary_product_ID = 5

        primary_product_amount = product_amount[primary_product_ID]
        ratio_capacity_produce = min(GMP.divide(self.Attributes['capacity'][-1], primary_product_amount),1)
        product_amount = product_amount * ratio_capacity_produce
        feed_purchase_cost = feed_purchase_cost * ratio_capacity_produce

        subsidy = np.dot(product_amount[0:2], config.ethanol_equivilents) * subsidy_rate
        if tech_type >= 5: # subsidy for bioelectricity
            subsidy += product_amount[5]*subsidy_rate
        revenue = np.dot(product_amount, product_prices) + feed_sale_revenue - feed_sale_revenue # !!!!!!!! assume additional feedstock is not sold

        if self.States['production_year'][-1]<0:
            production_cost = 0
        else:
            production_cost = fix_cost * self.Attributes['capacity'][-1] + via_cost * primary_product_amount * ratio_capacity_produce

        tax = (revenue + subsidy - production_cost - feed_contract_cost - feed_purchase_cost - interest_payment - taxs[0]) * taxs[1]
        tax=max(0,tax)
        profit = revenue + subsidy - production_cost - feed_contract_cost - feed_purchase_cost - tax - interest_payment

        # check if contracts will be ended
        contracted_patch_price = copy.deepcopy(self.States['contracted_patch_price'][-1])
        contracted_patch_dist = copy.deepcopy(self.States['contracted_patch_dist'][-1])
        contracted_patch_amount = copy.deepcopy(self.States['contracted_patch_amount'][-1])
        contracted_patch_ID = copy.deepcopy(self.States['contracted_patch_ID'][-1]).astype(int)
        contracted_farmer_ID = copy.deepcopy(self.States['contracted_farmer_ID'][-1].astype(int))
        N_contract = contracted_patch_ID.__len__()
        # for ind in range(N_contract):
        ind = 0
        temp_ind = 0
        while ind < N_contract:
            patch_ID_temp = np.argwhere(
                farmer_list[contracted_farmer_ID[ind-temp_ind]].Attributes['patch_ID'] == contracted_patch_ID[ind-temp_ind])
            temp1_land_use = farmer_list[contracted_farmer_ID[ind-temp_ind]].States['land_use'][-1][patch_ID_temp]
            temp2_peren_age = farmer_list[contracted_farmer_ID[ind-temp_ind]].States['peren_age'][-1][patch_ID_temp]
            if temp2_peren_age >= GMP.cal_contract_length(temp1_land_use):
                contracted_patch_price = np.delete(contracted_patch_price, ind-temp_ind)
                contracted_patch_dist = np.delete(contracted_patch_dist, ind-temp_ind)
                contracted_patch_amount = np.delete(contracted_patch_amount, ind-temp_ind, 0)
                contracted_patch_ID = np.delete(contracted_patch_ID, ind-temp_ind)
                contracted_farmer_ID = np.delete(contracted_farmer_ID, ind-temp_ind)
                temp_ind += 1
            ind += 1

        self.States['contracted_patch_price'][-1] = contracted_patch_price
        self.States['contracted_patch_dist'][-1] = contracted_patch_dist
        self.States['contracted_patch_amount'][-1] = contracted_patch_amount
        self.States['contracted_patch_ID'][-1] = contracted_patch_ID
        self.States['contracted_farmer_ID'][-1] = contracted_farmer_ID

        self.States['accepted_subsidy'].append(subsidy)
        self.States['biofuel_production'].append(product_amount[0:2])
        self.States['byproduct_production'].append(product_amount[2:])
        self.States['production_year'].append(self.States['production_year'][-1]+1)
        return total_feedstock_amount, product_amount, profit

    def cal_stop_production(self):
        # function to determine if the refinery is going to close
        N_years = 0 # the number of years in production
        terminal_cap = self.Attributes['capacity'][-1]
        for cap in reversed(self.Attributes['capacity']):
            if cap < terminal_cap:
                break
            else:
                N_years += 1

        comu_loss = 0 # the comulative loss
        if self.States['profit'][-1] <0:
            i = 1
            while i < self.States['profit'].__len__():
                if self.States['profit'][-i]<0:
                    comu_loss = comu_loss + self.States['profit'][-i]
                    i = i + 1
                else:
                    break

        if np.abs(comu_loss) > config.allowable_defecit * self.Attributes['invest']:
            close = 1
        elif N_years > config.refinery_life_span:
            close = 1
        else:
            close = 0

        return close, N_years

    def pick_contracts(self,contracted_patch_amount):
        # function to select the contracts when the total amount of potential contract exceeds the capacity of the refinery
        # contracted_patch_amount: a 10*N matrix of the contracted feedstock amount
        feed_contract_amount = contracted_patch_amount.sum(0)  # feed_contract_amount is a 10 element 1d array
        feed_contract_amount[5] = self.Attributes['bagasse_amount']
        tech_type = self.Attributes['tech_type']
        product_amount = self.produce_biofuel(feed_contract_amount * (1 - config.mis_storage_loss - config.mis_trans_loss))
        if tech_type < 5:
            main_product_type = np.argmax(product_amount[0:2] * config.ethanol_equivilents)  # among ethaol and biodiesel, the one with larger ethanol equivilent is the primary product for refinery
            main_product_amount = max(product_amount[0:2] * config.ethanol_equivilents)
        else:  # if the 'refinery' agent is a biofacility, then the primary product is set as electricity
            main_product_type = 5
            main_product_amount = product_amount[5]

        K = copy.deepcopy(self.Attributes['capacity'][-1])
        if main_product_amount <= K:
            selected_ID = np.ones(contracted_patch_amount.shape[0]) > 0 # the IDs of contracts finally selected by the refinery, could be T/F, or the ID numbers
        else: # select the first few contact until the capacity is filled
            prod_amount_contracts = np.zeros(contracted_patch_amount.shape[0])
            for i in range(contracted_patch_amount.shape[0]):
                prod_amount_contracts[i] = self.produce_biofuel(contracted_patch_amount[i,:]*
                                                                (1 - config.mis_storage_loss - config.mis_trans_loss))[main_product_type]
            sort_ID = np.argsort(prod_amount_contracts)
            j = -1
            cumu_prod = 0
            selected_ID = np.empty(0,int)
            feed_temp = np.zeros(10)
            feed_temp[5] = self.Attributes['bagasse_amount']
            prod_bagasse_only = copy.deepcopy(self.produce_biofuel(feed_temp*(1 - config.mis_storage_loss - config.mis_trans_loss)))
            while cumu_prod < K - prod_bagasse_only[main_product_type]:
                selected_ID = np.append(selected_ID,int(sort_ID[j]))
                cumu_prod = cumu_prod + prod_amount_contracts[sort_ID[j]]
                j = j - 1
        return selected_ID

    def if_continue_contract(self,contracted_patch_amount):
        # function to identify if the refinery still needs to make more feedstock contracts
        # contracted_patch_amount: a 10*N matrix of the contracted feedstock amount
        tech_type = self.Attributes['tech_type']
        feed_contract_amount = contracted_patch_amount.sum(0)  # feed_contract_amount is a 10 element 1d array
        feed_contract_amount[5] = self.Attributes['bagasse_amount']
        product_amount = self.produce_biofuel(feed_contract_amount * (1 - config.mis_storage_loss - config.mis_trans_loss))

        if tech_type < 5:
            main_product_type = np.argmax(product_amount[0:2] * config.ethanol_equivilents)  # among ethaol and biodiesel, the one with larger ethanol equivilent is the primary product for refinery
            main_product_amount = max(product_amount[0:2] * config.ethanol_equivilents)
        else:  # if the 'refinery' agent is a biofacility, then the primary product is set as electricity
            main_product_type = 5
            main_product_amount = product_amount[5]

        K = copy.deepcopy(self.Attributes['capacity'][-1])
        if main_product_amount <= K:
            is_continue = 1
        else:
            is_continue = 0
        return is_continue

    def make_contracts(self,farmer_list,ref_list,feed_prices,trans_costs,storage_costs,PBE,market_demand,market_supply,community_list,year):
        # function to identify the contract and contract prices with farmers farmer_list: a list of farmer agents
        # feed_prices: prices of feedstocks out of system boundary, 0 for corn, 1 for soy, 2 for corn stover,
        # 3 for miscanthus, 4 for switchgrass, 5 for bagasse, 6 for sorghum, 7 for lipidcane trans_costs:
        # transportation costs for feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus,
        # 4 for switchgrass, 5 for bagasse, 6 for sorghum, 7 for lipidcane storage_costs: storage costs for
        # feedstock, 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus, 4 for switchgrass, 5 for bagasse,
        # 6 for sorghum, 7 for lipidcane PBE: 0 for corn, 1 for soy, 2 for corn stover, 3 for miscanthus,
        # 4 for switchgrass, 5 for bagasse, 6 for sorghum, 7 for lipidcane market_demand, market_supply: the total
        # demand and supply for the feedstock required by the refinery in the market community_list: a list of local
        # communities

        tech_type = copy.deepcopy(self.Attributes['tech_type'])
        # contract_feed_type: the feedstock type that will be contracted, 1 for mis,swtich, or stover, 2 for sorghum, 3 for cane
        if tech_type == 1:
            contract_feed_type = 0
            ref_price = PBE[0]
            trans_cost = trans_costs[0]
            storage_cost = storage_costs[0]
            market_feedstock_price = feed_prices[0]
        elif (tech_type == np.asarray([2,5,6,7])).sum()>0:
            contract_feed_type = 1
            ref_price = PBE[3]
            trans_cost = trans_costs[3]
            storage_cost = storage_costs[3]
            market_feedstock_price = feed_prices[3]
        elif (tech_type == 3) & (tech_type == 4):
            contract_feed_type = 3
            ref_price = PBE[6]
            trans_cost = trans_costs[6]
            storage_cost = storage_costs[6]
            market_feedstock_price = feed_prices[6]

        dist_farmer = copy.deepcopy(self.Attributes['dist_farmer'])
        ID_neighbor_farmer = np.argwhere(dist_farmer <= config.patch_influence_range)
        N_neighbors = ID_neighbor_farmer.shape[0]
        dist_farmer = dist_farmer[ID_neighbor_farmer]

        contracted_patch_price = copy.deepcopy(self.States['contracted_patch_price'][-1])
        contracted_patch_dist = copy.deepcopy(self.States['contracted_patch_dist'][-1])
        contracted_patch_amount = copy.deepcopy(self.States['contracted_patch_amount'][-1])
        contracted_patch_ID = copy.deepcopy(self.States['contracted_patch_ID'][-1]).astype(int)
        contracted_farmer_ID = copy.deepcopy(self.States['contracted_farmer_ID'][-1].astype(int))

        contracted_patch_ID_all_refs = np.empty(0)
        contracted_farmer_ID_all_refs = np.empty(0)
        for i in range(ref_list.__len__()):
            contracted_farmer_ID_all_refs = np.concatenate([contracted_farmer_ID_all_refs, ref_list[i].States['contracted_farmer_ID'][-1].astype(int)])
            contracted_patch_ID_all_refs = np.concatenate([contracted_patch_ID_all_refs, ref_list[i].States['contracted_patch_ID'][-1].astype(int)])

        N_contract = contracted_patch_ID.__len__()
        for ind in range(N_contract):
            patch_ID_temp = np.argwhere(farmer_list[contracted_farmer_ID[ind]].Attributes['patch_ID']==contracted_patch_ID[ind])
            temp1_land_use = farmer_list[contracted_farmer_ID[ind]].States['land_use'][-1][patch_ID_temp]
            temp2_peren_age = farmer_list[contracted_farmer_ID[ind]].States['peren_age'][-1][patch_ID_temp]
            if temp2_peren_age >= GMP.cal_contract_length(temp1_land_use):
                contracted_patch_price = np.delete(contracted_patch_price,ind)
                contracted_patch_dist = np.delete(contracted_patch_dist, ind)
                contracted_patch_amount = np.delete(contracted_patch_amount, ind, 0)
                contracted_patch_ID = np.delete(contracted_patch_ID, ind)
                contracted_farmer_ID = np.delete(contracted_farmer_ID, ind)

        if_new_contract = self.if_continue_contract(contracted_patch_amount)
        if if_new_contract == 0:
            pass
        else:
            for idx in np.random.permutation(N_neighbors):
                farmer_agent = farmer_list[ID_neighbor_farmer[idx][0]]
                risks = farmer_agent.States['climate_forecasts'][-1]
                is_flood = copy.deepcopy(risks[0])
                is_drought = copy.deepcopy(risks[1])
                contract_max_price = ref_price - trans_cost*dist_farmer[idx] - storage_cost
                LB = market_feedstock_price # farmer's minimum acceptable price
                # land_uses_n = -999 * np.ones(is_contract.__len__())
                if LB > contract_max_price:
                    continue
                else:
                    contract_price = GME.cal_bargain_price(LB,contract_max_price,market_demand,market_supply)
                    is_contract, opt_land_use = farmer_agent.identify_contract_patches(community_list,contract_feed_type,contract_price,year)

                    for j in range(is_contract.size):
                        if is_contract[j] < 1:
                            continue
                        elif np.isin(farmer_agent.Attributes['patch_ID'][j],contracted_patch_ID_all_refs):
                            continue
                        else:
                            farmer_agent.Temp['contract_land_use'][j] = opt_land_use[j]
                            if opt_land_use[j]>10:
                                land_use_n = int(opt_land_use[j]/10)
                            else:
                                land_use_n = int(opt_land_use[j])
                            output = GMP.look_up_table_crop_no_physical_model(farmer_agent.Attributes['patch_ID'][j],is_flood,
                                                                          is_drought,farmer_agent.Attributes['patch_slope'][j],
                                                                          farmer_agent.States['land_use'][-1][j],land_use_n,0,
                                                                          2,1,1)
                            feed_amount_patch = farmer_agent.Attributes['patch_areas'][j] * output['yield']
                            market_demand -= feed_amount_patch
                            if farmer_agent.States['land_use'][-1][j] == land_use_n:
                                market_supply = market_supply - feed_amount_patch
                            else:
                                pass # adjust the market demand and supply in real time after the contract is made
                            contracted_patch_dist = np.append(contracted_patch_dist,dist_farmer[idx])
                            contracted_patch_price = np.append(contracted_patch_price,contract_price)
                            contracted_amount_temp = np.zeros((1, 10))
                            feed_type = GMP.land_use_to_feed_ID(land_use_n,tech_type)
                            contracted_amount_temp[0,int(feed_type)] = feed_amount_patch
                            contracted_patch_amount = np.append(contracted_patch_amount, contracted_amount_temp,axis=0)
                            contracted_patch_ID = np.append(contracted_patch_ID,farmer_agent.Attributes['patch_ID'][j])
                            contracted_farmer_ID = np.append(contracted_farmer_ID,ID_neighbor_farmer[idx])
                        is_continue = self.if_continue_contract(contracted_patch_amount)
                        if is_continue == 0:
                            break
                # farmer_agent.Temp['contract_land_use'] = land_uses_n
                is_continue = self.if_continue_contract(contracted_patch_amount)
                if is_continue == 0:
                    break

        # selected_ID = self.pick_contracts(contracted_patch_amount)
        self.States['contracted_patch_price'].append(contracted_patch_price)
        self.States['contracted_patch_dist'].append(contracted_patch_dist)
        self.States['contracted_patch_amount'].append(contracted_patch_amount)
        self.States['contracted_patch_ID'].append(contracted_patch_ID)
        self.States['contracted_farmer_ID'].append(contracted_farmer_ID)
        return market_demand, market_supply

    def check_contract_continuity(self,farmer_list):
        # function to check if certain contract is at the end of its length
        contracted_patch_price = copy.deepcopy(self.States['contracted_patch_price'][-1])
        contracted_patch_dist = copy.deepcopy(self.States['contracted_patch_dist'][-1])
        contracted_patch_amount = copy.deepcopy(self.States['contracted_patch_amount'][-1])
        contracted_patch_ID = copy.deepcopy(self.States['contracted_patch_ID'][-1]).astype(int)
        contracted_farmer_ID = copy.deepcopy(self.States['contracted_farmer_ID'][-1].astype(int))
        N_contract = contracted_patch_ID.__len__()
        for ind in range(N_contract):
            patch_ID_temp = np.argwhere(
                farmer_list[contracted_farmer_ID[ind]].Attributes['patch_ID'] == contracted_patch_ID[ind])
            temp1_land_use = farmer_list[contracted_farmer_ID[ind]].States['land_use'][-1][patch_ID_temp]
            temp2_peren_age = farmer_list[contracted_farmer_ID[ind]].States['peren_age'][-1][patch_ID_temp]
            if temp2_peren_age >= GMP.cal_contract_length(temp1_land_use):
                contracted_patch_price = np.delete(contracted_patch_price, ind)
                contracted_patch_dist = np.delete(contracted_patch_dist, ind)
                contracted_patch_amount = np.delete(contracted_patch_amount, ind, 0)
                contracted_patch_ID = np.delete(contracted_patch_ID, ind)
                contracted_farmer_ID = np.delete(contracted_farmer_ID, ind)

        return contracted_farmer_ID, contracted_patch_ID, contracted_patch_amount, contracted_patch_dist, contracted_patch_price
