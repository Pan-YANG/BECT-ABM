import numpy as np

"""
note some parameters are in the csv files
"""

folder = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Model/simulations/baseline'
result_file = './results/ABM_result_small_subsidy_BCAP_TMDL.out'
simu_horizon = 20


# farmer related parameters
land_rent = 218 * 2.47  # in $/ha
marginal_land_rent_adj = 0.5 # the land rent of marginal land
enhanced_neighbor_impact = [0, 1, 2] # the level of neighborhood impact
farm_type_change_prob=0

# industry related parameters
learn_by_do_rate = 4*np.log(0.95)/np.log(2) # the cost updating rate for learning by doing
# learn_by_do_rate = 0 # the cost updating rate for learning by doing, assuming no learning by doing
base_cap_for_learn_by_do = 200*10**6 # the base capacity for learning by doing, L
base_BCHP_cap_for_learn_by_do = 4 # the base capacity of BCHP, in unit of BCHP plant
base_feed_amount_for_learn_by_do = 4*10**4 # the base feedstock amount for the learning by doing of supply chain business, ton
allowable_defecit = 0.4

is_ref_learn_by_do = 1 # if the refinery production cost need to be updated based on learning by doing
ref_inv_cost_adj_his = np.ones((simu_horizon + 1, 4))  # the investment cost adjustment of learning by doing
ref_inv_cost_adj_his[np.round(simu_horizon/2).astype(int):,1] = 0.5
ref_pro_cost_adj_his = np.ones((simu_horizon + 1, 4))  # the production cost adjustment of learning by doing
ref_pro_cost_adj_his[np.round(simu_horizon/2).astype(int):,1] = 0.5

# consumer related parameters
ini_WP = 0.01    # initial willingness to pay extra for cellulosic ethanol in $/gallon
IRW = 1.5      # increasing rate factor of WP
max_WP = 0.5    # maximum value of WP

price_update_rate = 0.3 # the parameter for forecasting prices

# government related parameters
maintain_RFS = 0 # if gov determined to maintain the RFS mandate
CRP_subsidy = 600    #$/ha
CRP_relax = 0 # if perennial grass harvesting in CRP land is allowed

# TMDL_subsidy = 60 * 2.47
TMDL_subsidy=0

# BCAP_subsidy = 0 # in $/ha
# BCAP_cost_share = 0
BCAP_subsidy = 1000 * 2.47/15 # in $/ha
BCAP_cost_share = 0.5

tax_deduction = 0
tax_rate = 0

carbon_price = 0.1 # $/t CO2e
nitrogen_price = 0.1 # $/kg

