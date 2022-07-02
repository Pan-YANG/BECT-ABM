import config
import numpy as np
import agents
import general_methods_physical as GMP
import gen_agents
import parameters as params
import copy
import os

os.chdir(params.folder)

CRP_subsidy = params.CRP_subsidy    #$/ha
CRP_relax = params.CRP_relax

TMDL_subsidy = params.TMDL_subsidy

# BCAP_subsidy = 0 # in $/ha
# BCAP_cost_share = 0
BCAP_subsidy = params.BCAP_subsidy # in $/ha
BCAP_cost_share = params.BCAP_cost_share

tax_deduction = params.tax_deduction
tax_rate = params.tax_rate

# carbon_price = 13 # $/t CO2e
# nitrogen_price = 2.2 # $/kg
carbon_price = params.carbon_price # $/t CO2e
nitrogen_price = params.nitrogen_price # $/kg

# TMDL = 56500
TMDL = np.asarray([135,135,135,135,135,135,135,135,135,135,135,130,130,130,130,130,130,130,130,130,130,125,125,125,125,125,125,125,125,125,125,])
slope_limits = np.asarray([0.055,0.05,0.045,0.04,0.035,0.03,])
TMDL_N_limits = np.asarray([0.155,0.15,0.145,0.14,0.135,0.13,0.125,0.12,0.115,0.11,0.105,0.1])

# RFS
RFS_volume = np.loadtxt('./data/RFS_volume.csv',delimiter=',',skiprows = 1,usecols=(1))
scaling_factor = 0.5
RFS_signal = [0] # the initial signal of RFS mandate change

# refinery related
gov_ref_tax_subsidy = np.loadtxt('./data/gov_ref_tax_subsidy.csv',delimiter=',',skiprows = 1)
ref_subsidys = copy.deepcopy(gov_ref_tax_subsidy[:,1:4]) # 0 for cost share (in $), 1 for cost share (in %), 2 for production subsidy (in $/L ethanol equivilent)
ref_taxs = copy.deepcopy(gov_ref_tax_subsidy[:,4:]) # 0 for tax deduction, 1 for tax rate




