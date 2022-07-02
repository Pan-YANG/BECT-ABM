import numpy as np
def cal_ref_contract_patch_information(ref_list,ref_no,var_name,output_type,N_year):
    output = np.zeros(N_year)
    temp = 0
    for i in range(ref_list[ref_no].States[var_name].__len__()):
        if i < ref_list[ref_no].Attributes['start_year']:
            output[i] = 0
            temp = i + 1
        elif (i - temp) < ref_list[ref_no].States['biofuel_production'].__len__():
            if output_type == 'sum':
                output[i] = ref_list[ref_no].States[var_name][i].sum()
            elif output_type == 'mean':
                output[i] = ref_list[ref_no].States[var_name][i].mean()
            elif output_type == 'size':
                output[i] = ref_list[ref_no].States[var_name][i].size
    return output


ref_contract_amount_PC1 = test.cal_ref_contract_patch_information(data_PC1['ref_list'],11,'contracted_patch_supply','sum',30)
ref_contract_amount_CRP1 = test.cal_ref_contract_patch_information(data_CRP1['ref_list'],10,'contracted_patch_supply','sum',30) + \
                           test.cal_ref_contract_patch_information(data_CRP1['ref_list'],11,'contracted_patch_supply','sum',30) + \
                           test.cal_ref_contract_patch_information(data_CRP1['ref_list'],12,'contracted_patch_supply','sum',30)
plt.plot(ref_contract_amount_PC1/(25*10**6))
plt.plot(ref_contract_amount_CRP1/(30*10**6))
plt.legend(['PC1','CRP1'])


