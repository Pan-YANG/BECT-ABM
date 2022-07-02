import shapely.speedups
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import gdal
import rasterio
from scipy.interpolate import griddata
import numpy as np
from rasterio.transform import from_origin
from rasterstats import zonal_stats
import copy

dir_with_cvs = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/daycent data'
os.chdir(dir_with_cvs)


loc_dir = 'C:/Users/pyangac/Documents/research/cabbi/ABM_pilot/Version_3/data/GIS/'
farmer_map = gpd.read_file(loc_dir+'farms_sagamon_farm1000_with_urban.shp')
farmer_map.columns=['uni_ID', 'lat', 'long', 'SS_GROUP', 'area', 'patch_ID', 'farmer_ID','geometry']
farmer_map=farmer_map.drop(farmer_map[farmer_map['patch_ID'] == -1].index[0])
farmer_map = farmer_map.to_crs("EPSG:4326")


def daycent_to_csv(crop_type):
    daycent_data = pd.read_csv('watershed_daycent_results_'+crop_type+'.csv')
    for i in range(30):
        try:
            daycent_to_csv_loop(crop_type,daycent_data,i)
        except ValueError:
            print(str(2018+i) + "is not here")

def daycent_to_csv_loop(crop_type,daycent_data,i):
    daycent_data_temp = daycent_data.loc[daycent_data['time'] == 2018+i]
    print(2018+i)

    # interpolate and save the point data into raster
    grid_x, grid_y = np.mgrid[-90.69866901694759:-88.00182533360399:1014j, 39.1004311854944:40.7994581134248:639j]
    points = np.vstack((daycent_data_temp['long'].ravel(), daycent_data_temp['lat'].ravel())).T
    grid_cgrain = griddata(points, daycent_data_temp['crmvst'].ravel(), (grid_x, grid_y), method='linear')
    grid_somtc = griddata(points, daycent_data_temp['somtc'].ravel(), (grid_x, grid_y), method='linear')
    grid_strmac2 = griddata(points, daycent_data_temp['strmac.2.'].ravel(), (grid_x, grid_y), method='linear')
    grid_fertappN = griddata(points, daycent_data_temp['fertapp.N.'].ravel(), (grid_x, grid_y), method='linear')
    grid_annet = griddata(points, daycent_data_temp['annet'].ravel(), (grid_x, grid_y), method='linear')

    with rasterio.open('Temp.tif', 'w', driver='GTiff', height=grid_cgrain.shape[1],
                       width=grid_cgrain.shape[0], count=5, dtype=grid_cgrain.dtype, nodata=-999,
                       crs="EPSG:4326",
                       transform=from_origin(-90.69866901694759, 40.7994581134248, 0.002658776525422013,
                                             0.002658776525422013)) as dst:
        dst.write(grid_cgrain, 1)
        dst.write(grid_somtc, 2)
        dst.write(grid_strmac2, 3)
        dst.write(grid_fertappN, 4)
        dst.write(grid_annet, 5)

    # perform zonal statistic for shapes
    temp_result = pd.DataFrame(zonal_stats(vectors=farmer_map['geometry'], raster='Temp.tif', band=1, stats='mean'))
    temp_result.columns=['crmvst (gC/m2)']
    temp_result['somtc (gC/m2)'] = \
    pd.DataFrame(zonal_stats(vectors=farmer_map['geometry'], raster='Temp.tif', band=2, stats='mean'))['mean']
    temp_result['strmac.2. (g N/m2)'] = \
    pd.DataFrame(zonal_stats(vectors=farmer_map['geometry'], raster='Temp.tif', band=3, stats='mean'))['mean']
    temp_result['fertapp.N. (g N/m2)'] = \
    pd.DataFrame(zonal_stats(vectors=farmer_map['geometry'], raster='Temp.tif', band=4, stats='mean'))['mean']
    temp_result['annet (cm/year)'] = \
    pd.DataFrame(zonal_stats(vectors=farmer_map['geometry'], raster='Temp.tif', band=5, stats='mean'))['mean']
    temp_result.to_csv(crop_type+'_'+str(2018+i)+'.csv')


def csv_to_res_mat(crop_type_list):
    # convert all csv files into response matrix
    N_crop = len(crop_type_list) # number of crops considered in the response matrix
    for i in range(30):
        try:
            yield_df = pd.DataFrame()
            carbon_df = pd.DataFrame()
            Nrealease_df = pd.DataFrame()
            fertilizer_df = pd.DataFrame()
            water_df = pd.DataFrame()
            for j in range(N_crop):
                temp_df = pd.read_csv(crop_type_list[j]+'_'+str(2018+i)+'.csv')
                yield_df = pd.concat([yield_df, temp_df['crmvst (gC/m2)']], axis=1)
                carbon_df = pd.concat([carbon_df, temp_df['somtc (gC/m2)']], axis=1)
                Nrealease_df = pd.concat([Nrealease_df, temp_df['strmac.2. (g N/m2)']], axis=1)
                fertilizer_df = pd.concat([fertilizer_df, temp_df['fertapp.N. (g N/m2)']], axis=1)
                water_df = pd.concat([water_df, temp_df['annet (cm/year)']], axis=1)

            yield_df.columns = crop_type_list
            carbon_df.columns = crop_type_list
            Nrealease_df.columns = crop_type_list
            fertilizer_df.columns = crop_type_list
            water_df.columns = crop_type_list

            yield_df = yield_df/43.5 # the unit is ton/ha
            carbon_df = carbon_df/100 # the unit is ton/ha
            Nrealease_df = Nrealease_df/100 # the unit is ton/ha
            fertilizer_df = fertilizer_df*10 # the unit is kg/ha
            water_df = water_df * 10 ** (-4) # the unit is mcm/(ha*year)

            # output response matrix
            yield_df.to_csv('patch_yield_table_' + crop_type_list[0]+ str(1 + i) + '.csv',index_label='ton/ha')
            carbon_df.to_csv('patch_carbon_table_' +crop_type_list[0] + str(1 + i) + '.csv',index_label='ton/ha')
            Nrealease_df.to_csv('patch_N_loads_' + crop_type_list[0] + str(1 + i) + '.csv',index_label='ton/ha')
            fertilizer_df.to_csv('patch_ferti_table_' + crop_type_list[0] + str(1 + i) + '.csv',index_label='kg/ha')
            water_df.to_csv('patch_water_use_' + crop_type_list[0] + str(1 + i) + '.csv',index_label='mcm/(ha*year)')
        except FileNotFoundError:
            continue

def fill_missing_values(crop_type_list):
    N_crop = len(crop_type_list)
    for i in range(N_crop):
        for j in range(30):
            try:
                file_name = 'patch_yield_table_' +crop_type_list[i] + str(1 + j) + '.csv'
                data = pd.read_csv(file_name)
                data = data.fillna(method='ffill')
                data.to_csv(file_name, index=False)

                file_name = 'patch_carbon_table_' + crop_type_list[i] + str(1 + j) + '.csv'
                data = pd.read_csv(file_name)
                data = data.fillna(method='ffill')
                data.to_csv(file_name, index=False)

                file_name = 'patch_N_loads_' + crop_type_list[i] + str(1 + j) + '.csv'
                data = pd.read_csv(file_name)
                data = data.fillna(method='ffill')
                data.to_csv(file_name, index=False)

                file_name = 'patch_ferti_table_' + crop_type_list[i] + str(1 + j) + '.csv'
                data = pd.read_csv(file_name)
                data = data.fillna(method='ffill')
                data.to_csv(file_name, index=False)

                file_name = 'patch_water_use_' + crop_type_list[i] + str(1 + j) + '.csv'
                data = pd.read_csv(file_name)
                data = data.fillna(method='ffill')
                data.to_csv(file_name, index=False)
            except FileNotFoundError:
                continue


def deleting_first_row(crop_type_list):
    N_crop = len(crop_type_list)
    for i in range(N_crop):
        for j in range(30):
            file_name = 'patch_yield_table_' + crop_type_list[i] + str(1 + j) + '.csv'
            data = pd.read_csv(file_name)
            data=data.drop(data.columns[0], axis=1)
            data.to_csv(file_name, index=False)

            file_name = 'patch_carbon_table_' + crop_type_list[i] + str(1 + j) + '.csv'
            data = pd.read_csv(file_name)
            data=data.drop(data.columns[0], axis=1)
            data.to_csv(file_name, index=False)

            file_name = 'patch_N_loads_' + crop_type_list[i] + str(1 + j) + '.csv'
            data = pd.read_csv(file_name)
            data=data.drop(data.columns[0], axis=1)
            data.to_csv(file_name, index=False)

            file_name = 'patch_ferti_table_' + crop_type_list[i] + str(1 + j) + '.csv'
            data = pd.read_csv(file_name)
            data=data.drop(data.columns[0], axis=1)
            data.to_csv(file_name, index=False)

            file_name = 'patch_water_use_' + crop_type_list[i] + str(1 + j) + '.csv'
            data = pd.read_csv(file_name)
            data=data.drop(data.columns[0], axis=1)
            data.to_csv(file_name, index=False)

def generate_response_matrix(crop_type_list):
    N_crop = len(crop_type_list)
    for i in range(30):
        data_yield =pd.DataFrame({'contcorn' : []})
        data_carbon =pd.DataFrame({'contcorn' : []})
        data_N_load =pd.DataFrame({'contcorn' : []})
        data_ferti =pd.DataFrame({'contcorn' : []})
        data_water =pd.DataFrame({'contcorn' : []})
        for j in range(N_crop):
            file_name = 'patch_yield_table_' + crop_type_list[j] + str(1 + i) + '.csv'
            data_temp = pd.read_csv(file_name)
            data_yield[crop_type_list[j]] = data_temp[crop_type_list[j]]
            
            file_name = 'patch_carbon_table_' + crop_type_list[j] + str(1 + i) + '.csv'
            data_temp = pd.read_csv(file_name)
            data_carbon[crop_type_list[j]] = data_temp[crop_type_list[j]]
            
            file_name = 'patch_N_loads_' + crop_type_list[j] + str(1 + i) + '.csv'
            data_temp = pd.read_csv(file_name)
            data_N_load[crop_type_list[j]] = data_temp[crop_type_list[j]]
            
            file_name = 'patch_ferti_table_' + crop_type_list[j] + str(1 + i) + '.csv'
            data_temp = pd.read_csv(file_name)
            data_ferti[crop_type_list[j]] = data_temp[crop_type_list[j]]
            
            file_name = 'patch_water_use_' + crop_type_list[j] + str(1 + i) + '.csv'
            data_temp = pd.read_csv(file_name)
            data_water[crop_type_list[j]] = data_temp[crop_type_list[j]]
        
        data_yield['sorghum'] = 0
        data_yield['cane'] = 0
        
        data_carbon['sorghum']=data_carbon['miscanthus']
        data_carbon['cane']=data_carbon['miscanthus']
        data_carbon['fallow']=data_carbon['miscanthus']
        data_carbon['CRP']=data_carbon['miscanthus']
        
        data_N_load['sorghum']=data_N_load['miscanthus']
        data_N_load['cane']=data_N_load['miscanthus']
        data_N_load['fallow']=data_N_load['miscanthus']
        data_N_load['CRP']=data_N_load['miscanthus']
        
        data_ferti['sorghum']=data_ferti['miscanthus']
        data_ferti['cane']=data_ferti['miscanthus']
        data_ferti['fallow']=0
        data_ferti['CRP']=0
        
        data_water['sorghum']=data_water['switchgrass']
        data_water['cane']=data_water['switchgrass']
        data_water['fallow']=data_water['switchgrass']
        data_water['CRP']=data_water['switchgrass']
        
        data_yield.to_csv('patch_yield_table'+str(i+1)+'.csv',index_label='ton/ha')
        data_carbon.to_csv('patch_carbon_table_'+str(i+1)+'.csv',index_label='ton/ha')
        data_N_load.to_csv('patch_N_loads_'+str(i+1)+'.csv',index_label='ton/ha')
        data_ferti.to_csv('patch_ferti_table_'+str(i+1)+'.csv',index_label='kg/ha')
        data_water.to_csv('patch_water_use_'+str(i+1)+'.csv',index_label='mcm/(ha*year)')
        
        
            
            

daycent_to_csv('cornsoy')
csv_to_res_mat(['cornsoy'])

daycent_to_csv('contcorn')
csv_to_res_mat(['contcorn'])

daycent_to_csv('miscanthus')
csv_to_res_mat(['miscanthus'])

daycent_to_csv('switchgrass')
csv_to_res_mat(['switchgrass'])


fill_missing_values(['cornsoy','contcorn','miscanthus','switchgrass'])

# deleting_first_row(['cornsoy','contcorn','miscanthus','switchgrass'])

generate_response_matrix(['contcorn','cornsoy','miscanthus','switchgrass'])





# dataset = rasterio.open('Temp.tif')
# temp1 = dataset.transform*(0,0)
# temp2 = dataset.transform*(dataset.shape[1],dataset.shape[0])
# im_extent = (temp1[0],temp2[0],temp2[1],temp1[1])
# data=copy.copy(dataset.read(3))
# data[data==dataset.nodata] = None
# # print(data[200,103])
# fig, axes = plt.subplots()
# img=axes.imshow(data,interpolation='none',extent=im_extent)
# farmer_map.geometry.boundary.plot(ax=axes,color=None, edgecolor='k', linewidth=1.5)

