import numpy as np
import pandas as pd


# @st.cache()  # We use this to cache the info and not load the data every time we scroll up/down
def load_sov_data(filepath):
    sov_data_ = pd.read_csv(FILES_LOCATION + filepath)
    sov_data_.drop_duplicates(inplace=True)
    return sov_data_


def load_shap_data2(filepath):
    shap_data_ = pd.read_csv(filepath)
    ctr_curve_ = [0.233, 0.205, 0.133, .087, .063, .047, .038, .031, .027, .023, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0]

    available_positions = len(shap_data_['Position Organic'].unique())
    ctr_curve_ = ctr_curve_[:available_positions]

    est_relevant_tr_ = pd.DataFrame({"CTR": ctr_curve_, "Position": shap_data_['Position Organic'].unique()})

    shap_data_ = pd.merge(shap_data_, est_relevant_tr_, how='left', left_on=['Position Organic'], right_on=['Position'])
    shap_data_.drop(['Position'], axis=1, inplace=True)
    return shap_data_


def sov_shap_merge2(shap_data_goes_here, msv_df_goes_here):
    msv_df_goes_here.rename(columns={'grouping_keyword': 'Keyword', 'URL': 'Final URL'}, inplace=True)
    msv_df_goes_here = msv_df_goes_here[
        ['Final URL', 'Keyword', 'Volume', 'Est. Relevant Traffic', 'state_id', 'state_name', 'city']]
    msv_df_goes_here = msv_df_goes_here.drop_duplicates()

    msv_df_goes_here = pd.pivot_table(msv_df_goes_here, index=['Final URL', 'Keyword'],
                                      values=['Volume', 'Est. Relevant Traffic'], aggfunc=np.max)
    msv_df_goes_here = msv_df_goes_here.reset_index()
    msv_df_goes_here['Volume'] = msv_df_goes_here['Volume'].fillna(0)
    data_ = pd.merge(shap_data_goes_here, msv_df_goes_here, on=['Keyword', 'Final URL'], how='left')
    # data_.drop_duplicates(inplace=True) ## -- Angela: commenting this out since duplicate removal was reducing the number of rows of data from 88200 to 78051, which i'm not sure is what we want
    return data_


def shap_domain_merge2(data_, domain_df_):
    domain_df_.rename(columns={'Category': 'Domain Type'}, inplace=True)
    domain_df_['Domain Type'] = [x.title() for x in domain_df_['Domain Type']]  # # -- Angela: Added this after noticing some capitalization inconsistencies
    data_ = pd.merge(data_, domain_df_, on='Domain', how='left')
    return data_


def new_kw_cluster_merge(data_, cluster_map_):
    cluster_map_.rename(columns={'Cluster':'Old Cluster', 'New Cluster':'Cluster'}, inplace=True)
    cluster_map_['Cluster'] = [x.title() for x in cluster_map_['Cluster']]
    data_ = data_.drop('Cluster', axis=1)
    data_ = pd.merge(data_, cluster_map_, on='Keyword', how='left')
    data_['Cluster ID'] = data_['Cluster ID'].fillna(999)
    data_['Old Cluster'] = data_['Old Cluster'].fillna('999')
    data_['Cluster'] = data_['Cluster'].fillna('999')
    return data_


FILES_LOCATION = 'D://DataCenter/Professional/Arcalea/Repos/CANN_webapp/data/'

cluster_map = pd.read_csv(FILES_LOCATION + 'Revised Keyword Clusters_12-15-21_AC - Ver2.csv')[['Keyword','Cluster ID', 'Cluster', 'New Cluster']]
sov_df = pd.read_parquet(FILES_LOCATION + 'CANNSOV_input_for_SRF_new_12-13-21_2.parquet')
kwd_dist_df = pd.read_csv(FILES_LOCATION + 'cann_keyword_distribution.csv')
shap_data_input = load_shap_data2(FILES_LOCATION + 'full_set_shap_xgb_12-14-21.csv')  # Replace file p
# shap_data_input
print(shap_data_input.shape)
data = sov_shap_merge2(shap_data_input, sov_df)
print(data.shape)
data = new_kw_cluster_merge(data, cluster_map)
print(data.shape)
domain_df = pd.read_csv(FILES_LOCATION + 'Cannabis_SRF_Domains - Cannabis_SRF_Domains_New.csv')[['Domain', 'Category']]
data = shap_domain_merge2(data, domain_df)
print(data.shape)
include_columns = [x for x in data.columns if '_Count' not in x]
data = data[include_columns]
print(data.shape)
data.drop(columns=['Text', 'Meta Description'], inplace=True)
print(data.shape)
data.to_parquet(FILES_LOCATION + 'data.parquet')
print(data.shape)

kwd_dist_file = pd.ExcelFile(FILES_LOCATION + 'Cannabis Industry Keyword Distribution' + '.xlsx')

companies = ['Harvest', 'Curaleaf', 'Cresco', 'Green Thumb Industries', 'Trulieve', 'MedMen', 'The Botanist', 'RISE',
             'Acreage Holdings', 'Ayr Wellness, Inc.']


def process_keyword_distribution_data():
    kwd_dist_list = []

    for company in companies:
        kwd_df_temp = pd.read_excel(kwd_dist_file, company)
        kwd_df_temp['Company'] = company
        kwd_dist_list.append(kwd_df_temp)

    kwd_dist_data_df = pd.concat(kwd_dist_list, ignore_index=True)
    kwd_dist_data_df = kwd_dist_data_df[kwd_dist_data_df['Position'] <= 100]
    kwd_dist_data_df.drop(columns='URL', inplace=True)

    kwd_dist_data_df['variable'] = pd.cut(kwd_dist_data_df['Position'],
                                     bins=[1, 4, 11, 101], labels=['#1-3', '#4-10', '#11-100'],
                                     right=False)
    kwd_dist_data_df.sort_values(by='Position', ascending=True, inplace=True)

    kwd_dist_data_df = pd.pivot_table(kwd_dist_data_df, index=['Company', 'Branded or Non-Branded?', 'variable'],
                                      values=['Keyword'], aggfunc=pd.Series.nunique).reset_index()
    kwd_dist_data_df.columns = ['Company', 'Branded or Non-Branded?', 'variable', 'group_total']
    kwd_dist_data_df['Branded or Non-Branded?'] = kwd_dist_data_df['Branded or Non-Branded?'].str.lower()
    kwd_dist_data_df = kwd_dist_data_df[kwd_dist_data_df['group_total'] > 0]

    branded_non_branded = pd.pivot_table(kwd_dist_data_df, index=['Company', 'variable'],
                                         columns='Branded or Non-Branded?', values=['group_total'], aggfunc=np.sum)
    branded_non_branded.reset_index(inplace=True)
    branded_non_branded.columns = ['Company', 'variable', 'branded', 'non-branded']

    kwd_dist_data = pd.pivot_table(kwd_dist_data_df, index=['Company', 'variable'],
                                   values=['group_total'], aggfunc=np.sum)
    kwd_dist_data.reset_index(inplace=True)
    kwd_dist_data.columns = ['Company', 'variable', 'group_total']

    kwd_dist_data_df = pd.merge(kwd_dist_data, branded_non_branded, on=['Company', 'variable'],
                                how='left')
    kwd_dist_data_df['Branded %'] = kwd_dist_data_df['branded'] / kwd_dist_data_df['group_total']
    kwd_dist_data_df['Non-Branded %'] = kwd_dist_data_df['non-branded'] / kwd_dist_data_df['group_total']
    kwd_dist_data_df.drop(columns=['branded', 'non-branded'], inplace=True)

    kwd_dist_data_df['variable_order'] = np.where(kwd_dist_data_df['variable'] == '#1-3', 0,
                                                  np.where(kwd_dist_data_df['variable'] == '#4-10', 1, 2))

    kwd_dist_data_df.to_parquet(FILES_LOCATION + 'keyword_distribution.parquet')

    return kwd_dist_data_df


kwd_dist = process_keyword_distribution_data()
