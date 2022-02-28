import numpy as np
import pandas as pd

import app_text as txt


def impute_cps(cps, cluster_avg_cps, avg_cps):
    if ~np.isnan(cps):
        return cps
    elif ~np.isnan(cluster_avg_cps):
        return cluster_avg_cps
    else:
        return avg_cps


# Building ctr curve
ctr_curve = [0.233, 0.205, 0.133, .087, .063, .047, .038, .031, .027, .023, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
est_relevant_traffic = pd.DataFrame({"CTR": ctr_curve, "Position": np.arange(1, len(ctr_curve) + 1)})

# Reading full set shap (for the Clusters)
shap_data_input = pd.read_csv(txt.FILES_LOCATION + txt.client_full_set_shap_file_name)
shap_clusters = shap_data_input[['Keyword', 'Original Cluster', 'Cluster']].drop_duplicates()

# Reading SOV input file
sov_columns = ['URL', 'keyword', 'Position', 'Volume', 'CPS', 'Clicks']
sov = pd.read_csv(txt.FILES_LOCATION + txt.sov_wastewater_for_preprocessing_file_name)[sov_columns]
sov = sov.rename(columns={'keyword': 'Keyword'})
print(" - sov dataset contains {} records".format(sov.shape[0]))
dedup_keywords = pd.read_csv(txt.FILES_LOCATION + txt.sov_grouped_grouping_keywords_file_name)
dedup_keywords.columns = ['grouping_keyword', 'Keyword']
dedup_keywords['Keyword'] = [x.strip() for x in dedup_keywords['Keyword']]
sov = pd.merge(sov, dedup_keywords, on='Keyword', how='left')
sov['grouping_keyword'] = sov['grouping_keyword'].fillna(0)
sov['grouping_keyword'] = np.where(sov['grouping_keyword'] == 0, sov['Keyword'], sov['grouping_keyword'])
sov.drop(columns='Keyword', inplace=True)
sov = sov.rename(columns={'grouping_keyword': 'Keyword', 'URL': 'Final URL'})
sov = pd.merge(sov, shap_clusters, on='Keyword', how='left')
print(" - sov dataset contains {} records".format(sov.shape[0]))
sov['cluster_avg_cps'] = sov['CPS'].groupby(sov['Original Cluster']).transform('mean')  # Using Original cluster for CPS imputation
sov['avg_cps'] = sov['CPS'].mean()
sov['CPS (Imputed)'] = sov.apply(lambda x: impute_cps(x['CPS'], x['cluster_avg_cps'], x['avg_cps']), axis=1)
sov['Clicks'] = sov['Volume'] * sov['CPS (Imputed)']
sov = pd.merge(sov, est_relevant_traffic, on='Position', how='left')
sov['Est. Mo Traffic'] = sov['Clicks'] * sov['CTR']

sov = pd.pivot_table(sov, index=['Final URL', 'Keyword'], values=['Est. Mo Traffic', 'Volume', 'Clicks'],
                     aggfunc='sum').reset_index()
print(" - sov records after grouper keywords were implemented are {}".format(sov.shape[0]))

print(" - shap_data_input records before sov merge are {}".format(shap_data_input.shape[0]))
shap_data_input = pd.merge(shap_data_input, sov, on=['Final URL', 'Keyword'], how='left')
print(" - shap_data_input records after sov merge are {}".format(shap_data_input.shape[0]))

include_columns = [x for x in shap_data_input.columns if '_Count' not in x]
shap_data_input = shap_data_input[include_columns]
# shap_data_input.to_csv(txt.FILES_LOCATION + 'data.csv', index=False)
shap_data_input.to_parquet(txt.FILES_LOCATION + 'data.parquet')
