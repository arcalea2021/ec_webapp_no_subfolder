# Load libraries
import shap
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import app_functions as func
import app_text as txt
import app_style as sty

warnings.filterwarnings("ignore")

shap.initjs()

# ##################### Page title #####################################################################################
# Set title and favicon
st.set_page_config(page_title=txt.client_industry_for_title + ': Machine Learning Factor Analysis',
                   page_icon='https://arcalea.com/wp-content/uploads/2019/02/Arc-favicon-150x150.png',
                   layout="wide")

# ##################### CSS Styling - and setting plt font family ######################################################
with st.container():
    # Hide rainbow bar
    st.markdown(sty.hide_decoration_bar_style, unsafe_allow_html=True)
    # Hide hamburger menu & footer
    st.markdown(sty.hide_streamlit_style, unsafe_allow_html=True)
    # General font (body)
    st.markdown(sty.body_font_family, unsafe_allow_html=True)
    # Set font family
    plt.rcParams["font.family"] = "AvenirBold"

# ##################### Reading Data ###################################################################################
with st.container():

    sov_branded_df = pd.read_csv(txt.FILES_LOCATION + txt.sov_branded_input_file_name)
    sov_heating_df = pd.read_csv(txt.FILES_LOCATION + txt.sov_heating_input_file_name)
    sov_laundry_df = pd.read_csv(txt.FILES_LOCATION + txt.sov_laundry_input_file_name)
    sov_wastewater_df = pd.read_csv(txt.FILES_LOCATION + txt.sov_wastewater_input_file_name)

    sov_df = pd.concat([sov_branded_df, sov_heating_df, sov_laundry_df, sov_wastewater_df], ignore_index=True)

    keyword_distribution = pd.read_csv(txt.FILES_LOCATION + txt.kwd_dist_file_name)

    data = pd.read_parquet(txt.FILES_LOCATION + txt.data_file_name)
    type_info = pd.read_csv(txt.FILES_LOCATION + txt.domain_file_name)
    kw_rd_data = pd.read_csv(txt.FILES_LOCATION + txt.kw_rd_data_file_name)
    rd_rd_data = pd.read_csv(txt.FILES_LOCATION + txt.rd_rd_data_file_name)

# ####################### HEADINGS / SUBTITLES #########################################################################
# Headings/introduction copy
with st.container():
    st.info(txt.info_message)
    st.markdown(
        '<h1 style=" ' + sty.style_string + ' ">' + 'Cannabis Industry' + ': <br> Machine Learning Factor Analysis </h1>',
        unsafe_allow_html=True)
    st.markdown('<h4 style=" ' + sty.style_string + ' "> Overview </h4>',
                unsafe_allow_html=True)
    st.markdown(txt.header_paragraph1, unsafe_allow_html=True)
    st.markdown(txt.header_paragraph2, unsafe_allow_html=True)

# ######################################################## NAVIGATION MENU #############################################
with st.sidebar:
    st.markdown(" <h3 style=" + sty.style_string + "> <b> Table of Contents </b></h3>", unsafe_allow_html=True)
    st.markdown(txt.content_table_paragraph1, unsafe_allow_html=True)
    st.markdown(txt.content_table_overview, unsafe_allow_html=True)
    st.markdown(txt.content_table_sov, unsafe_allow_html=True)
    st.markdown(txt.content_table_kwd_distribution, unsafe_allow_html=True)
    st.markdown(txt.content_table_kwd_clusters, unsafe_allow_html=True)
    st.markdown(txt.content_table_srf_overall, unsafe_allow_html=True)
    st.markdown(txt.content_table_srf_clusters, unsafe_allow_html=True)
    st.markdown(txt.content_table_rd_backlinks, unsafe_allow_html=True)
    st.markdown(txt.content_table_return_to_top, unsafe_allow_html=True)

# ########################################################## SHARE OF VOICE ############################################
st.markdown('<h4 style=' + sty.style_string + '> Share of Voice </h4>', unsafe_allow_html=True)
with st.expander("Click to Expand/Collapse"):
    st.markdown(txt.sov_paragraph1, unsafe_allow_html=True)

    st.markdown('<p style=' + sty.style_string + '> <b> Branded Share of Voice (SOV) </b> </p>', unsafe_allow_html=True)

    branded_sov = func.particular_sov_get_srf_barchart(sov_branded_df)
    st.plotly_chart(branded_sov, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<p style=' + sty.style_string + '> <b> Heating Share of Voice (SOV) </b> </p>', unsafe_allow_html=True)
    heating_sov = func.particular_sov_get_srf_barchart(sov_heating_df)
    st.plotly_chart(heating_sov, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<p style=' + sty.style_string + '> <b> Laundry Share of Voice (SOV) </b> </p>', unsafe_allow_html=True)
    laundry_sov = func.particular_sov_get_srf_barchart(sov_laundry_df)
    st.plotly_chart(laundry_sov, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<p style=' + sty.style_string + '> <b> Wastewater Share of Voice (SOV) </b> </p>', unsafe_allow_html=True)
    wastewater_sov = func.particular_sov_get_srf_barchart(sov_wastewater_df)
    st.plotly_chart(wastewater_sov, use_container_width=True, config=sty.plotly_config_dict)

# ######################################################## KEYWORD DISTRIBUTION  #######################################
st.markdown('<h4 style=' + sty.style_string + '> Keyword Distribution </h4>', unsafe_allow_html=True)

with st.expander("Click to Expand/Collapse"):

    st.dataframe(keyword_distribution)

    # READING KWD Distribution Data - START
    keyword_distribution['company_total'] = keyword_distribution.groupby(['Company'])['group_total'].transform('sum')
    keyword_distribution = keyword_distribution.sort_values(by=['company_total', 'variable order'])
    kw_dist_plot_ft = func.keyword_distribution_barchart(keyword_distribution, txt.client_industry)
    # READING KWD Distribution Data - END

    st.markdown(txt.kwd_dist_paragraph1, unsafe_allow_html=True)
    st.markdown(txt.kwd_dist_paragraph2, unsafe_allow_html=True)
    st.plotly_chart(kw_dist_plot_ft, use_container_width=True, config=sty.plotly_config_dict)

# ########################################################## KEYWORD CLUSTERING #######################################
st.markdown('<h4 style=' + sty.style_string + '> Keyword Clustering </h4>', unsafe_allow_html=True)
with st.expander("Click to Expand/Collapse"):
    with st.container():  # text
        st.markdown(txt.kwd_clustering_paragraph1, unsafe_allow_html=True)
        st.markdown(txt.kwd_clustering_paragraph2, unsafe_allow_html=True)
        st.markdown(txt.kwd_clustering_paragraph3, unsafe_allow_html=True)
        st.markdown(txt.kwd_clustering_paragraph4, unsafe_allow_html=True)

    with st.container():
        data_no999 = data[-(data['Cluster ID'] == 999)]  # Remove "local" 999 cluster from data for this section
        cluster_label_list = list(data_no999['Cluster'].unique())

        # cluster_list.sort()
        selected_cluster_shap = st.selectbox("Select a Keyword Cluster", cluster_label_list, key='cluster_box_shap_kwc',
                                             index=0)  # Add a dropdown element
        # selected_cluster_shap = int(selected_cluster_shap)
        df_selected_cluster = data_no999.copy()
        df_selected_cluster = df_selected_cluster[df_selected_cluster['Cluster'] == selected_cluster_shap]  # Filter by selected cluster

        # Display keywords in the cluster
        st.markdown('<p style=' + sty.style_string + '> <b> Keywords in this Cluster (Desc. by Search Volume) </b> </p>', unsafe_allow_html=True)

        volume_keywords_in_this_cluster = df_selected_cluster.groupby(by='Keyword')['Volume'].max().sort_values(ascending=False).dropna()
        volume_keywords_in_this_cluster = pd.DataFrame(volume_keywords_in_this_cluster)
        st.write(volume_keywords_in_this_cluster.style.format("{0:,.0f}"))

        cluster_msv_value = sum(df_selected_cluster.groupby(by='Keyword')['Volume'].max().sort_values(ascending=False).dropna())
        st.markdown(
            '<p style=' + sty.style_string + '>' + 'The monthly search volume (MSV) for this cluster is <b>' + "{0:,.0f}".format(cluster_msv_value) + '</b>.' + '</p>',
            unsafe_allow_html=True)

        domain_type_list = list(df_selected_cluster['Domain Type'].unique())
        domain_type_list = sorted(domain_type_list)

        # Plot ratios using pie charts
        st.markdown(txt.kwd_clustering_ratios_paragraph1, unsafe_allow_html=True)
        st.markdown(txt.kwd_clustering_ratios_paragraph2, unsafe_allow_html=True)

        both_ratios = func.calculate_both_ratios(df_selected_cluster, domain_type_list)

        ratios_fig = func.get_ratios_chart(both_ratios)
        st.plotly_chart(ratios_fig, use_container_width=True, config=sty.plotly_config_dict)
        
        st.table(data=type_info)

        # Calculate SOV by Domain
        selected_cluster_sov_fig = func.get_sov_barchart(df_selected_cluster)

        st.markdown('<p style=' + sty.style_string + '> <b> Share of Voice in this Cluster </b> </p>',
                    unsafe_allow_html=True)
        st.plotly_chart(selected_cluster_sov_fig, use_container_width=True, config=sty.plotly_config_dict)

# ############################################ RANKING FACTORS IMPORTANCE - OVERALL ####################################
st.markdown('<h4 style=' + sty.style_string + '> Ranking Factor Importance (Overall) </h4>', unsafe_allow_html=True)
with st.expander("Click to Expand/Collapse"):
    st.markdown(txt.srf_overall_paragraph1, unsafe_allow_html=True)
    st.markdown(txt.srf_overall_paragraph2, unsafe_allow_html=True)
    st.markdown(txt.srf_overall_paragraph3, unsafe_allow_html=True)

    top_10_shap = func.get_top_n_shap(data, n=5)

    fig_shap_global = func.get_srf_barchart_fig(top_10_shap, 'Ranking Factor Importance (Absolute Contribution) Top 5')

    st.plotly_chart(fig_shap_global, use_container_width=True, config=sty.plotly_config_dict)

    # Summary Plot
    st.markdown('<h5 style=' + sty.style_string + '> Summary Plot </h5>', unsafe_allow_html=True)
    st.markdown(txt.srf_overall_paragraph4, unsafe_allow_html=True)

    # Define features and SHAP values - select only top 10
    top_10_X = top_10_shap.index
    top_10_Shap_X = top_10_shap.index + [' SHAP']
    top_10_Shap_X = top_10_X.append(top_10_Shap_X)

    fig_summ_plot_overall, summ_plot_overall_x_min, summ_plot_overall_x_max = func.get_summary_plot(data, top_10_Shap_X)

    st.pyplot(fig_summ_plot_overall)
    st.markdown(txt.srf_overall_summary_plot_paragraph1a +
                str(summ_plot_overall_x_min) + ' and +' + str(summ_plot_overall_x_max) +
                txt.srf_overall_summary_plot_paragraph1b, unsafe_allow_html=True)

    # Dependence plot
    st.markdown('<h5 style=' + sty.style_string + '> Dependence Plot </h5>',
                unsafe_allow_html=True)
    st.markdown(txt.srf_overall_dependence_plot_paragraph1, unsafe_allow_html=True)

    feature_x = st.selectbox("Select a Ranking Factor", list(top_10_shap.index), key='ranking_factor_box_overall',
                             index=0)  # Add a dropdown element
    feature_x_shap = feature_x + " SHAP"
    hover_data = list(top_10_shap.index)

    all_domain_type_list = list(data['Domain Type'].unique())

    color_dict = {domain_type_list[i]: sty.arc_colors[i] for i in range(len(domain_type_list))}

    fig_dependence_overall = func.get_dependence_plot(data, feature_x, feature_x_shap, color_dict, hover_data)

    st.plotly_chart(fig_dependence_overall, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown(txt.recommendations[feature_x], unsafe_allow_html=True)

# ############################################ RANKING FACTORS IMPORTANCE - SPECIFIC CLUSTERS ##########################

st.markdown('<h4 style=' + sty.style_string + '> Ranking Factor Importance (Specific Clusters) </h4>',
            unsafe_allow_html=True)

with st.expander("Click to Expand/Collapse"):
    st.markdown(txt.srf_specific_clusters_paragraph1, unsafe_allow_html=True)
    st.markdown(txt.srf_specific_clusters_paragraph2, unsafe_allow_html=True)
    st.markdown(txt.srf_specific_clusters_paragraph3, unsafe_allow_html=True)

    # Create a multi-select to select the cluster
    selected_cluster_shap_sc = st.selectbox("Select a Keyword Cluster", cluster_label_list, index=0)  # Add a dropdown element

    df_selected_cluster, df_selected_cluster_abs, top_10_shap_sc = func.get_df_sc(data_no999, selected_cluster_shap_sc)

    # Bar Chart
    st.markdown(txt.srf_specific_clusters_plot_paragraph1, unsafe_allow_html=True)

    fig_shap_selected_cluster = func.get_srf_barchart_fig(top_10_shap_sc,
                                                          'Ranking Factor Importance (Absolute Contribution) Top 5')

    st.plotly_chart(fig_shap_selected_cluster, use_container_width=True, config=sty.plotly_config_dict)

    # Summary Plot
    st.markdown('<h5 style=' + sty.style_string + '> Summary Plot for Selected Cluster</h5>', unsafe_allow_html=True)
    st.markdown(txt.srf_specific_clusters_summary_plot_paragraph1, unsafe_allow_html=True)

    # Define features and SHAP values -- select only top 10
    top_10_X_sc = top_10_shap_sc.index
    top_10_Shap_X_sc = top_10_shap_sc.index + [' SHAP']
    top_10_Shap_X_sc = top_10_X_sc.append(top_10_Shap_X_sc)

    # Summary plot
    data_sc = data_no999[data_no999['Cluster'] == selected_cluster_shap_sc]
    fig_summ_plot_sc, summ_plot_sc_x_min, summ_plot_sc_x_max = func.get_summary_plot(data_sc, top_10_Shap_X_sc)

    st.pyplot(fig_summ_plot_sc)

    st.markdown(txt.srf_specific_clusters_summary_plot_paragraph1a +
                str(summ_plot_sc_x_min) + ' and +' + str(summ_plot_sc_x_max) +
                txt.srf_specific_clusters_summary_plot_paragraph1b, unsafe_allow_html=True)

    # Dependence plot
    st.markdown('<h5 style=' + sty.style_string + '> Dependence Plot for Selected Cluster </h5>',
                unsafe_allow_html=True)
    st.markdown(txt.srf_specific_clusters_dependence_plot_paragraph1, unsafe_allow_html=True)

    df_dependence_sc = data[data.Cluster == selected_cluster_shap_sc]

    feature_x_sc = st.selectbox("Select a Ranking Factor", list(top_10_shap_sc.index), key='ranking_factor_box_sc',
                                index=0)  # Add a dropdown element
    feature_x_sc_shap = feature_x_sc + " SHAP"
    hover_data_sc = list(top_10_shap_sc.index)

    all_domain_type_list_sc = list(data['Domain Type'].unique())
    color_dict_sc = {all_domain_type_list_sc[i]: sty.arc_colors[i] for i in range(len(all_domain_type_list_sc))}

    fig_dependence_sc = func.get_dependence_plot(df_dependence_sc, feature_x_sc, feature_x_sc_shap, color_dict_sc,
                                                 hover_data_sc)

    st.plotly_chart(fig_dependence_sc, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown(txt.recommendations[feature_x_sc], unsafe_allow_html=True)

    # Get the range of variable values that yield the .99 percentile of SHAP Values
    range_max_shap = pd.DataFrame(
        {"Variable Value": df_dependence_sc[feature_x_sc], "SHAP Value": df_dependence_sc[feature_x_sc_shap]})
    range_max_shap.sort_values(by="SHAP Value", ascending=False, inplace=True)
    value_range = range_max_shap[range_max_shap['SHAP Value'] > range_max_shap['SHAP Value'].quantile(.90)]

# ############################################ REFERRING DOMAINS VS BACKLINKS ##########################################
st.markdown('<h4 style=' + sty.style_string + '> Referring Domains and Backlinks Analysis </h4>',
            unsafe_allow_html=True)

with st.expander("Click to Expand/Collapse"):

    st.markdown('<h5 style=' + sty.style_string + '> Backlinks vs Traffic </h5>', unsafe_allow_html=True)
    backlinks_fig, backlinks_result = func.get_regression_plot(rd_rd_data, 'Total Backlinks', 'Organic Traffic')
    st.plotly_chart(backlinks_fig, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<h5 style=' + sty.style_string + '> Referring Domains vs Traffic </h5>', unsafe_allow_html=True)
    rd_fig, rd_result = func.get_regression_plot(rd_rd_data, 'Total Referring Domains', 'Organic Traffic')
    st.plotly_chart(rd_fig, use_container_width=True, config=sty.plotly_config_dict)

# ############################################ REFERRING DOMAINS VS BACKLINKS ##########################################
st.markdown('<h4 style=' + sty.style_string + '> Keywords Analysis </h4>',
            unsafe_allow_html=True)

with st.expander("Click to Expand/Collapse"):

    st.markdown('<h5 style=' + sty.style_string + '> Organic Traffic vs Total Keywords </h5>', unsafe_allow_html=True)
    kwd_total_fig, kwd_total_result = func.get_regression_plot(kw_rd_data, 'Total Keywords', 'Organic Traffic')
    st.plotly_chart(kwd_total_fig, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<h5 style=' + sty.style_string + '> Organic Traffic vs 1-3 Keywords </h5>', unsafe_allow_html=True)
    kwd_1_3_fig, kwd_1_3_result = func.get_regression_plot(kw_rd_data, '#1-3', 'Organic Traffic')
    st.plotly_chart(kwd_1_3_fig, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<h5 style=' + sty.style_string + '> Organic Traffic vs 4-10 Keywords </h5>', unsafe_allow_html=True)
    kwd_4_10_fig, kwd_4_10_result = func.get_regression_plot(kw_rd_data, '#4-10', 'Organic Traffic')
    st.plotly_chart(kwd_4_10_fig, use_container_width=True, config=sty.plotly_config_dict)

    st.markdown('<h5 style=' + sty.style_string + '> Organic Traffic vs 11-100 Keywords </h5>', unsafe_allow_html=True)
    kwd_11_100_fig, kwd_11_100_result = func.get_regression_plot(kw_rd_data, '#11-100', 'Organic Traffic')
    st.plotly_chart(kwd_11_100_fig, use_container_width=True, config=sty.plotly_config_dict)


st.text("")
st.text("")

# Footnote
st.markdown(txt.questions_comments, unsafe_allow_html=True)
