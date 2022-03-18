import shap
import time

import matplotlib.colorbar
import matplotlib.colors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots import make_subplots
from urllib.parse import urlparse

import app_style as sty

FILES_LOCATION = 'data/'
MAPBOX_TOKEN = 'pk.eyJ1IjoiYWxlLWFyY2FsZWEiLCJhIjoiY2t3Y2FzYmliMDhzdjJ0dDNibThlbml6YSJ9.asBwUAQjxszirN3xsUD8lA'


# Function configuration


def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return 'Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec)


# Load Avenir font
def get_font():
    # url = 'https://github.com/Papillard/Avenir/blob/master/AvenirBold.ttf'
    fm.fontManager.ttflist += fm.createFontList(['AvenirBold.ttf'])
    return plt.rc('font', family='AvenirBold')


# @st.cache()  # We use this to cache the info and not load the data every time we scroll up/down
def load_sov_data(filepath):
    sov_data_ = pd.read_csv(FILES_LOCATION + filepath)
    sov_data_.drop_duplicates(inplace=True)
    return sov_data_


#@st.cache
def load_shap_data(filepath):
    shap_data_ = pd.read_csv(filepath)
    ctr_curve_ = [0.233, 0.205, 0.133, .087, .063, .047, .038, .031, .027, .023, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0]

    available_positions = len(shap_data_['Position Organic'].unique())
    ctr_curve_ = ctr_curve_[:available_positions]

    est_relevant_tr_ = pd.DataFrame({"CTR": ctr_curve_, "Position": range(1, available_positions + 1)})

    shap_data_ = pd.merge(shap_data_, est_relevant_tr_, how='left', left_on=['Position Organic'], right_on=['Position'])
    shap_data_.drop(['Position'], axis=1, inplace=True)
    return shap_data_


def load_shap_data2(filepath):
    shap_data_ = pd.read_csv(filepath)
    ctr_curve_ = [0.233, 0.205, 0.133, .087, .063, .047, .038, .031, .027, .023, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0]

    available_positions = len(shap_data_['Position Organic'].unique())
    ctr_curve_ = ctr_curve_[:available_positions]

    est_relevant_tr_ = pd.DataFrame({"CTR": ctr_curve_, "Position": range(1, available_positions + 1)})

    shap_data_ = pd.merge(shap_data_, est_relevant_tr_, how='left', left_on=['Position Organic'], right_on=['Position'])
    shap_data_.drop(['Position'], axis=1, inplace=True)
    return shap_data_


@st.cache
def sov_shap_merge(shap_data_goes_here, msv_df_goes_here):
    msv_df_goes_here.rename(columns={'grouping_keyword': 'Keyword'}, inplace=True)
    data_ = pd.merge(shap_data_goes_here, msv_df_goes_here[['Keyword', 'Volume', 'Est. Relevant Traffic']],
                     on=['Keyword'])
    data_.drop(['CTR'], axis=1, inplace=True)
    return data_


def sov_shap_merge2(shap_data_goes_here, msv_df_goes_here):
    msv_df_goes_here.rename(columns={'grouping_keyword': 'Keyword'}, inplace=True)
    data_ = pd.merge(shap_data_goes_here, msv_df_goes_here[['Keyword', 'Volume', 'Est. Relevant Traffic']],
                     on=['Keyword'])
    data_.drop(['CTR'], axis=1, inplace=True)
    return data_


@st.cache
def shap_domain_merge(data_, domain_df_):
    domain_df_.rename(columns={'Category': 'Domain Type'}, inplace=True)
    data_ = pd.merge(data_, domain_df_, on='Domain', how='left')
    return data_


def shap_domain_merge2(data_, domain_df_):
    domain_df_.rename(columns={'Category': 'Domain Type'}, inplace=True)
    data_ = pd.merge(data_, domain_df_, on='Domain', how='left')
    return data_


def particular_sov_get_srf_barchart(df_, top_n=10):
    """Function is used to create a barchart with the sov for the dataframe

    Parameters:
        df_ (dataframe): dataframe for which the SOV will be calculated
        top_n (int): number of domains to be included in the chart

    Returns:
        fig_sov_: plotly barchart
    """
    df_ = df_.groupby(['Domain']).agg({'SOV': 'sum'})
    df_ = df_.sort_values(by='SOV', ascending=False)
    df_.reset_index(inplace=True)
    df_ = df_.iloc[:top_n, :]

    # Bar Chart
    title = "Share of Voice"
    fig_sov_ = px.bar(df_, x="Domain", y="SOV", text="SOV", title=title, width=1000*0.85, height=750*0.85)
    fig_sov_.update_traces(texttemplate='%{text:%}', marker_color='#1C2D54')
    fig_sov_.update_xaxes(title='Domain')
    fig_sov_.update_layout(font_family='Avenir,Helvetica Neue,sans-serif',
                           title_font_family='Avenir,Helvetica Neue,sans-serif')

    return fig_sov_


def color(text, client_name=None):
    # color: hexadecimal

    color_ = sty.arc_colors2[0] if text == client_name else sty.arc_colors2[5]

    s = "<span style='color:" + str(color_) + "'>" + str(text) + "</span>"

    s = '<b>' + s + '</b>' if text == client_name else s

    return s


def keyword_distribution_barchart(kw_df, title_, client_name):
    fig_ = px.bar(kw_df, x="Company", y="group_total", color="KW Group",
                  custom_data=['Branded %', 'Non-Branded %', 'KW Group'],
                  # title="Keyword Distribution - " + title_,
                  color_discrete_sequence=['#D7E6F2', '#9EC1DE', '#324680'], height=500, width=800)

    for data_ in fig_.data:
        data_['width'] = 0.50

    fig_.update_traces(hovertemplate="%{x}" +
                                     "<br>%{customdata[2]} keyword count %{y:,.0f}" +
                                     "<br>Branded: %{customdata[0]:.2%}" +
                                     "<br>Contextual: %{customdata[1]:.2%}")

    fig_.layout.font.family = 'Avenir,Helvetica Neue,sans-serif'
    fig_.update_layout(yaxis_title='Company', font_family='Avenir,Helvetica Neue,sans-serif',
                       title_font_family='Avenir,Helvetica Neue,sans-serif', legend_title_text=' ')
    fig_.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_.update_layout(margin=dict(l=20, r=20, t=100, b=100))

    # Adding color and bold to client's name in the visual
    x_axis_labels = kw_df['Company'].unique()
    tick_text = [color(k, client_name) for k in x_axis_labels]

    fig_.update_layout(
        xaxis=dict(tickmode='array', ticktext=tick_text, tickvals=x_axis_labels),
        title=title_
    )

    return fig_


def get_sov_barchart(selected_cluster_df, top_n=10):
    if 'Domain' not in selected_cluster_df:
        selected_cluster_df['Domain'] = selected_cluster_df['Final URL'].apply(lambda x: urlparse(x).netloc)

    if 'Est. Mo Traffic' in selected_cluster_df:
        selected_cluster_df = pd.DataFrame(
            {"Est. Mo Traffic": selected_cluster_df.groupby('Domain')['Est. Mo Traffic'].sum()})
        selected_cluster_df = selected_cluster_df.sort_values(by='Est. Mo Traffic', ascending=False)

        selected_cluster_df['SOV'] = selected_cluster_df['Est. Mo Traffic'] / selected_cluster_df['Est. Mo Traffic'].sum()
        selected_cluster_df = selected_cluster_df.sort_values(by='SOV', ascending=False)
        selected_cluster_df = selected_cluster_df.head(top_n)

    # elif 'Domain' in selected_cluster_df.columns:
    #    selected_cluster_df = pd.DataFrame({"SOV": selected_cluster_df.groupby('Domain')['Est. Relevant Traffic'].sum()})
    #    selected_cluster_df = selected_cluster_df.sort_values(by='SOV', ascending=False)
    #    selected_cluster_df = selected_cluster_df / selected_cluster_df[selected_cluster_df.columns].sum()
    #    selected_cluster_df = selected_cluster_df.head(top_n)

    fig_sov_ = px.bar(selected_cluster_df, x=selected_cluster_df.index, y="SOV",
                      height=400, text="SOV",
                      title="")
    fig_sov_.layout.font.family = 'Avenir'
    fig_sov_.update_traces(texttemplate='%{text:%}')
    fig_sov_.update_traces(marker_color='#1C2D54')

    if 'Domain' in selected_cluster_df.columns:
        fig_sov_.update_xaxes(title='Domain')

    fig_sov_.update_layout(font_family='Avenir,Helvetica Neue,sans-serif',
                           title_font_family='Avenir,Helvetica Neue,sans-serif')

    return fig_sov_


def calculate_both_ratios(df_selected_cluster, domain_type_list):
    ratio_values_dict = {}
    traffic_values_dict = {}
    for i in range(len(domain_type_list)):
        ratio_values_dict[i] = sum(df_selected_cluster['Domain Type'] == domain_type_list[i]) / len(
            df_selected_cluster)
        traffic_values_dict[i] = df_selected_cluster[df_selected_cluster['Domain Type'] == domain_type_list[i]][
                                     'Est. Relevant Traffic'].sum() / df_selected_cluster[
                                     'Est. Relevant Traffic'].sum()

    ratio_values_df = pd.DataFrame(ratio_values_dict.items())
    ratio_values_df.columns = ['Domain Type', 'Result Ratio']
    ratio_values_df['Domain Type'] = domain_type_list
    ratio_values_df.sort_values(by='Result Ratio', ascending=False, inplace=True)
    ratio_values_df = ratio_values_df.reset_index(drop=True)
    ratio_values_df['R_Rank'] = ratio_values_df.index + 1

    traffic_values_df = pd.DataFrame(traffic_values_dict.items())
    traffic_values_df.columns = ['Domain Type', 'Traffic Ratio']
    traffic_values_df['Domain Type'] = domain_type_list
    traffic_values_df.sort_values(by='Traffic Ratio', ascending=False, inplace=True)
    traffic_values_df = traffic_values_df.reset_index(drop=True)
    traffic_values_df['T_Rank'] = traffic_values_df.index + 1

    both_ratios = traffic_values_df.merge(ratio_values_df, on='Domain Type', how='left')
    both_ratios['Agg_Rank'] = (both_ratios['T_Rank'] + both_ratios['R_Rank']) / 2
    both_ratios = both_ratios.drop(['T_Rank', 'R_Rank'], axis=1)
    both_ratios.sort_values(by='Agg_Rank', ascending=True, inplace=True)
    both_ratios['Agg_Rank'] = both_ratios.reset_index(drop=True).index + 1

    return both_ratios


def get_ratios_chart(both_ratios):
    ratios_fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    ratios_fig.add_trace(
        go.Pie(labels=both_ratios['Domain Type'], values=both_ratios['Result Ratio'], name="Result Ratio",
               marker_colors=sty.ratio_charts_color, textfont=dict(color="#FFFFFF")), 1, 1)
    ratios_fig.add_trace(
        go.Pie(labels=both_ratios['Domain Type'], values=both_ratios['Traffic Ratio'], name="Traffic Ratio",
               marker_colors=sty.ratio_charts_color, textfont=dict(color="#FFFFFF")), 1, 2)
    ratios_fig.update_traces(hole=.5, hoverinfo="label+value")
    ratios_fig.layout.font.family = 'Avenir'
    ratios_fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)))
    ratios_fig.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)))
    ratios_fig.update_layout(annotations=[dict(text='Result Ratio', x=0.19, y=0.5, font_size=13, showarrow=False),
                                          dict(text='Traffic Ratio', x=0.81, y=0.5, font_size=13, showarrow=False)])
    ratios_fig.update_layout(font_family='Avenir,Helvetica Neue,sans-serif',
                             title_font_family='Avenir,Helvetica Neue,sans-serif')

    return ratios_fig


def get_top_n_shap(df_, n=10):
    # Get top 10 SHAP Values
    shap_cols = [col for col in df_.columns if 'SHAP' in col]  # Find columns with 'SHAP' in them
    df_abs_all = df_[shap_cols].abs()  # Get absolute value of SHAP Values
    top_n_shap = (df_abs_all[shap_cols].mean()).T.sort_values(ascending=False)[
                 :n]  # Select the top 5 most impactful variables
    top_n_shap = pd.DataFrame(top_n_shap)
    top_n_shap.index = [ind.replace(' SHAP', '') for ind in top_n_shap.index]  # Remove the word ' SHAP' from the index
    top_n_shap.columns = ["Impact on Page-One Probability"]

    return top_n_shap


def get_srf_barchart_fig(top_n_shap_df, title_):
    top_n_shap_df.columns = ['Impact on Page-One Probability']
    fig_ = px.bar(top_n_shap_df, x=top_n_shap_df.index, y='Impact on Page-One Probability', color=top_n_shap_df.index,
                  text='Impact on Page-One Probability',
                  title=title_)
    fig_.layout.font.family = 'Avenir'
    fig_.update_traces(texttemplate='%{text:%}')
    fig_.update_traces(marker_color='#1c2d54')
    fig_.update_traces(showlegend=False)
    fig_.update_xaxes(title='Ranking Factors')

    fig_.update_layout(font_family='Avenir,Helvetica Neue,sans-serif',
                       title_font_family='Avenir,Helvetica Neue,sans-serif')

    return fig_


def get_summary_plot(df, top_n_shap_x, n=5):
    df_summary_plot = df.copy()
    df_summary_plot = df_summary_plot[top_n_shap_x]
    x_summary_plot = df_summary_plot.iloc[:, :n]
    shap_values_summary_plot = df_summary_plot.iloc[:, n:]
    shap_values_summary_plot = shap_values_summary_plot.to_numpy()

    # Plot the summary without showing it
    summ_fig_pyplot = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values_summary_plot, x_summary_plot, show=False, color_bar=False, plot_type='dot',
                      max_display=10)

    # Change the colormap of the artists -- define top and bottom colormaps for custom gradient
    top = cm.get_cmap('Blues_r', 128)  # r means reversed version
    bottom = cm.get_cmap('RdPu', 128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                           bottom(np.linspace(0, 1, 128))))
    my_cmap = plt.get_cmap(ListedColormap(newcolors, name='arctest'))

    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(my_cmap)

    # Change X-axis label
    ax = plt.gca()
    ax.set_xlabel('Impact on Page-One Probability', fontsize=12)

    # Create new feature-value gradient
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="2%", pad=0.1)
    summ_fig_pyplot.add_axes(ax_cb)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=my_cmap, orientation='vertical')
    ticklabels = ['Low', 'High']
    cb1.set_ticks(np.linspace(0, 1, len(ticklabels)))
    cb1.set_ticklabels(ticklabels)
    cb1.set_label('Feature Value')

    x_min = str('{:.1f}'.format(min([min(shap_values_summary_plot[:, 0]), min(shap_values_summary_plot[:, 1]),
                                     min(shap_values_summary_plot[:, 2]), min(shap_values_summary_plot[:, 3]),
                                     min(shap_values_summary_plot[:, 4])]) * 100))
    x_max = str('{:.1f}'.format(max([max(shap_values_summary_plot[:, 0]), max(shap_values_summary_plot[:, 1]),
                                     max(shap_values_summary_plot[:, 2]), max(shap_values_summary_plot[:, 3]),
                                     max(shap_values_summary_plot[:, 4])]) * 100))

    return summ_fig_pyplot, x_min, x_max


def get_dependence_plot(df_, feature_x, feature_x_shap, hover_data, color_dict=None):

    df_dependence = df_.copy()

    if color_dict is None:
        fig_dependence = px.scatter(df_dependence, x=feature_x, y=feature_x_shap, opacity=0.5,
                                    hover_data=hover_data, range_x=[-1, df_dependence[feature_x].quantile(.90) + 1])
        fig_dependence.update_traces(marker=dict(color=sty.arc_colors2[6]))
        fig_dependence.update_traces(showlegend=False)
    else:
        fig_dependence = px.scatter(df_dependence, x=feature_x, y=feature_x_shap, color="Domain Type",
                                    color_discrete_map=color_dict, opacity=0.5,
                                    hover_data=hover_data, range_x=[-1, df_dependence[feature_x].quantile(.90) + 1])
        fig_dependence.update_traces(showlegend=True)

    fig_dependence.layout.font.family = 'Avenir'
    fig_dependence.update_yaxes(title='Impact on Page-One Probability')
    fig_dependence.update_layout(font_family='Avenir,Helvetica Neue,sans-serif',
                                 title_font_family='Avenir,Helvetica Neue,sans-serif')

    return fig_dependence


def get_regression_plot(df_, x_, y_, x_axis_title=None):

    fig = px.scatter(df_, x=x_, y=y_, trendline="ols", trendline_color_override=sty.arc_colors2[0])

    fig.update_traces(marker_symbol=3, marker_color=sty.arc_colors2[6])
    fig.update_layout(font_family='Avenir,Helvetica Neue,sans-serif',
                      title_font_family='Avenir,Helvetica Neue,sans-serif',
                      width=150 * 0.85, height=500 * 0.85)

    results = px.get_trendline_results(fig)

    fig.update_layout(
        xaxis_title=x_axis_title
    )

    # fig.write_image(title_ + ".svg", width=10 * 300, height=7.5 * 300, scale=1, engine='kaleido')

    return fig, results


def get_df_sc(data_no999, selected_cluster_shap_sc):
    df_selected_cluster = data_no999.copy()

    df_selected_cluster = df_selected_cluster[df_selected_cluster['Cluster'] == selected_cluster_shap_sc]  # Filter by selected cluster

    # Get top 10 SHAP Values for the specific cluster
    shap_cols_sc = [col for col in df_selected_cluster.columns if 'SHAP' in col]  # Find columns with 'SHAP' in them
    df_selected_cluster_abs = df_selected_cluster[shap_cols_sc].abs()  # Get absolute value of SHAP Values
    df_selected_cluster_abs['Cluster'] = \
    df_selected_cluster[df_selected_cluster['Cluster'] == selected_cluster_shap_sc]['Cluster']
    df_selected_cluster_abs['Cluster'] = df_selected_cluster_abs['Cluster'].astype('str')  # Add the cluster to use as a filter

    shap_cols_sc.append('Cluster')  # Add the cluster to use as a filter

    top_10_shap_sc = (df_selected_cluster_abs[shap_cols_sc].groupby('Cluster').mean().loc[
        str(selected_cluster_shap_sc)]).T.sort_values(ascending=False)[:5]
    top_10_shap_sc = pd.DataFrame(top_10_shap_sc)
    top_10_shap_sc.index = [ind.replace(' SHAP', '') for ind in top_10_shap_sc.index]  # Remove the word ' SHAP' from the index
    top_10_shap_sc.columns = [selected_cluster_shap_sc]

    return df_selected_cluster, df_selected_cluster_abs, top_10_shap_sc


def get_cluster_names(df, exceptions_cluster_number=None, exceptions_cluster_names=None):
    cluster_names_df = df[['Cluster', 'Volume', 'Keyword']]
    cluster_names_df = cluster_names_df.fillna(0).drop_duplicates()
    cluster_names_df = cluster_names_df.sort_values(by=['Cluster', 'Volume'], ascending=[True, False])
    cluster_names_df = cluster_names_df.groupby('Cluster').head(1)
    cluster_names_df = cluster_names_df[cluster_names_df['Volume'] > 0]
    cluster_names_df = cluster_names_df[['Cluster', 'Keyword']]

    # Specific names for top clusters
    if exceptions_cluster_number is not None:
        for i in range(0, len(exceptions_cluster_number)):
            number_ = exceptions_cluster_number[i]
            name_ = exceptions_cluster_names[i]
            cluster_names_df.loc[(cluster_names_df['Cluster'] == number_, 'Keyword')] = name_

    return cluster_names_df
