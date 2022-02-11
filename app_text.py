import app_style as sty

# Section to update client's information
# Define client brand and client brand industry - replace with different QMAs
client = 'Ellis Corporation'  # Client name here
client_industry = 'Industrial Water Treatment Solutions'  # Client industry here
client_industry_for_title = 'Industrial Water Treatment Solutions'

FILES_LOCATION = 'data/'

# Files from pre-processing have the objective of reducing the data stored in the server and help app run faster
sov_branded_input_file_name = 'EC_branded_sov_output.csv'  # this is a parquet file from pre-processing
sov_heating_input_file_name = 'EC_heating_sov_output.csv'
sov_laundry_input_file_name = 'EC_laundry_sov_output.csv'
sov_wastewater_input_file_name = 'EC_wastewater_sov_output.csv'

kwd_dist_file_name = 'EC_kw_dist_output.csv'
data_file_name = 'data.parquet'  # this is a parquet file from pre-processing
domain_file_name = 'Cannabis_SRF_Domains - Domain Type Breakdown.csv'

kw_rd_data_file_name = 'EC_wa_regression_analysis_keyword_output.csv'
rd_rd_data_file_name = 'EC_RD_Analysis-RegressionData.csv'

# Text for header
info_message = "The following analysis leverages real-time calculations on the current dataset. Please allow approximately 60 seconds for the interface to load. Once loaded, each section can be expanded using the (+) buttons; navigation between expanded sections can be achieved through the Table of Contents bar to the left."
header_paragraph1 = '<p style=' + sty.style_string + '>The following application is an interactive page that provides insights and recommendations intended to help a brand maximize a domain’s probability of ranking on Google’s first page within the ' + client_industry + ' industry.</p>'
header_paragraph2 = '<p style=' + sty.style_string + '> Select the following boxes to expand/collapse the different analysis sections. </p>'

# Text for Table of Contents
content_table_paragraph1 = " <p style=" + sty.style_string + "> Once all analysis sections have been expanded, feel free to use the Table of Contents below to navigate between sections. </br> </p>"
content_table_overview = " <p style=" + sty.style_string + "> <a href='#overview'>Overview </a> </p>"
content_table_sov = " <p style=" + sty.style_string + "> <a href='#share-of-voice'>Share of Voice </a> </p>"
content_table_kwd_distribution = " <p style=" + sty.style_string + "> <a href='#keyword-distribution'>Keyword Distribution </a> </p>"
content_table_kwd_clusters = " <p style=" + sty.style_string + "> <a href='#keyword-clustering'>Keyword Clustering </a> </p>"
content_table_srf_overall = " <p style=" + sty.style_string + "> <a href='#ranking-factor-importance-overall'>Ranking Factor Importance (Overall) </a> </p>"
content_table_srf_clusters = " <p style=" + sty.style_string + "> <a href='#ranking-factor-importance-specific-clusters'>Ranking Factor Importance (Specific Clusters) </a> </p>"
content_table_rd_backlinks = " <p style=" + sty.style_string + "> <a href='#referring-domains-and-backlinks-analysis'>Referring Domains and Backlinks Analysis </a> </p>"
content_table_return_to_top = " <p style=" + sty.style_string + "> <a href='#cannabis-industry-machine-learning-factor-analysis'>Return to Top </a> </p>"

# text for Share of Voice
sov_paragraph1 = '<p style=' + sty.style_string + '> The following bar chart approximates the Share of Voice <a href="https://arcalea.com/blog/share-of-voice-as-a-strategic-accelerator/" target="_blank"><b>(SOV)</b></a>, or the current relative visibility that industry players have for ' + client_industry + ' related searches in the United States. </p>'

# Text for Keyword Distribution
kwd_dist_paragraph1 = '<p style=' + sty.style_string + '>Organic keywords are one of the greatest drivers of online visibility and traffic. When a brand’s website features comprehensive, industry-relevant on-page content, the domain is able to gain relevance in Google’s algorithm. While the Search Ranking Factors analysis gives insight into what makes a particular webpage competitive for page-one, each brand’s Keyword Distribution correspond to the sheer opportunities a brand has to appear in results at all. Simply stated, the total number of organic keywords a brand has corresponds to the <b>number of opportunities</b> to rank. Thus, when strategic content production is paired with the Search Ranking Factors recommendations, such as authoritative link building, the brand is able to graduate the domain’s known keywords from positions 11-100 to 4-10, and subsequently from 4-10 to 1-3. This two-fold approach increases both the <b>number of ranking opportunities</b>, as well as the <b>potential for page one placement per opportunity</b>, ultimately allowing brands to command greater click-through-rate (CTR) on search result pages for given queries and increase organic visibility.</p>'
kwd_dist_paragraph2 = '<p style=' + sty.style_string + '>Below, the competitive keyword distribution for players within the ' + client_industry + ' industry. Due to screen size and visualization limitations, some x-axis labels may be minimized; please hover the cursor over the bars within each plot for further information on label names and the number of keywords in each ranking group (#1-3, #4-10, #11-100). The hover tooltip also details the percentage of branded vs. contextual keywords within that ranking group, which gives insight into the opportunity that is available for industry relevance ownership.</p>'

# Text for Keyword Clustering
kwd_clustering_paragraph1 = '<p style=' + sty.style_string + '>By analyzing the number of shared search results between queries, Arcalea’s intelligent clustering method partitions keywords in distinct topic clusters, which enables brands to better understand target topics for content creation.</p>'
kwd_clustering_paragraph2 = '<p style=' + sty.style_string + '>This analysis provides a comprehensive assessment of the search queries that are nested within an industry’s various topics. Through building extensive content hubs around ' + client_industry + ' industry topics using cluster keyword sets, ' + client + ' has the opportunity to increase its digital relevance and page-one-probability around various topics. Furthermore, by parsing out the ratios for different domain types, as well as outlining the SOV within each cluster, this analysis provides invaluable insights as to which topics competitors are ranking for, and which topics may be short-term targets for brand content creation. Finally, the aggregate search volume for each cluster provides information around the value a page within a topic cluster may have, which empowers ' + client + ' to fortify content where there is ample search opportunity.</p>'
kwd_clustering_paragraph3 = '<p style=' + sty.style_string + '>	Please use the drop-down bar to select specific clusters of interest within the ' + client_industry + ' industry. Once a cluster is selected, key pieces of information will populate below, including:</p>'
kwd_clustering_paragraph4 = '<ul style=' + sty.style_string + '> <li>Keywords in the selected cluster.</li> <li>Monthly search volume (MSV) of the selected cluster.</li> <li>Result and Traffic Ratios per domain type of the selected cluster.</li> <li>Bar chart SOV of the selected cluster.</li> </ul> <br> '

kwd_clustering_ratios_paragraph1 = '<p style=' + sty.style_string + '> <b> Result and Traffic Ratios in this Cluster</b> </br> The following pie charts break down the Result Ratio and Traffic Ratio per domain type in the selected cluster. Definitions for each ratio type are provided below, as well as a domain type overview table. </p>'
kwd_clustering_ratios_paragraph2 = '<ul style=' + sty.style_string + '> <li> <b> Result ratio </b> refers to the percentage of results belonging to a specific domain type across a cluster. Generally, for a specific keyword, there are 10 opportunities to rank, corresponding to the 10 URLs on a SERP. Thus, if there are five keywords in a cluster, then there would be 50 potential URL results. If two of the 50 URL opportunities are content aggregators, then the cluster result ratio for content aggregators would be 4%.</li>' + \
                                   '<li> <b> Traffic ratio </b> refers to the percentage of traffic belonging to a specific domain type across a cluster. Traffic ratio takes into account not only the number of results for a specific domain type, but also the keyword search volume and position of those results (recall that SERP click-through-rates decline significantly after the top three results) for that cluster. Following the previous example, if content aggregator domains hold the number one position for two high-volume keywords in a five keyword cluster (2 of the 50 URL opportunities), the <i>result ratio</i> would be 4%, However, the <i>traffic ratio</i> for content aggregators is expected to be much higher than 4%, given that the content aggregator domains rank in position one for two high-volume keywords. This example illustrates that a domain type may have a low result ratio, but a high traffic ratio for a cluster if URLs of that domain type are high ranking on the SERP, and/or if the keywords that domain type ranks for have high search volume. </li> </ul> <br> '

# Text for SRF Overall
srf_overall_paragraph1 = '<p style=' + sty.style_string + '>Machine Learning algorithms were leveraged to mimic Google’s search ranking system to determine the most impactful variables contributing to page-one probability within the overarching ' + client_industry + ' industry. </p>'
srf_overall_paragraph2 = '<p style=' + sty.style_string + '>While the specific details of the methodology may be complex, the output is fairly straightforward. In essence, the contribution of each of the variables, in this case, ranking factors, is calculated by measuring the <b>marginal</b> contribution of each with respect to the predicted outcome (probability of ranking on the 1st page); depending on the value of each ranking factor, this marginal contribution can either be positive, negative, or neutral. The marginal nature of each of these ranking factors must be emphasized, as it highlights the concept that page-one probability is the cumulative outcome of a host of variables.</p>'
srf_overall_paragraph3 = '<p style=' + sty.style_string + '>To visualize the absolute contribution of each variable, the following bar plot provides the top 5 variables (in descending order) that have an impact on page-one probability.</p>'
srf_overall_paragraph4 = '<p style=' + sty.style_string + '>The bar plot above shows the <b>absolute</b> contribution of each of the variables. However, some variable values may have a negative impact on page-one probability. To take a more nuanced view at how specific values of each variable impact page-one probability, the following summary plot can be referenced. The summary plot contains the same top variables in descending order, but provides additional insight into how specific values of the variables contribute to page-one probability.</p>'

srf_overall_summary_plot_paragraph1a = '<p style=' + sty.style_string + '>In the above plot, the X-axis represents the impact on page-one probability; data points to the left side of this axis have negative impact on page-one probability, while data points to the right side of this axis have positive impact on page-one probability. In this example, the range of X-axis values is between '
srf_overall_summary_plot_paragraph1b = ' percentage points. On the right side of the summary plot is a Feature Value gradient bar that indicates the value of a particular variable. The color blue represents a low value for that variable, while the color purple represents a high value for that variable. For example, the variable Domain Rating is on a scale from 0-100 so, a blue point would be closer to 0 and a purple point would be closer to 100. Together, the X-axis and the color scale provide information on the general relationship directionality between each variable’s values and page-one probability.</p>'

srf_overall_dependence_plot_paragraph1 = '<p style=' + sty.style_string + '>In order to understand the impact of each top-ranking factor on page-one probability, dependence plots of each of the variables can be analyzed. Dependence plots visualize a ranking factor on the x-axis against its contribution to page-one probability on the y-axis, revealing the ranges in which each factor maximizes page-one probability. Also overlaid on the dependence plots are the domain types for each URL observation, which may give further insight into how the combination of ranking factors and domain types influence page one probability (a general overview on domain types can be found in the ' + " <a href='#keyword-clustering'>Keyword Clustering</a>" + ' section). The following drop-down allows users to select a top-ranking factor for this industry for dependence plot generation. A brief description for that top ranking factor may be found below the dependence plots.</p>'

# Text for SRF Specific Clusters
srf_specific_clusters_paragraph1 = '<p style=' + sty.style_string + '>Machine Learning algorithms were leveraged to mimic Google’s search ranking system to determine the most impactful variables contributing to page-one probability within the overarching ' + client_industry + ' industry. </p>'
srf_specific_clusters_paragraph2 = '<p style=' + sty.style_string + '>While the specific details of the methodology may be complex, the output is fairly straightforward. In essence, the contribution of each of the variables, in this case, ranking factors, is calculated by measuring the <b>marginal</b> contribution of each with respect to the predicted outcome (probability of ranking on the 1st page); depending on the value of each ranking factor, this marginal contribution can either be positive, negative, or neutral. The marginal nature of each of these ranking factors must be emphasized, as it highlights the concept that page-one probability is the cumulative outcome of a host of variables.</p>'
srf_specific_clusters_paragraph3 = '<p style=' + sty.style_string + '><b>This section of the application provides perspective into ranking factors for specific topic clusters within the general query set.</b> For more information on Arcalea’s clustering methodology, please see the ' + " <a href='#keyword-clustering'>Keyword Clustering</a>" + ' section. The drop-down bar below allows the user to select a topic cluster of interest.</p>'

srf_specific_clusters_plot_paragraph1 = '<p style=' + sty.style_string + '>To visualize the absolute contribution of each variable for the selected cluster, the following bar plot provides the top 5 variables (in descending order) that have an impact on page-one probability for the selected cluster.</p>'

srf_specific_clusters_summary_plot_paragraph1 = '<p style=' + sty.style_string + '>The bar plot above shows the <b>absolute</b> contribution of each of the variables, however, some variables have a negative impact on page-one probability. To take a more nuanced view at how specific values of each variable impact page-one probability, the following summary plot can be referenced, which contains the same variables in descending order, but also provides insight into the specific values of the variables.</p>'

srf_specific_clusters_summary_plot_paragraph1a = '<p style=' + sty.style_string + '>In the above plot, the X-axis represents the impact on page-one probability; data points to the left side of this axis have negative impact on page-one probability, while data points to the right side of this axis have positive impact on page-one probability. In this example, the range of X-axis values is between '
srf_specific_clusters_summary_plot_paragraph1b = ' percentage points. On the right side of the summary plot is a Feature Value gradient bar that indicates the value of a particular variable. The color blue represents a low value for that variable, while the color purple represents a high value for that variable. For example, the variable Domain Rating is on a scale from 0-100 so, a blue point would be closer to 0 and a purple point would be closer to 100. Together, the X-axis and the color scale provide information on the general relationship directionality between each variable’s values and page-one probability.</p>'

srf_specific_clusters_dependence_plot_paragraph1 = '<p style=' + sty.style_string + '>In order to understand the impact of each top-ranking factor on page-one probability, dependence plots of each of the variables can be analyzed. Dependence plots visualize a ranking factor on the x-axis against its contribution to page-one probability on the y-axis, revealing the ranges in which each factor maximizes page-one probability. Also overlaid on the dependence plots are the domain types for each URL observation, which may give further insight into how the combination of ranking factors and domain types influence page one probability (a general overview on domain types can be found in the ' + " <a href='#keyword-clustering'>Keyword Clustering</a>" + ' section). The following drop-down allows users to select a top-ranking factor for this industry for dependence plot generation. A brief description for that top ranking factor may be found below the dependence plots.</p>'


# Footer text
questions_comments = " <p style=" + sty.style_string + ">If you have any questions or comments, please feel free to send us an email at team@arcalea.com.</p>"

# Organic recommendations and further variable explanation
recommendations = {
    'Title Length': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Title Length</b> measures the number of characters on the page’s title tag. Similar to Meta Length, this is a variable that can have an indirect impact on search rankings. While having a title tag length within a specific range might not directly affect rankings, having an optimized title that entices users to click from the SERP does have an effect on rankings. In order to avoid truncation, optimized titles are typically within a certain recommended range, but the main goal of page title experimentation is to improve CTR from the SERP. </p>""",
    'KW In Title': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>KW in Title</b> indicates whether the search query keyword is present within the page's Title. Having the target keyword present in a page's Title may indirectly influence search rankings, as it is a indicator of relevance for users browsing the SERP and can increase user click-through rate. Generally, however, keywords should only be included in the page title if it is related to the page's actual content; having multiple keywords in the title may not always be beneficial, as Google tends to truncate overly long page Titles on the SERP, and has historically punished 'keyword stuffing' behavior. </p>""",
    'KW in Meta': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>KW in Meta</b> indicates whether the search query keyword is present within the page's meta description. Having the target keyword present in a page's Meta Description may indirectly influence search rankings, as it is an indicator of relevance for users browsing the SERP and can increase user click-through rate. Generally, however, keywords should only be included in the page's Meta Description if it is related to the page's actual content; having multiple keywords in the meta description may not always be beneficial, as Google has historically punished 'keyword stuffing' behavior. </p>""",
    'Meta Length': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Meta Length</b> measures the number of characters on the page’s meta description. Similar to Title Length, this is a variable that can have an indirect impact on search rankings. While having a meta description length within a specific range might not directly affect rankings, having an optimized meta description to entice users to click from the SERP does have an effect on rankings. In order to avoid truncation, optimized titles are typically within a certain recommended range, but the main goal of page title experimentation is to improve CTR from the SERP. </p>""",
    'Size (bytes)': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Size (bytes)</b> measures the approximate size of the page’s HTML (reported in bytes). The size of a page plays an important role in how quickly the page is loaded, therefore, a page should be optimized to be as “lean” as possible without compromising the user experience and the quality of the content. Furthermore, page size shouldn’t be the only performance metric to be optimized, other user-centric metrics should also be considered, such as the “Core Web Vitals”, and the quality of the overall user experience. Developers may check the Size (bytes) of a page using tools such as <a href="https://www.rankwatch.com/free-tools/website-page-size-checker" target="_blank"> this.</a></p>""",
    'Word Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Word Count</b> measures the approximate number of words in the page’s content (limited to words inside the body tag). The HTML markup is excluded from the count. Word Count can be an indirect measure of content depth, but it is also important to consider that some queries require a shorter answer format, therefore the Word Count range recommendation should be directional, and should not in any way compromise the readability or quality of the content.</p>""",
    'Text Ratio': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Text Ratio</b> measures the ratio between user-readable text vs. HTML code. Higher Text Ratios may indicate high-quality/cleaner code, and for that reason, this metric can have a great impact on how quickly a page is loaded and also the user experience, therefore, it should be optimized in parallel with Size (bytes) and other user-centric metrics. Developers may check the Text Ratio of a page using tools such as <a href="https://smallseotools.com/code-to-text-ratio-checker/" target="_blank">this.</a></p>""",
    'Unique External Outlinks': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Unique External Outlinks</b> measures the number of unique hyperlinks (originating from the observation URL page) pointing to pages on other domains. External outlinks are often used by content writers to link references within page copy, and can help signal to search engine crawlers the relevance of a page's content to certain topics. While Unique External Outlinks may not be a direct search engine ranking factor, this metric holds implications for content rich-ness of pages, which can lead to increased user duration and better user experience, which are known ranking factors. </p>""",
    'Response Time': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Response Time</b> measures the time it takes to issue an HTTP request and get the full HTTP response back from the server. The figure displayed is in seconds. There are several ways to improve Response Time, including using reliable and fast web hosting, using a CDN, database optimization, keeping a lightweight theme (WordPress), configuring caching, and minifying scripts. More information on how to optimize this variable can be found <a href="https://developers.google.com/speed/docs/insights/Server" target="_blank">here.</a></p>""",
    'Modified 2021': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Modified 2021</b> indicates whether or not the web page has been updated or modified in 2021. Especially for more dynamic industries, having relevant and up-to-date content on page can be helpful for ranking on page one. Keeping web pages updated with fresh content may increase click-through-rate, a known search optimization factor, as users are more likely to click on more recent search results. </p>""",
    'H1 Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>H1 Count</b> measures the number of H1 tags identified on the page. This metric is an indirect indicator of the quality of the content's structure. Best practices suggest that only one H1 tag should be used for a page, therefore, increased use of H1 tags might indicate poor content structure. It is important to highlight the fact that the variable is an indirect indicator, and therefore not something that should be optimized for a specific number for the sake of including more/less H1 tags. </p>""",
    'H2 Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>H2 Count</b> measures the number of H2 tags identified on the page. This metric is an indirect indicator of both content depth and structure. A page that covers a topic in-depth is more likely to have more sub-sections, and therefore, if it has the correct semantic HTML structure, it will also have more H2 tags. It is important to highlight the fact that the variable is an indirect indicator, and therefore not something that should be optimized for a specific number for the sake of including more/less H2 tags. </p>""",
    'H3 Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>H3 Count</b> measures the number of H3 tags identified on the page. This metric is an indirect indicator of both content depth and structure. A page that covers a topic in-depth is more likely to have more sub-sections, and therefore, if it has the correct semantic HTML structure, it will also have more H3 tags. It is important to highlight the fact that the variable is an indirect indicator, and therefore not something that should be optimized for a specific number for the sake of including more/less H3 tags. </p>""",
    'H4 Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>H4 Count</b> measures the number of H4 tags identified on the page. This metric is an indirect indicator of both content depth and structure. A page that covers a topic in-depth is more likely to have more sub-sections, and therefore, if it has the correct semantic HTML structure, it will also have more H4 tags. It is important to highlight the fact that the variable is an indirect indicator, and therefore not something that should be optimized for a specific number for the sake of including more/less H4 tags. </p>""",
    'YouTube Links': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>YouTube Links</b> measures the number of YouTube links found on the web page. Linking out to YouTube videos may be a sign of rich content and can help further relevance around the content topic, both of which may improve user experience on page and page-one probability.</p>""",
    'Price Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>Price Count</b> measures the number of price mentions found on the web page. Depending on the type of industry and search query, the number of price mentions can be helpful to consider when understanding page content. </p>""",
    '$ Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>$ Count</b> measures the number of $ symbols found on the web page. Depending on the type of industry and search query, the presence of $ symbols on a page can be helpful to consider when understanding page content. </p>""",
    'Image Tag Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>Image Tag Count</b> measures the number image tags found on the web page. Image Tag Count may indirectly influence search rankings, as it can be a proxy measurement for content richness which improves user experience on page. However, overloading a page with images can also be detrimental to page-one ranking probability, as non-optimized, "heavy" images can increase page loading times, which can detract from the user experience on page and even increase user bounce probability. Therefore, it is important to consider the optimal thresholds for Image Tag Count. </p>""",
    'Largest Contentful Paint Time': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b> Largest Contentful Paint Time (LCP)</b> measures perceived load-speed as a user-centric metric. Largest Contentful Paint Time (LCP) is part of the “Core Web Vitals” metrics that Google has <a href="https://developers.google.com/search/blog/2020/11/timing-for-page-experience" target="_blank">confirmed</a> as ranking signals, therefore, it is important to consider this and other performance metrics to maximize the likelihood of ranking on Google’s first page. This is a technical variable, and more information on how to optimize it can be found <a href="http://web.dev/lcp/" target="_blank">here.</a></p>""",
    'First Input Delay Time': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>First Input Delay (FID)</b> indicates a user’s first impression of a site’s interactivity and responsiveness. It measures the time from when a user first interacts with a page to the time when the browser is actually able to respond to that interaction.The main cause of a poor FID is heavy JavaScript execution. Optimizing how JavaScript parses, compiles, and executes on a web page will directly reduce FID. First Input Delay (FID) is part of the “Core Web Vitals” metrics that Google has confirmed as ranking signals. More information on how to optimize for this metric can be found <a href="https://web.dev/optimize-fid/" target="_blank">here.</a></p>""",
    'Cumulative Layout Shift': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Cumulative Layout Shift (CLS)</b> measures the visual stability of a page. CLS is another user-centric metric and is part of the “Core Web Vitals” metrics that Google has confirmed as ranking signals. This is a technical variable, and more information on how to optimize it can be found <a href="https://web.dev/cls/" target="_blank">here.</a></p>""",
    'Image Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>Image Count</b> measures the number of images found on the web page. Image Count may indirectly influence search rankings, as it can be a proxy measurement for content richness which improves user experience on page. However, overloading a page with images can also be detrimental to ranking probability, as non-optimized, "heavy" images can increase page loading times, which can detract from the user experience on page and even increase user bounce probability. Therefore, it is important to consider the optimal thresholds for Image Count. The main difference between this metric and Image Tag Count is that this also considers images that might be embedded in an iframe; not both variables are used in all instances. </p>""",
    'Media Count': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>Media Count</b> measures the number of media pieces found on the web page. Media Count may indirectly influence search rankings, as it can be a proxy measurement for content richness which improves user experience on page. However, overloading a page with media can also be detrimental to ranking probability, as non-optimized, "heavy" media can increase page loading times, which can detract from the user experience on page and even increase user bounce probability. Therefore, it is important to consider the optimal thresholds for Media Count.</p>""",
    'Origin Core Web Vitals Assessment Passed': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>Origin Core Web Vitals Assessment Passed</b> indicates whether or not a page passes the Origin Core Web Vitals Assessment. This assessment considers a blend of technical page quality indicators that influence user experience, and compares the page against a known threshold. More information on this variable can be found <a href="https://web.dev/vitals/" target="_blank">here.</a></p>""",
    'RefDomains': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"><b>Referring Domains</b> measures the number of unique websites pointing to the observation URL. When managing campaigns to increase referring domains it is recommended to consider the quality of the domains and not only the quantity, mainly because high quality and relevant referring domains can be exponentially more valuable than lower quality and irrelevant domains. </p>""",
    'UR': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>UR, or URL Rating</b> measures the popularity and strength of a specific page's link profile. It is based on the number of referring pages (backlinks) pointing to the URL and the quality of each of those pages. To effectively increase UR, it is important to focus not only on acquiring more referring pages but to actually acquire referring pages that also have a high UR themselves, as using the strategy of quality over quantity should yield better results. Additionally, it is also important to consider the relevance of each referring page within the context of the website they point to. While this metric doesn’t account for relevance, Google’s search algorithms do. </p>""",
    'DR': """ <p style="font-family:Avenir,Helvetica Neue,sans-serif;"> <b>DR, or Domain Rating</b> measures the popularity and strength of a domain’s link profile. It is based on the number of referring domains pointing to the domain and the quality of each of those domains. To effectively increase DR, it is important to focus not only on acquiring more referring domains but to actually acquire referring domains that also have a high DR themselves, as using the strategy of quality over quantity should yield better results. Additionally, it is also important to consider the relevance of each referring domain within the context of the website they point to. While this metric doesn’t account for relevance, Google’s search algorithms do. </p>"""
}
