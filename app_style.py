# Hide rainbow bar
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''

# Hide hamburger menu & footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

# Font family
body_font_family = """
  <style>
    body {
    font-family: Avenir,Helvetica Neue,sans-serif;
    }
  </style>"""

# CSS Font
style_string = "font-family:Avenir,Helvetica Neue,sans-serif;"

# Configuration for plotly charts, this removes some options for the plotly charts (reducing visual clutter)
plotly_config_dict = {
    'modeBarButtonsToRemove': ['pan2d', 'pan', 'lasso2d', 'select', 'toggleSpikelines', 'hoverCompareCartesian',
                               'hoverClosestCartesian', 'resetViews']}

# Colors for ratio charts (pie charts) for domains
ratio_charts_color = ['rgb(0, 25, 51)', 'rgb(0, 51, 102)', 'rgb(44, 93, 135)', 'rgb(88, 128, 162)',
                      'rgb(131, 163, 190)', 'rgb(175, 198, 217)', 'rgb(185, 223, 249)', 'rgb(185, 202, 250)',
                      'rgb(212, 205, 255)',
                      'rgb(199, 173, 239)', 'rgb(176, 132, 195)', 'rgb(163, 79, 154)', 'rgb(142, 39, 132)',
                      'rgb(115, 19, 67)', 'rgb(118, 55, 18)', 'rgb(168, 90, 31)', 'rgb(213, 149, 37)',
                      'rgb(240, 176, 65)']

arc_colors = ["#df9567", "#87a926", "#3f74c0", "#9e6cb7", "#6f92c3", "#481362", "#415f8b", "#92bb20", "#873eab",
              "#b7c8e1"]

arc_colors2 = ['#A34F9A', '#0D7BBD', '#56AEE3', '#2C4784', '#EDF2F9', '#4A4A4A', '#1C2D54']

figure_height = 7.5
figure_width = 10
