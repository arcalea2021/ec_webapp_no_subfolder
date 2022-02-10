mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"angela@arcalea.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
primaryColor='#A34F9A'\n\
backgroundColor='#FFFFFF'\n\
secondaryBackgroundColor='#EDF2F9'\n\
font = 'AvenirBold'\n\
textColor = '#4A4A4A'\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT
" > ~/.streamlit/config.toml