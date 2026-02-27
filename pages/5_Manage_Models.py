import streamlit as st
from streamlit_file_browser import st_file_browser
import os
from chem_prop_util import *

app_vars: AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']
else:
    st.write(f"Go back to home page to start the applications.")
    st.stop()

if not app_vars.is_admin:
    st.write(f"You do not have the privilege to manually manage model files.")
    st.stop()
    
event = st_file_browser(
    os.path.join('.', "app_data"),
    file_ignores=('.models'),
    show_delete_file=True,
    key="A"
)

st.write(event)
