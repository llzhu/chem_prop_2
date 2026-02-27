import streamlit as st
import pandas as pd
import numpy as np
from rdkit.Chem.Descriptors import ExactMolWt
import os
import math
import plotly.figure_factory as ff
from chem_prop_util import *
from chem_prop_comp import *
from chembl_webresource_client.new_client import new_client
import torch
import json
import math
from icecream import ic


st.set_page_config(page_title='What this page is about - ', layout='wide')

env = Env(  st.secrets['src_data'],
            st.secrets['app_data'],
            st.secrets['admins'],       
            st.secrets['modelers']
        ) 

app_header()

login_name, study, apply_log, excluded_list, new_model, overriddeen_container = app_setup()


checkpoints_user = os.path.join(env.app_data, rf'{login_name}/{study}/checkpoints')
save_checkpoints_user = os.path.join(env.app_data, rf'{login_name}/{study}/save_checkpoints')
save_checkpoints = os.path.join(env.app_data, rf'{study}/save_checkpoints')  # master saving folder

input_smiles_user = os.path.join(env.app_data, rf'{login_name}/{study}/input_smiles')
save_smiles_user = os.path.join(env.app_data, rf'{login_name}/{study}/save_smiles')
save_smiles_dir = os.path.join(env.app_data, rf'{study}/input_smiles')

torch_file_paths = TorchFilePaths(  input_smiles_user,
                                        save_smiles_user,
                                        save_smiles_dir,
                                        checkpoints_user,
                                        save_checkpoints_user,
                                        save_checkpoints
                                )

app_vars = AppVars( study,                    
                    login_name in env.admins, 
                    login_name,
                    apply_log=apply_log
                )

st.session_state['new_model'] = new_model
st.session_state['app_vars'] = app_vars   # basic paa vars are also in session
st.session_state['env'] = env
st.session_state['torch_file_paths'] = torch_file_paths

if study == '--':
        st.error('You must select a dataset to create/upload a model. You must be an admin to manage models.' )
        st.stop()
       
if not new_model:
    st.write(f"An existing model for {study} will be used.")
    st.stop()


if not os.path.exists(checkpoints_user):
    os.makedirs(checkpoints_user)

if not os.path.exists(input_smiles_user):
    os.makedirs(input_smiles_user)
elif new_model:
    delete_contents(input_smiles_user)
   
df_g = None
bin_0 = [0.1]    # dfault value for bin size

if study == DELANEY:
    url = os.path.join(env.src_data, 'delaney.csv')
    df_g = pd.read_csv(url)
    df_g = df_g[['Compound ID', 'log_M', 'SMILES']]
    expt_col_name = 'log_Solubility_uM'
    orig_col_name = 'Solubility_uM'
    df_g = df_g.rename(columns={'log_M':expt_col_name})

    apply_log = True   # Special logic
    
    # overriddeen_container.warning('The above selection is overridden to False')
   
elif study == THROBIN_IC50:
    activity = new_client.activity
    data_ic50 = activity.filter(target_chembl_id="CHEMBL204").filter(standard_type='IC50').filter(standard_relation='=').only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units'])

    df_g = pd.DataFrame(data_ic50)
    orig_col_name = 'IC50'
    df_g = df_g.rename(columns={'standard_value': orig_col_name})
    st.dataframe(df_g)

    (df_g, expt_col_name)  = standarize(df_g, THROBIN_IC50, orig_col_name, apply_log)
    
    
    if not apply_log:
        bin_0 = 100
   
elif study == AD_HOC:
    uploaded_data_file = None
    df_upload = None
    uploaded_data_file = st.sidebar.file_uploader("Upload a Data CSV file.")
    st.sidebar.markdown("<small>A SMILES col is required.</small>", unsafe_allow_html=True)
    orig_col_name = expt_col_name = ''
    if uploaded_data_file:
        df_upload = pd.read_csv(uploaded_data_file)
        col_all = df_upload.columns
        col_all = col_all.insert(0, '--')
        if 'SMILES' not in col_all:
            st.stop()

        expt_col = st.sidebar.selectbox('Select Experimental Value Column:', options=col_all)  
        if  expt_col != '--':
           
            if apply_log:
                df_negative = df_upload[df_upload[expt_col]<=0]
                if len(df_negative) > 0:
                    apply_log = False
                    overriddeen_container.warning('Some experimemtal value are not positive. Logarithm cannot be applied')
                    st.stop()

                orig_col_name = expt_col
                expt_col_name = f'log_{expt_col}'
                df_upload[expt_col_name] = df_upload[expt_col].apply(lambda x: math.log10(x))
            else:
                expt_col_name = orig_col_name = expt_col
               
              
        id_col = st.sidebar.selectbox('Select Compund ID Column if available:', options=col_all)  
        if  id_col != '--':
            cmpd_list = df_upload[id_col].tolist()
        else:
            df_g = df_upload
         

if df_g is not None and expt_col_name:
    smiles_file_user = os.path.join(input_smiles_user, INPUT_SMILES_FILE)
    df_g.to_csv(smiles_file_user, index=False)

    csv_input = get_df_csv(df_g)
    st.sidebar.download_button("Download Smiles file", data=csv_input.getvalue(), file_name=f'smiles_{study}.csv', mime='text/csv')

    # save them to session state
    app_vars.expt_col_name = expt_col_name
    app_vars.orig_col_name = orig_col_name
    app_vars.apply_log = apply_log
    app_vars.dset_size = len(df_g)
   
    # Also persist it. Used only for Ad hoc
    app_file_name = os.path.join(torch_file_paths.input_smiles_user, APP_FILE)
    with open(app_file_name, 'w', encoding='utf-8') as f:
        json.dump(app_vars.__dict__, f, indent=4)


    y_0 = [df_g[expt_col_name]]
    t_0 = [expt_col_name]

    c_data, c_fig = st.columns(2)
    with c_data:
        st.write('Total rows = ', app_vars.dset_size)
        st.dataframe(df_g)

    with c_fig:
        fig_0 = ff.create_distplot(y_0, t_0, bin_size=bin_0)
        st.write(f'{expt_col_name} Distribution')
        st.plotly_chart(fig_0)


