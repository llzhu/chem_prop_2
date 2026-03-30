import streamlit as st
import chemprop
from chemprop import data, featurizers, models, nn
import torch
from lightning import pytorch as pl
from sklearn.metrics import root_mean_squared_error, r2_score
import plotly.express as px
import seaborn as sns
from chem_prop_util import *
from chem_prop_comp import *
import json
import pickle
import os

torch_file_paths: TorchFilePaths = None
if 'torch_file_paths' in st.session_state:
    torch_file_paths = st.session_state['torch_file_paths']

app_vars: AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']
else:
    st.write(f"Go back to home page to start the applications.")
    st.stop()
    
with st.sidebar:
    mol_container = st.container()


col1, col2 = st.columns([1,2])

smiles = ''
df_input = None
with col1:
    mpnn, model_paras = get_model_and_paras(torch_file_paths, app_vars)
    
    SMI_LIST = 'SMILES lists'
    FILE_UPLOAD = 'File Upload'
    mol_input = st.radio('Mol input:', [SMI_LIST, FILE_UPLOAD], horizontal=True)
    
    smiles_list = []
    cmpd_list = []
    exp_val_list = []
    preds = []
    if mol_input == SMI_LIST:
        mols_in = st.text_area('SMILES List (separate by , or newline):', key='mols_in')
        if mols_in:
            smiles_list = get_list(mols_in)
    
    else:
        logarithmic_scale = st.checkbox('Convert to Logarithm for experimental value')
        uploaded_smiles_file = st.file_uploader("Upload a SMILES CSV file. A SMILES column is required. Expt val are optional for comparison")
        if uploaded_smiles_file:
            df_input = pd.read_csv(uploaded_smiles_file)
            col_all = df_input.columns
            col_all = col_all.insert(0, '--')
            
            smile_col = st.selectbox('Select required Smile Column:', options=col_all)
            if smile_col != '--':
                smiles_list = df_input[smile_col].tolist()
                    
            id_col = st.selectbox('Select Compund ID Column if available:', options=col_all)  
            if  id_col != '--':
                cmpd_list = df_input[id_col].tolist() 

            exp_col = st.selectbox('Select Experiment val Column if available:', options=col_all)  
            if  exp_col != '--':
                exp_val_list = df_input[exp_col].tolist()   


    df_pred = None
    if smiles_list:
        mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smiles_list]
        data_point = get_datapoint(mols, model_paras,ys=None, modify_model_paras=False)
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        d_set = data.MoleculeDataset(data_point, featurizer=featurizer)
        d_loader = data.build_dataloader(d_set, shuffle=False)


        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator="auto",
                devices=1
            )
            preds = trainer.predict(mpnn, d_loader)

        preds = np.concatenate(preds, axis=0)
        
        preds = [ p.item() for p in preds]
    else:
        # w/o SMILES list, nothing can be done
        st.stop()  
        
    
with col2:

    if exp_val_list:
        expt_label = exp_col
        pred_label = f'pred_{expt_label}'
        if cmpd_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, exp_val_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', expt_label, pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, exp_val_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', expt_label, pred_label])
    else:
        expt_label=''
        pred_label = f'pred_{app_vars.expt_col_name}'
        if cmpd_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', pred_label])
            

    row_id = df_pred.index.to_numpy()
    df_pred.insert(loc=0, column='row_id', value=row_id)

    if expt_label:
        highlight_only = st.checkbox('Only display selected mol in the correlation fig')    
    df_container = st.container()
    
    if expt_label and expt_label in df_pred.columns:
        fig_df_structure(df_pred, expt_label, pred_label, df_container, mol_container, highlight_only=highlight_only)
    else:
        st.dataframe(df_pred, hide_index=True)
