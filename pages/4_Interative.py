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
from StreamJSME import StreamJSME

torch_file_paths: TorchFilePaths = None
if 'torch_file_paths' in st.session_state:
    torch_file_paths = st.session_state['torch_file_paths']

app_vars: AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']
else:
    st.write(f"Go back to home page to start the applications.")
    st.stop()


mpnn, model_paras = get_model_and_paras(torch_file_paths, app_vars)
    
c1, c2 = st.columns(2)
with c1:
    st.write(f"Predicated {app_vars.expt_col_name} Value:")
with c2:
    pred_container = st.container()

input_smiles = st.text_input('Input SMILES:', value='')
update_smiles = StreamJSME(smiles=input_smiles, height=500, width=800)
mol = utils.make_mol(update_smiles, keep_h=False, add_h=False)
data_point = get_datapoint([mol], model_paras,ys=None, modify_model_paras=False)
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
    
        
    

pred_container.write(preds[0])
    
