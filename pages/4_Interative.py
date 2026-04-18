import streamlit as st
from chemprop import data, featurizers
import torch
from lightning import pytorch as pl
from chem_prop_util import *
from chem_prop_comp import *
from StreamJSME import StreamJSME



env: Env = None
if 'env' in st.session_state:
    env = st.session_state['env']

app_vars: AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']

if not env or not app_vars:
    st.write(f"Go back to home page to start the applications.")
    st.stop()
    

mpnn, model_paras, app_vars = get_model_and_paras_from_s3(env, app_vars)
    
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
    
