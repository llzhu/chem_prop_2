import streamlit as st
import plotly.express as px
from chem_prop_util import *
import os
from sklearn.metrics import root_mean_squared_error, r2_score
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
import pickle
import json
import seaborn as sns
from chem_prop_util import *
from chem_prop_comp import *

env: Env = None
if 'env' in st.session_state:
    env = st.session_state['env']

app_vars: AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']

if not env or not app_vars:
    st.write(f"Go back to home page to start the applications.")
    st.stop()

col0, col1, col2, col3= st.columns([3,3,3, 3])
with col0:
    para_container = st.container()
    para_container.write(app_vars.study)
with col1:
    sel_models = ['--', 'My model', 'Master Model']
    loaded_model = st.selectbox(f'Load a saved model:', sel_models)
with col2:
    data_sets = ['--', 'Test', 'Validation', 'Train']
    data_set = st.selectbox(f'Select Dataset:', data_sets)
with col3:
    copy2master = False
    if app_vars.is_admin:
        st.write("")
        st.write("")
        copy2master = st.button('Copy my model to master', disabled=(not app_vars.is_admin) )


user_dir = os.path.join(env.app_data, app_vars.login_name, app_vars.study)
master_dir = os.path.join(env.app_data, app_vars.study)
base_dir = master_dir if loaded_model == 'Master Model' else user_dir

mpnn, model_paras, app_vars = get_model_paras_from_s3(env, app_vars, base_dir)
app_file_dir = os.path.join(base_dir, INPUT_FILES_DIR)  

if data_set == 'Test':
    test_file_s3key = os.path.join(app_file_dir, TEST_SMILES_FILE).replace("\\", "/")
    data_file = get_from_s3(env.s3_bucket, test_file_s3key)
    # data_file = os.path.join(app_file_dir, TEST_SMILES_FILE)
elif data_set == 'Validation':
    data_file = get_from_s3(env.s3_bucket, os.path.join(app_file_dir, VAL_SMILES_FILE).replace("\\", "/"))
    # data_file = os.path.join(app_file_dir, VAL_SMILES_FILE)
elif data_set == 'Train':
    data_file = get_from_s3(env.s3_bucket, os.path.join(app_file_dir, TRAIN_SMILES_FILE).replace("\\", "/"))
    # data_file = os.path.join(app_file_dir, TRAIN_SMILES_FILE)
else:
    st.stop()
    
# ph = st.container()
    
para_container.json(model_paras.__dict__, expanded=False)
        
   
df = pd.read_csv(BytesIO(data_file["Body"].read()))
# df = pd.read_csv(data_file)
smis = df.loc[:, 'SMILES'].values
mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
# data_point = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
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
d_preds = trainer.predict(mpnn, d_loader)
d_preds = np.concatenate(d_preds, axis=0)

expt_label = app_vars.expt_col_name
pred_label = f'pred_{expt_label}'
    
df[pred_label] = d_preds

if app_vars.apply_log:
    expt_label_ori = app_vars.orig_col_name
    pred_label_ori = f'pred_{expt_label_ori}'
        
    df[expt_label_ori] = df[expt_label].apply(lambda y: pow(10,y))
    df[pred_label_ori] = df[pred_label].apply(lambda y: pow(10,y))

row_id = df.index.to_numpy()
df.insert(loc=0, column='row_id', value=row_id)


with st.sidebar:
    mol_container = st.container()

st.write('***')
c1, c2 = st.columns(2)
with c1:
    highlight_only = st.checkbox('Only display selected mol in the correlation fig')
    df_container = st.container()
    

with c2:
    
    draw_exp_scale = False  
    if app_vars.apply_log:
        draw_exp_scale =  st.checkbox('Draw with Non-log scale')
    
      
    if draw_exp_scale:
        # y_expt = df[expt_label_ori]
        # y_pred = df[pred_label_ori]
        expt_label = expt_label_ori
        pred_label = pred_label_ori
        
   
    y_expt = df[expt_label]
    y_pred = df[pred_label]
        
    r2 = r2_score(y_expt, y_pred)
    rmse = root_mean_squared_error(y_expt, y_pred)
        

    st.write(f'R2: {round(r2, 2)}; RSME: {round(rmse, 2)}')

    
    fig_df_structure(df, expt_label, pred_label, df_container, mol_container, highlight_only=highlight_only)


    
    
    
    # if copy2master:
    #     if os.path.exists(torch_file_paths.save_checkpoints):
    #         shutil.rmtree(torch_file_paths.save_checkpoints)
    #     shutil.copytree(torch_file_paths.save_checkpoints_user, torch_file_paths.save_checkpoints)
        
    #     if os.path.exists(torch_file_paths.save_smiles_dir):
    #         shutil.rmtree(torch_file_paths.save_smiles_dir)
    #     shutil.copytree(torch_file_paths.save_smiles_user, torch_file_paths.save_smiles_dir)
        
    #     ph.write("Your model has been copied to master model")