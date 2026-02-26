import streamlit as st
import chemprop
from chemprop import data, featurizers, models, nn
import torch
from lightning import pytorch as pl
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from chem_prop_util import *
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
    
env:Env = None
if 'env' in st.session_state:
    env = st.session_state['env']
    
    
col1, col2 = st.columns([1,2])

smiles = ''
df_input = None
with col1:
    use_saved_model = st.selectbox(f'Use previously saved {app_vars.study} model:', MODEL_OPTIONS)
    app_dir = ''
    checkpoint_dir = ''
    if use_saved_model == MY_MODEL:
        app_dir = torch_file_paths.save_smiles_user
        checkpoint_dir = torch_file_paths.save_checkpoints_user
    elif use_saved_model == MASTER_MODEL:
        app_dir = torch_file_paths.save_smiles_dir
        checkpoint_dir = torch_file_paths.save_checkpoints
    
    app_file = os.path.join(app_dir, APP_FILE)
    if os.path.isfile(app_file):
        with open(app_file, 'r', encoding='utf-8') as f:
            app_vars = json.load(f)
            app_vars =AppVars(**app_vars)
    else:
        st.write(f"Selected model does not exist. Check out {MASTER_MODEL if use_saved_model == MY_MODEL else MASTER_MODEL} ")
        st.stop()


    paras_file = os.path.join(checkpoint_dir, PARAS_FILE)
    if os.path.isfile(paras_file):
        with open(paras_file, 'rb') as f:
            model_paras = pickle.load(f) 
    else:
        st.write(f"Selected model does not exist.")
        st.stop()

    
    model_files = os.listdir(checkpoint_dir)
    model_files = [ f for f in model_files if re.match(MODEL_FILE_PATTERN, f)]
    best_model = min(model_files, key=lambda x:  get_loss_val(x))
    mpnn = models.MPNN.load_from_file(os.path.join(checkpoint_dir, best_model))
    
    
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
    
    # print(app_vars.TARGET, app_vars.TARGET_ORI, app_vars.expt_col_name, '================')
    
    expt_label = app_vars.expt_col_name
    pred_label = f'pred_{app_vars.expt_label}'
    
    if cmpd_list:
        if exp_val_list:
            list_of_tuples = list(zip(cmpd_list, smiles_list, exp_val_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', expt_label, pred_label])
        else:
            list_of_tuples = list(zip(cmpd_list, smiles_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['Compound_ID', 'SMILES', pred_label])
    else:
        if exp_val_list:
            list_of_tuples = list(zip(smiles_list, exp_val_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', expt_label, pred_label])
        else:
            list_of_tuples = list(zip(smiles_list, preds))
            df_pred = pd.DataFrame(list_of_tuples, columns=['SMILES', pred_label])
            

    st.dataframe(df_pred)
    
    if  expt_label in df_pred.columns:
        st.write('***')
        y_expt = df_pred[expt_label]
        y_pred = df_pred[pred_label]
            
        r2_pred = r2_score(y_expt, y_pred)
        rmse_pred = root_mean_squared_error(y_expt, y_pred)
        
        st.write(f'R2: {round(r2_pred, 2)}; RSME: {round(rmse_pred, 2)}')
        
        fig, ax = plt.subplots()
        sns.regplot(data=df_pred, x=expt_label, y=pred_label, ax=ax)
        exp_min = df_pred[expt_label].min()
        exp_max = df_pred[expt_label].max()
        
        pred_min = df_pred[pred_label].min()
        pred_max = df_pred[pred_label].max()
        
        ax_min = min(exp_min, pred_min)
        ax_max = max(exp_max, pred_max)
        ax_len = ax_max-ax_min
        ax_min -= ax_len*0.05
        ax_max += ax_len*0.05
        
         
        plt.xlim(ax_min, ax_max)
        plt.ylim(ax_min, ax_max)
        # ax.set_aspect('equal', adjustable='box')
        st.pyplot(fig)

