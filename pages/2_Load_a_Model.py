import streamlit as st
import matplotlib.pyplot as plt
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

torch_file_paths: TorchFilePaths = None
if 'torch_file_paths' in st.session_state:
    torch_file_paths = st.session_state['torch_file_paths']

if not torch_file_paths:
    st.write('Go back to home page to start the applications.')
    st.stop() 

app_vars: AppVars = None
if 'app_vars' in st.session_state:
    app_vars = st.session_state['app_vars']
else:
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

app_file_dir = ''
if loaded_model == 'Master Model':
    checkpoint_dir = torch_file_paths.save_checkpoints
    app_file_dir = torch_file_paths.save_smiles_dir
    if data_set == 'Test':
        data_file = os.path.join(torch_file_paths.save_smiles_dir, TEST_SMILES_FILE)
    elif data_set == 'Validation':
        data_file = os.path.join(torch_file_paths.save_smiles_dir, VAL_SMILES_FILE)
    elif data_set == 'Train':
        data_file = os.path.join(torch_file_paths.save_smiles_dir, TRAIN_SMILES_FILE)
    else:
        st.stop()
    
elif loaded_model == 'My model':
    checkpoint_dir = torch_file_paths.save_checkpoints_user
    app_file_dir = torch_file_paths.save_smiles_user
    if data_set == 'Test':
        data_file = os.path.join(torch_file_paths.save_smiles_user, TEST_SMILES_FILE)  
    elif data_set == 'Validation':
        data_file = os.path.join(torch_file_paths.save_smiles_user, VAL_SMILES_FILE)
    elif data_set == 'Train':
        data_file = os.path.join(torch_file_paths.save_smiles_user, TRAIN_SMILES_FILE)
    else:
        st.stop()
     
else:
    st.stop()



app_file = os.path.join(app_file_dir, APP_FILE)
if os.path.isfile(app_file):
    with open(app_file, 'r', encoding='utf-8') as f:
        app_vars = json.load(f)
        app_vars =AppVars(**app_vars)
else:
    st.write(f"Selected model does not exist. Check out {MASTER_MODEL if loaded_model == MY_MODEL else MASTER_MODEL} ")
    st.stop()
    
ph = st.container()



if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir))==0:
    st.error('No saved model available.')
    st.stop()
    
model_files = os.listdir(checkpoint_dir)
model_files = [ f for f in model_files if re.match(MODEL_FILE_PATTERN, f)]
         
best_model = min(model_files, key=lambda x:  get_loss_val(x))
    
  
paras_file = os.path.join(checkpoint_dir, PARAS_FILE)
with open(paras_file, 'rb') as f:
    model_paras = pickle.load(f)
            
para_container.json(model_paras.__dict__, expanded=False)
        
   

df = pd.read_csv(data_file)
smis = df.loc[:, 'SMILES'].values
mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
# data_point = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
data_point = get_datapoint(mols, model_paras,ys=None, modify_model_paras=False)

# mpnn = models.MPNN.load_from_checkpoint(os.path.join(checkpoint_dir, best_model ))
mpnn = models.MPNN.load_from_file(os.path.join(checkpoint_dir, best_model ))

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
d_set = data.MoleculeDataset(data_point, featurizer=featurizer)
d_loader = data.build_dataloader(d_set, shuffle=False)


st.write('***')
c1, c2 = st.columns(2)
with c1:

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

    st.dataframe(df)

with c2:
    
    draw_exp_scale = False  
    if app_vars.apply_log:
        draw_exp_scale =  st.checkbox('Draw with Non-log scale')
    
      
    if draw_exp_scale:
        y_expt = df[expt_label_ori]
        y_pred = df[pred_label_ori]
        expt_label = expt_label_ori
        pred_label = pred_label_ori
        
    else:
        y_expt = df[expt_label]
        y_pred = df[pred_label]
        
        
    
       
            
    

    r2 = r2_score(y_expt, y_pred)
    rmse = root_mean_squared_error(y_expt, y_pred)
        

    st.write(f'R2: {round(r2, 2)}; RSME: {round(rmse, 2)}')

    fig, ax = plt.subplots()
    sns.regplot(data=df, x=expt_label, y=pred_label,  ax=ax)
    
    exp_min = df[expt_label].min()
    exp_max = df[expt_label].max()
        
    pred_min = df[pred_label].min()
    pred_max = df[pred_label].max()
        
    ax_min = min(exp_min, pred_min)
    ax_max = max(exp_max, pred_max)
    ax_len = ax_max-ax_min
    ax_min -= ax_len*0.05
    ax_max += ax_len*0.05
        
         
    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
       
    st.pyplot(fig)
    
    
    if copy2master:
        if os.path.exists(torch_file_paths.save_checkpoints):
            shutil.rmtree(torch_file_paths.save_checkpoints)
        shutil.copytree(torch_file_paths.save_checkpoints_user, torch_file_paths.save_checkpoints)
        
        if os.path.exists(torch_file_paths.save_smiles_dir):
            shutil.rmtree(torch_file_paths.save_smiles_dir)
        shutil.copytree(torch_file_paths.save_smiles_user, torch_file_paths.save_smiles_dir)
        
        ph.write("Your model has been copied to master model")