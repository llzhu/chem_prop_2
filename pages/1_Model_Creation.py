import streamlit as st
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn
from chem_prop_util import *
from timeit import default_timer as timer
import os
from datetime import timedelta
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle
import json
import seaborn as sns
from datetime import date
from icecream import ic


start = timer()
    
new_model = False
if 'new_model' in st.session_state:
    new_model = st.session_state['new_model']

if not new_model:
    st.write('You are using an existing model. No new models will be created.')
    st.stop()

torch_file_paths: TorchFilePaths = None
if 'torch_file_paths' in st.session_state:
    torch_file_paths = st.session_state['torch_file_paths']

if not torch_file_paths:
    st.write('Go back to home page to start the applications.')
    st.stop()

# get complete app_vars
app_file = os.path.join(torch_file_paths.input_smiles_user, APP_FILE)
with open(app_file, 'r', encoding='utf-8') as f:
    app_vars = json.load(f)
    app_vars =AppVars(**app_vars)
   
col1, col2, col3, col4, col5, col6,  = st.columns(6)
with col1:
    st.write(app_vars.login_name)
    st.write(app_vars.study)
with col2:
    epochs = st.text_input('Epochs:', value = '50')
with col3:
    mol_graph_options = [BMP, AMP]
    mol_graph = st.selectbox(f'Mol Graph Passing?', mol_graph_options)
with col4:
    split_options = [SPLIT_RANDOM, SPLIT_SCAFFOLD_BALANCED]
    split_type = st.selectbox('Split Strategy:', split_options)
with col5:
    if app_vars.is_admin:
        save_options = ['--', 'No', 'To my folder', 'To Master']
    else:
        save_options = ['--', 'No', 'To my folder']
    save_model = st.selectbox('Save the model', save_options)
with col6:
    random_seed = st.text_input('Seed for spliting:', value='0')  

   


    
col11, col12, col13, col14, col15, col16 = st.columns(6)

with col11:
    message_depth = st.text_input('Message Depth:', value='2')
with col12:
    message_hidden_dim = st.text_input('Message Hidden Dim:', value='400')
with col13:
    ffn_num_layers = st.text_input('ffn num of layers:', value='2')
with col14:
    ffn_hidden_dim = st.text_input('ffn hidden dim:', value='2200')
with col15:
    add_fp = st.checkbox("\+ Morgan FP")
    add_dc = st.checkbox("\+ 2D Descriptors")
with col16:
    scale_dc = st.checkbox("Scale 2D Descriptors?", disabled=not add_dc, help='Use scaled 2D descriptors only when non-scaled fails.')


go = st.button('Create a new model!', disabled=((not epochs) or (save_options=='--') ) )

num_workers = 0 

hp = {'message_depth': int(message_depth),
      'message_hidden_dim': int(message_hidden_dim),
      'ffn_num_layers': int(ffn_num_layers),
      'ffn_hidden_dim': int(ffn_hidden_dim)
     }


if go:

    model_paras :ModelParas = ModelParas(mol_graph, int(epochs), split_type, app_vars.dset_size, add_fp, add_dc, scale_dc, model_created=str(date.today()))
    # st.session_state['model_paras'] = model_paras
    # ic(model_paras)
    
    delete_contents(torch_file_paths.checkpoints_user)
    
    df_input = pd.read_csv(os.path.join(torch_file_paths.input_smiles_user, INPUT_SMILES_FILE))

    # st.dataframe(df_input)

    smis = df_input.loc[:, 'SMILES'].values
    mols = [utils.make_mol(smi, keep_h=False, add_h=False) for smi in smis]
    ys = df_input.loc[:, [app_vars.expt_col_name]].values

    datapoints = get_datapoint(mols, model_paras, ys)

    # ic(model_paras)
    # ic(model_paras.dc_scaler)  
    extra_feature_size = model_paras.extra_feature_size
    
  
    train_indices, val_indices, test_indices = data.make_split_indices(mols, split_type, (0.8, 0.1, 0.1))
 
    train_data, val_data, test_data = data.split_data_by_indices(datapoints, train_indices, val_indices, test_indices)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)
    

    targets_scaler = train_dset.normalize_targets()
    extra_scaler = train_dset.normalize_inputs("X_d")


    val_dset.normalize_targets(targets_scaler)
    if extra_scaler is not None:
        val_dset.normalize_inputs("X_d", extra_scaler)
    

    # Featurize the train and val datasets to save computation time.
    train_dset.cache = True
    val_dset.cache = True

    train_loader = data.build_dataloader(train_dset)
    val_loader = data.build_dataloader(val_dset, shuffle=False)
    test_loader = data.build_dataloader(test_dset, shuffle=False)


    if mol_graph == BMP:
        mp = nn.BondMessagePassing(depth=hp['message_depth'], d_h=hp['message_hidden_dim'])
    elif mol_graph == AMP:
        mp = nn.AtomMessagePassing(depth=hp['message_depth'], d_h=hp['message_hidden_dim'])

    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(targets_scaler) 
    input_dim =  mp.output_dim + extra_feature_size
    
    ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=input_dim, n_layers=hp['ffn_num_layers'], hidden_dim=hp['ffn_hidden_dim'] )

    batch_norm = False  ####
    metric_list = [nn.metrics.RMSE(), nn.metrics.R2Score()]
    if extra_scaler is not None:
        X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_scaler)
        mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list, X_d_transform=X_d_transform)
    else:
        mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=torch_file_paths.checkpoints_user,  # Specify the path where you want to save checkpoints
        filename=f'{mol_graph}' + '_{epoch}_{val_loss:.3f}',  # Optionally, specify the filename format
        save_top_k=5,  # Save the top 5 checkpoints
        monitor='val_loss',  # Monitor the validation loss
        mode='min'  # Save checkpoints with the minimum validation loss
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        max_epochs=int(epochs),  # number of epochs to train for
    )

    trainer.fit(mpnn, train_loader, val_loader)


    results = trainer.test(mpnn, test_loader)

    st.subheader(f"Overall Prediction Accuracy for {app_vars.expt_col_name}")
    st.markdown((pd.DataFrame(results)).style.hide(axis="index").to_html(), unsafe_allow_html=True)
    end = timer()
    st.write(f'Elapsed time: {timedelta(seconds=end - start)}')
    

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
            test_preds = trainer.predict(mpnn, test_loader)

        df_test = df_input.iloc[test_indices[0]]

        df_test.to_csv(os.path.join(torch_file_paths.input_smiles_user, TEST_SMILES_FILE), index=False)
        (df_input.iloc[val_indices[0]]).to_csv(os.path.join(torch_file_paths.input_smiles_user, VAL_SMILES_FILE), index=False)
        (df_input.iloc[train_indices[0]]).to_csv(os.path.join(torch_file_paths.input_smiles_user, TRAIN_SMILES_FILE), index=False)
        
        

        test_preds = np.concatenate(test_preds, axis=0)
        df_test['pred'] = test_preds

        st.dataframe(df_test)

    with c2:
        y_test = df_test[app_vars.expt_col_name]
        y_pred = df_test['pred']

        r2_test = r2_score(y_test, y_pred)
        rmse_test = root_mean_squared_error(y_test, y_pred)
        

        st.write(f'R2: {round(r2_test, 2)}; RSME: {round(rmse_test, 2)}')

        
        fig, ax = plt.subplots()
        sns.regplot(data=df_test, x=app_vars.expt_col_name, y='pred', ax=ax)
        exp_min = df_test[app_vars.expt_col_name].min()
        exp_max = df_test[app_vars.expt_col_name].max()
        
        pred_min = df_test['pred'].min()
        pred_max = df_test['pred'].max()
        
        ax_min = min(exp_min, pred_min)
        ax_max = max(exp_max, pred_max)
        ax_len = ax_max-ax_min
        ax_min -= ax_len*0.05
        ax_max += ax_len*0.05
        
        plt.xlim(ax_min, ax_max)
        plt.ylim(ax_min, ax_max)
        ax.set_aspect('equal', adjustable='box')
        st.pyplot(fig)

    # persist model parameters to a pickle file
    para_file_name = os.path.join(torch_file_paths.checkpoints_user, PARAS_FILE)
    with open(para_file_name, 'wb') as f:
        pickle.dump(model_paras, f)

    app_file_name = os.path.join(torch_file_paths.input_smiles_user, APP_FILE)
    with open(app_file_name, 'w', encoding='utf-8') as f:
        json.dump(app_vars.__dict__, f, indent=4)
    
    if save_model == 'To my folder':
        if os.path.exists(torch_file_paths.save_checkpoints_user):
            shutil.rmtree(torch_file_paths.save_checkpoints_user)
        shutil.copytree(torch_file_paths.checkpoints_user, torch_file_paths.save_checkpoints_user)
        
        if os.path.exists(torch_file_paths.save_smiles_user):
            shutil.rmtree(torch_file_paths.save_smiles_user)
        shutil.copytree(torch_file_paths.input_smiles_user, torch_file_paths.save_smiles_user)
        
    elif save_model == 'To Master':
        if os.path.exists(torch_file_paths.save_checkpoints):
            shutil.rmtree(torch_file_paths.save_checkpoints)
        shutil.copytree(torch_file_paths.checkpoints_user, torch_file_paths.save_checkpoints)
        
        if os.path.exists(torch_file_paths.save_smiles_dir):
            shutil.rmtree(torch_file_paths.save_smiles_dir)
        shutil.copytree(torch_file_paths.input_smiles_user, torch_file_paths.save_smiles_dir)
        












