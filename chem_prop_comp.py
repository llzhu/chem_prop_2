import streamlit as st
import torch
from chem_prop_util import *
import plotly.express as px
from chemprop import models
import json
import pickle


    
def app_header():
    any_gpu = torch.cuda.is_available()
    gpu_str = 'GPU is available' if any_gpu else 'GPU is not available'

    st.subheader(f'This app uses ChemProp 2.x for propertyies/activities training and prediction. {gpu_str}')
    read_me_exp = st.expander(f'About ChemProp and Datasets.', expanded=False)
    with read_me_exp:
        url = "https://chemprop.readthedocs.io/en/latest"
        st.subheader('ChemProp:')
        st.markdown("Chemprop is a message passing neural network for molecular property prediction. [ChemProp Documentation](%s)" % url)
        st.markdown('Yang, et al, **Analyzing Learned Molecular Representations for Property Prediction**, _J. Chem. Inf. Model._ 2019, 59, 3370−3388')
        st.subheader('Delaney Solubility Data Set:')
        st.markdown('Delaney, **Estimating Aqueous Solubility Directly from Molecular Structure**, _J. Chem. Inf. Comput. Sci._ 2004, 44, 3, 1000–1005.')
        st.subheader('Thrombin_IC50:')
        st.markdown('IC50 against human thrombin CHEMBL204 are from ChEMBL with the following query:')
        st.markdown("""activity.filter(target_chembl_id='CHEMBL204').filter(standard_type='IC50').filter(standard_relation='=')
                       .only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']""")
        st.write('***')
    
def app_setup():
    sel0, sel1, sel2 = st. columns(3)
    with sel0:
        login_name = st.text_input('User name', placeholder='Enter a user name you can remember')

    with sel1:
        study = st.selectbox('Pick a dataset', ['--', DELANEY, THROBIN_IC50, AD_HOC])
        apply_log: bool = st.checkbox("Apply log scale?")
        overriddeen_container = st.container()
        if study == DELANEY:
            apply_log = False

    with sel2:
        list_to_exclude = st.text_area('Exclude the following in the model training:')
        excluded_list=[]
        if list_to_exclude:
            excluded_list = get_list(list_to_exclude)
    
    st.write('***')

    new_or_existing = st.radio('New model or using existing model?', ['Work with an Existing Model', 'Create New Model'], 
                               horizontal=True, disabled=(login_name=='' or study=='--'))
    if new_or_existing == 'Create New Model':
        new_model = True
    else:
        new_model = False
    st.write('***')

    return login_name, study, apply_log, excluded_list, new_model, overriddeen_container

def get_model_and_paras(env:Env, app_vars:AppVars) -> models.MPNN:
    user_dir = os.path.join(env.app_data, app_vars.login_name, app_vars.study)
    master_dir = os.path.join(env.app_data, app_vars.study)
    input_files_user = os.path.join(user_dir, INPUT_FILES_DIR)
    checkpoints_user = os.path.join(user_dir, CHECKPOINTS_DIR)

    use_saved_model = st.selectbox(f'Use previously saved {app_vars.study} model:', MODEL_OPTIONS)
    app_dir = ''
    checkpoint_dir = ''
    if use_saved_model == MY_MODEL:
        app_dir = os.path.join(user_dir, INPUT_FILES_DIR)
        checkpoint_dir = os.path.join(user_dir, CHECKPOINTS_DIR)
    elif use_saved_model == MASTER_MODEL:
        app_dir = os.path.join(master_dir, INPUT_FILES_DIR)
        checkpoint_dir = os.path.join(master_dir, CHECKPOINTS_DIR)
    
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
        st.write("Selected model does not exist.")
        st.stop()

    model_files = os.listdir(checkpoint_dir)
    model_files = [ f for f in model_files if re.match(MODEL_FILE_PATTERN, f)]
    best_model = min(model_files, key=lambda x:  get_loss_val(x))
    mpnn = models.MPNN.load_from_file(os.path.join(checkpoint_dir, best_model))

    return mpnn, model_paras

def get_model_and_paras_from_s3(env:Env, app_vars:AppVars) -> models.MPNN:

    """ will be replaced by get_model_paras_from_s3 in chem_prop_util
    """
    user_dir = os.path.join(env.app_data, app_vars.login_name, app_vars.study)
    master_dir = os.path.join(env.app_data, app_vars.study)

    use_saved_model = st.selectbox(f'Use previously saved {app_vars.study} model:', MODEL_OPTIONS)
    app_dir = ''
    checkpoint_dir = ''
    if use_saved_model == MY_MODEL:
        app_dir = os.path.join(user_dir, INPUT_FILES_DIR)
        checkpoint_dir = os.path.join(user_dir, CHECKPOINTS_DIR)
    elif use_saved_model == MASTER_MODEL:
        app_dir = os.path.join(master_dir, INPUT_FILES_DIR)
        checkpoint_dir = os.path.join(master_dir, CHECKPOINTS_DIR)
    
    # get app_vars
    app_file_key = os.path.join(app_dir, APP_FILE).replace('\\', '/')
    app_file = get_from_s3(env.s3_bucket, app_file_key)
    app_vars = json.load(app_file['Body'])
    app_vars =AppVars(**app_vars)
        
    # get best model
    checkpoints_prefix = checkpoint_dir.replace('\\', '/')+'/'
    model_files = list_prefix(env.s3_bucket, checkpoints_prefix)
    model_files = [ f.replace(checkpoints_prefix, '') for f in model_files]


    if model_files and len(model_files)==0:
        st.error('No saved model available.')
        st.stop()

    model_files = [ f for f in model_files if re.match(MODEL_FILE_PATTERN, f)]
         
    best_model = min(model_files, key=lambda x:  get_loss_val(x))
    model_s3key = os.path.join(checkpoint_dir, best_model).replace('\\', '/')

    # get best model
    local_tmp_file = os.path.join(env.app_data, get_tmp_fiilename('tmp', 'ckp'))
    mpnn = load_model(env.s3_bucket, model_s3key, local_tmp_file)
    os.remove(local_tmp_file)
    # get model_paras
    model_paras = get_from_s3pickle(env.s3_bucket, os.path.join(checkpoint_dir, PARAS_FILE).replace('\\', '/'))


    return mpnn, model_paras, app_vars


def fig_df_structure(df, expt_label, pred_label, df_container, mol_container, highlight_only):

    fig = px.scatter(
        df,
        x=expt_label,
        y=pred_label,
        custom_data=["row_id"] 
    )

    fig.update_layout(
    shapes=[
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
            )
        ]
    )

    event = st.plotly_chart(
        fig,
        on_select="rerun",
        width='stretch'
    )

    if event and event.selection and event.selection["points"]:
        selected_ids = [
            point["customdata"]['0'] for point in event.selection["points"]
        ]

    
        def highlight_row(row):
            if row.name in selected_ids:
                return ['background-color: yellow'] * len(row)
            else:
                return [''] * len(row)

        if highlight_only:
            df = df[df["row_id"].isin(selected_ids)]

        style_df = df.style.apply(highlight_row, axis=1)
        df_container.dataframe(style_df, hide_index=True)

        
        smi = df.at[selected_ids[0], SMILES]
        mol = Chem.MolFromSmiles(smi)
        mol_container.write('Selected mol:')
        mol_container.write( moltosvg(mol), unsafe_allow_html=True) 

    else:
        df_container.dataframe(df, hide_index=True)