import streamlit as st
import torch
from chem_prop_util import *

    
def app_header():
    any_gpu = torch.cuda.is_available()
    gpu_str = 'GPU is available' if any_gpu else 'GPU is not available'

    st.subheader(f'This app uses ChemProp 2.x for propertyies/activities training and prediction. {gpu_str}')
    read_me_exp = st.expander(f'About ChemProp and Delaney Dataset.', expanded=False)
    with read_me_exp:
        url = "https://chemprop.readthedocs.io/en/latest"
        st.subheader('ChemProp:')
        st.markdown("Chemprop is a message passing neural network for molecular property prediction. [ChemProp Documentation](%s)" % url)
        st.markdown('Yang, et al, **Analyzing Learned Molecular Representations for Property Prediction**, _J. Chem. Inf. Model._ 2019, 59, 3370−3388')
        st.markdown('Stokes, et al, **A Deep Learning Approach to Antibiotic Discovery**, _Cell_. 2020, 180, 688-702')
        st.subheader('Delaney Solubility Data Set:')
        st.markdown('Delaney, **Estimating Aqueous Solubility Directly from Molecular Structure**, _J. Chem. Inf. Comput. Sci._ 2004, 44, 3, 1000–1005.')
        st.write('Delaney Solubility Data Set is included as a reference to validate our models.')
        st.write('***')
    
def app_setup():
    sel0, sel1, sel2 = st. columns(3)
    with sel0:
        login_name = st.text_input('User name')

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

    new_or_existing = st.radio('New model or using existing model?', ['Work with an Existing Model', 'Create New Model'], horizontal=True)
    if new_or_existing == 'Create New Model':
        new_model = True
    else:
        new_model = False
    st.write('***')

    return login_name, study, apply_log, excluded_list, new_model, overriddeen_container