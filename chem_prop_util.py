import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor,AllChem
from sklearn import preprocessing
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from io import StringIO, BytesIO
import os
import shutil
from chemprop import data, featurizers, models, nn, utils
from rdkit.Chem.Descriptors import *
import re
from dataclasses import dataclass, field
from typing import List
import math
from icecream import ic
import base64
from itertools import chain
import boto3
import pickle
import datetime
import random
import json

BMP = 'Bond_Message_Passing' 
AMP = 'Atom_Message_Passing'

INPUT_FILES_DIR = 'input_smiles'
CHECKPOINTS_DIR = 'checkpoints'
INPUT_SMILES_FILE = 'input_smiles.csv'
TEST_SMILES_FILE = 'test_smiles.csv'
VAL_SMILES_FILE = 'val_smiles.csv'
TRAIN_SMILES_FILE = 'train_smiles.csv'
PARAS_FILE = 'paras.pkl'
APP_FILE = 'app.json'

SPLIT_RANDOM = 'random'
SPLIT_SCAFFOLD_BALANCED = 'scaffold_balanced'
SPLIT_SCAFFOLD_RANDOM = 'scaffold_random'

MODEL_FILE_PATTERN = '^(\w*)_Message_Passing_epoch=(\d*)_val_loss=(.*)\.ckpt$'

DELANEY = 'Solubility_Delaney'
THROBIN_IC50 = 'Thrombin_IC50'
AD_HOC = 'ad_hoc'

SMILES = 'SMILES'
DO_NOT_HIGHLIGHT = "Do not highlight"
HIGHLIGHT_ALL = "Highlight All"
HIGHLIGHT_UNIQUE = "Highlight Unique"
COMPOUND_ID = 'Compound_ID'
STRUCTURE = 'Compound'
CHEMBL_UNIT = 'standard_units'
CHEMBL_SMILES = 'canonical_smiles'
CHEMBL_CMPD_ID = 'molecule_chembl_id'

RDKIT_2D_DESCRIPTOR_NONE = 'No 2D Descriptors'
RDKIT_2D_DESCRIPTOR = '2D Descriptors'
RDKIT_2D_DESCRIPTOR_SCALED = 'Scaled 2D Descriptors'

MY_MODEL = 'My model'
MASTER_MODEL = 'Master Model'
MODEL_OPTIONS = [MY_MODEL, MASTER_MODEL]


def get_tmp_fiilename(file_base, ext):
    ts = int(datetime.datetime.now().timestamp() * 1000000)
    rand_num = random.randint(0, 9999)
    return f'{file_base}_{ts}_{rand_num}.{ext}'

s3client = boto3.client(
    "s3",
    aws_access_key_id = st.secrets['aws_access_key_id'],
    aws_secret_access_key = st.secrets['aws_secret_access_key'],
    region_name = st.secrets['region_name']
)

s3resource = boto3.resource(
    "s3",
    aws_access_key_id = st.secrets['aws_access_key_id'],
    aws_secret_access_key = st.secrets['aws_secret_access_key'],
    region_name = st.secrets['region_name']
)


def download_from_s3(bucket_name, key, local_file):
    s3client.download_file(
    bucket_name,
    key,
    local_file
)
    

def load_model(bucket_name, key, local_file):
    download_from_s3(bucket_name, key, local_file)
    mpnn = models.MPNN.load_from_file(local_file)
    return mpnn


def pickle_to_s3(data_obj, bucket, key):
    # Serialize to memory
    buffer = BytesIO()
    pickle.dump(data_obj, buffer)
    buffer.seek(0)

    # Upload to S3
    s3client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue()
    )

def get_from_s3pickle(bucket, key):
    # Download pickle from S3
    ic(bucket)
    ic(key)
    pickle_obj = s3client.get_object(Bucket=bucket, Key=key)

    # Deserialize
    buffer = BytesIO(pickle_obj["Body"].read())
    data_obj = pickle.load(buffer)

    return data_obj

def get_from_s3(bucket, key):
    # Download pickle from S3
    obj = s3client.get_object(Bucket=bucket, Key=key)
    return obj

def any_contents(bucket, prefix):
    response = s3client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/"
    )
    return len(response.get("Contents", []))>0
    
    

def get_df_from_s3csv(bucket, key):
    # Get object from S3
    obj = s3client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(BytesIO(obj["Body"].read()))
    return df

def copy_to_s3(local_folder:str, bucket_name:str, s3_prefix:str):
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)

            # create relative path
            relative_path = os.path.relpath(local_path, local_folder)

            # convert to S3 key (use forward slashes)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            s3client.upload_file(local_path, bucket_name, s3_key)


def list_prefix(bucket_name, prefix):
    response = s3client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=prefix,
        Delimiter="/"
    )

    file_list = []
    for obj in response.get("Contents", []):
        file_list.append(obj["Key"])

    return file_list




def get_loss_val(model_file, pattern_str=MODEL_FILE_PATTERN):
    pattern =re.compile(pattern_str)
    match = pattern.match(model_file)
    if match:
        return float(match.group(3))


@dataclass
class AppVars:
    study: str
    is_admin: str
    login_name: str
    orig_col_name: str = ''  # column name of the original data
    expt_col_name: str = ''  # column name used in training/predication. = or log of orig_col_name
    apply_log: bool = True   # The data are log scaled for training; expt_col_name=log(orig_col_name)
    dset_size: int = 0

    
@dataclass
class ModelParas:
    mol_graph: str = BMP
    num_epoch: int = 50
    split_type: str = SPLIT_RANDOM
    dset_size: int = 0
    add_fp: bool = True
    add_descriptors: bool = True
    scale_descriptors: bool = False
    dc_scaler = None
    extra_feature_size:int = 0
    model_created: str = ''


@dataclass
class Env:
    src_data: str
    app_data: str
    admins: List[str] = field(default_factory=list)
    modelers: List[str] = field(default_factory=list)
    s3_bucket: str = ''

@st.cache_data
def get_model_paras_from_s3(env:Env, app_vars:AppVars, base_dir:str) -> models.MPNN:
    
    app_dir = os.path.join(base_dir, INPUT_FILES_DIR)
    checkpoint_dir = os.path.join(base_dir, CHECKPOINTS_DIR)
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

# Currently not used
def get_rdkit_fp(mol_list, nb=2048, radius=4):
    n_mol = len(mol_list)
    X = np.zeros([n_mol, nb])
    for i in range(n_mol):
        fp1 = AllChem.GetHashedMorganFingerprint(mol_list[i], radius, nBits=nb)
        array1 = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp1, array1)

        for j in range(nb):
            X[i, j] = array1[j]

    X = pd.DataFrame(data=X)  # Make it a dataframe
    return X

# Currently not used.
def remove_low_variance(input_data, threshold=0.1) -> pd.DataFrame:
    # input_data exexted to be np.ndarray or pd.Dataframe
    if isinstance(input_data, np.ndarray):
        input_data = pd.DataFrame(data=input_data)  
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]


def delete_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_df_csv(df):
    f = StringIO()
    df.to_csv(f, index=False)
    return f


def convert_ugperml_to_um(row):
    if row['unit'] == 'ug/mL':
        return (row['value']/row['mw'])*1000.0
    else:
        return row['value']

def convert_df_csv(df, index=False):
    return df.to_csv(index=index).encode('utf-8')


# def get_chemdl_activity_df(standard_type, target_organism):
#
#     activity = new_client.activity
#     activity = activity.filter(
#             standard_type=standard_type,
#             target_organism=target_organism,
#             dard_value__isnull=False,
#             standard_value__gt=0,
#             standard_units='mL.min-1.kg-1').only(
#             "molecule_chembl_id",
#             "canonical_smiles",
#             "target_organism",
#             "standard_type",
#             "standard_relation",
#             "standard_value",
#             "standard_units")
#     df = pd.DataFrame(activity)
#     df['mw'] = df['canonical_smiles'].apply(lambda x: ExactMolWt(Chem.MolFromSmiles(x)))
#
#     return df

@st.cache_data
def standarize(df_input: pd.DataFrame, study:str, value_column:str, apply_log:bool)-> tuple[pd.DataFrame, str]:
    ic(value_column)
    df_output = None
    if study == THROBIN_IC50:
        df_output = df_input[df_input[CHEMBL_UNIT]=='nM']
        df_output = df_output[~df_output[CHEMBL_SMILES].str.contains('.', regex=False, na=False)]
        df_output[value_column] = df_output[value_column].astype(float)
        if apply_log:
            expt_col_name = 'log_IC50'
            df_output[expt_col_name] = df_output[value_column].apply(math.log10)
        else:
            expt_col_name = 'IC50'
            df_output[expt_col_name] = df_output[value_column]

        column_map = {
                CHEMBL_CMPD_ID: COMPOUND_ID,
                CHEMBL_SMILES: SMILES
            }
        df_output = df_output.rename(columns=column_map) 
        col_output = [COMPOUND_ID, expt_col_name, SMILES]
        df_output = df_output[col_output] 
        df_output[expt_col_name] = df_output[expt_col_name].astype(float)


    return df_output, expt_col_name


def get_list(inputs: str)->list[str]: 
    input_list = []
    if inputs:
        input_list = re.split(',|\n', inputs)
        input_list = [input for input in input_list if input.strip()]
    return input_list

def get_floor(in_num: float, floor: float)-> float: 
    out_num = in_num
    if in_num < floor:
        out_num = floor
    return out_num

@st.cache_data
def get_fp(mols, radius=2, fp_keys = None):

    fps = [AllChem.GetMorganFingerprint(m, radius=2) for m in mols]
    all_keys = set()
    if fp_keys is None:
        ic('Create fp keys')
        for fp in fps:
            all_keys.update(fp.GetNonzeroElements().keys())
            all_keys = sorted(all_keys)
    else:
        all_keys = fp_keys

    ic(all_keys[0])
    ic(type(all_keys[0]))
    
    data = []
    for fp in fps:
        row = []
        elements = fp.GetNonzeroElements()
        for k in all_keys:
            row.append(elements.get(k, 0))
        data.append(row)

    df = pd.DataFrame(data, columns=all_keys)
    return df

@st.cache_data
def get_rdkit_descriptors(mol_list, scale_dc:bool, _scaler=None):
    descriptor_names = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    mol_descriptors = []
    for mol in mol_list:
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)

    
    if scale_dc:
        if _scaler == None:
            _scaler = preprocessing.StandardScaler().fit(mol_descriptors)
            mol_descriptors = _scaler.transform(mol_descriptors)
        else:
            mol_descriptors = _scaler.transform(mol_descriptors)

    return np.array(mol_descriptors), _scaler
    


def get_datapoint(mols, model_paras: ModelParas, ys=None, modify_model_paras=True):
    """ calculate datapoint; model_paras are updated if modify_model_paras=True
    """
    add_fp = model_paras.add_fp
    add_dc = model_paras.add_descriptors
    
    extra_features = None 
    if add_fp:
        fp_featurizer = featurizers.MorganBinaryFeaturizer()
        fp_features = np.array([fp_featurizer(mol) for mol in mols])
        extra_features = fp_features
        # extra_features = remove_low_variance(fp_features).to_numpy()
           
    if add_dc:
        dc_features, dc_scaler = get_rdkit_descriptors(mols, scale_dc=model_paras.scale_descriptors, scaler=model_paras.dc_scaler)
        if modify_model_paras:
            model_paras.dc_scaler = dc_scaler
       
        if extra_features is not None:
            extra_features = np.concatenate([extra_features, dc_features], axis=1)
        else:
            extra_features = dc_features
    if extra_features is not None:
        model_paras.extra_feature_size = extra_features.shape[1]
     

    if ys is None:
        if extra_features is not None:
            data_point = [data.MoleculeDatapoint(mol, x_d=X_d) for mol, X_d in zip(mols, extra_features)]
        else: 
            data_point = [data.MoleculeDatapoint(mol) for mol in mols]
    else:
        if extra_features is not None:
            data_point = [data.MoleculeDatapoint(mol, y, x_d=X_d) for mol, y, X_d in zip(mols, ys, extra_features)]
        else: 
            data_point = [data.MoleculeDatapoint(mol, y) for mol, y in zip(mols, ys)]

    return data_point


@st.cache_data
def moltosvg(mol, molSize = (800,400), kekulize = False, highlight_sub=None, highlight_mode=DO_NOT_HIGHLIGHT):
    
    if  highlight_sub == None: # Cannot highlight if highlight_sub not provided
        highlight_mode=DO_NOT_HIGHLIGHT
    
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    
    if highlight_mode == DO_NOT_HIGHLIGHT:
        drawer.DrawMolecule(mc)
    elif highlight_mode in (HIGHLIGHT_UNIQUE, HIGHLIGHT_ALL):
        highlight_tt = mc.GetSubstructMatches(highlight_sub)
        hightlight_shape = np.shape(highlight_tt)
        if hightlight_shape[0] == 1:
            highlight_tuple = tuple(chain.from_iterable(highlight_tt))
            drawer.DrawMolecule(mc, highlightAtoms=highlight_tuple)
        else:
            if highlight_mode == HIGHLIGHT_UNIQUE:
                drawer.DrawMolecule(mc)
            elif highlight_mode == HIGHLIGHT_ALL:
                highlight_tuple = tuple(chain.from_iterable(highlight_tt))
                drawer.DrawMolecule(mc, highlightAtoms=highlight_tuple)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:','')
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = rf'<img src="data:image/svg+xml;base64, {b64}"/>'
    return html