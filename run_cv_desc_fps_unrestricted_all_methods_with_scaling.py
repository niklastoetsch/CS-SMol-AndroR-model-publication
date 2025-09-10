
import pandas as pd
from rdkit import Chem
import pickle
import matplotlib.pyplot as plt

import ml
import utils
from analysis import CV, Predictions


andror_df_pub= pd.read_csv("/home/phns/projects/AndroR/HTS_model/Niklas_scripts/data/AndroR_bind_unrestricted_June_2025_rdkit_standardized_no_stereo_unique_SMILES_descriptors_and_fingerprints_new_rdkit.txt", sep="\t")

# adjust columns names
andror_df_pub=andror_df_pub.rename(columns={"SMILES_STD_FLAT": "flat_smiles"})
andror_df_pub=andror_df_pub.rename(columns={"class": "final class"})
#andror_df_pub['final class'] = andror_df_pub['final class'].replace({1: 'inhibitor', 0: 'inactive'})

fps = utils.get_fingerprints(andror_df_pub)

groups = utils.get_cluster_assignments_from_fps(fps, 0.65, chunk_size=5000)

data_desc_fp = andror_df_pub.drop(columns=['SUBSTANCE', 'flat_smiles', 'final class'])
#print(data_desc_fp.head())

#from sklearn import preprocessing

#scaler = preprocessing.StandardScaler().fit(data_desc_fp)

#data_desc_fp_scaled = scaler.fit_transform(data_desc_fp)
#print("scaling conducted")

#print(data_desc_fp_scaled.head())

#df2 = andror_df_pub.iloc[:, :-1024]
#data_desc = df2.drop(columns=['SUBSTANCE', 'flat_smiles', 'final class'])
#data_desc.head()

#df2 = andror_df_pub.drop(andror_df_pub.iloc[:, 3:220],axis = 1)
#data_fps = df2.drop(columns=['SUBSTANCE', 'flat_smiles', 'final class'])
#data_fps.head()

#models = ['gbt','svm','catboost','xgb','lr','rf']
#models = ['gbt','catboost','xgb','lr','rf','svm']
models = ['xgb']
for model in models:
    print(model)
    splits_desc_fp, pipelines_desc_fp = ml.run_or_retrieve_from_disc_model_scaled(
        X=data_desc_fp, 
        y=andror_df_pub["final class"], 
        groups=groups, 
        training_name=f"unrestricted_new_desc_fps_scaled_{model}_HPC",
        folder = "/home/phns/projects/AndroR/HTS_model/Niklas_scripts/",
        model = model
    )
