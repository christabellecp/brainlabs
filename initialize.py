from imports import *

def split_train_val(X_tr, X_v, y_tr, y_v, df):
    X_train_files = [f.split('.nii')[0] for f in X_tr]
    X_val_files = [f.split('.nii')[0] for f in X_v]
    y_train_files = [f.split('.nii')[0][:-4] for f in y_tr]
    y_val_files = [f.split('.nii')[0][:-4] for f in y_v]
    
    X_train = df[df['filenames'].isin(X_train_files)]
    X_val = df[df['filenames'].isin(X_val_files)]

    y_adas_train = X_train['ADAS11'].values
    y_adas_val = X_val['ADAS11'].values
    y_mmse_train = X_train['MMSE'].values
    y_mmse_val = X_val['MMSE'].values

    X_train = X_train.drop(columns=['filenames', 'ADAS11', 'MMSE'])
    X_val = X_val.drop(columns=['filenames', 'ADAS11', 'MMSE'])
    
    
    return X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val

def initialize_data():
    input_path = '/home/cpabalan/brainlabs_cp/brainlabs_prep/data/input_files/'
    target_path = '/home/cpabalan/brainlabs_cp/brainlabs_prep/data/target_files/'
    #csv_path = 'dfs/df_correct_dtypes.csv'
    csv_path = 'data/cleaned_df_4_13.csv'
    df = pd.read_csv(csv_path)
    df = df.drop(columns = ['MOCA_bl', 'EcogPtMem_bl', 'EcogPtLang_bl', 'EcogPtVisspat_bl',
       'EcogPtPlan_bl', 'EcogPtOrgan_bl', 'EcogPtDivatt_bl',
       'EcogPtTotal_bl', 'EcogSPMem_bl', 'EcogSPLang_bl',
       'EcogSPVisspat_bl', 'EcogSPPlan_bl', 'EcogSPOrgan_bl',
       'EcogSPDivatt_bl', 'EcogSPTotal_bl', 'Ventricles_bl',
       'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl',
       'MidTemp_bl', 'ICV_bl', 'FDG_bl', 'AV45_bl', 'Ventricles_bl',
       'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl',
       'MidTemp_bl', 'ICV_bl', 'FDG_bl', 'AV45_bl', 'DX_bl_AD',
       'DX_bl_CN', 'DX_bl_EMCI', 'DX_bl_LMCI', 'DX_bl_SMC', 'DX_bl_nan',
       'Ventricles_bl', 'Hippocampus_bl', 'WholeBrain_bl',
       'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl', 'ICV_bl', 'FDG_bl',
       'AV45_bl', 'DX_bl_AD', 'DX_bl_CN', 'DX_bl_EMCI', 'DX_bl_LMCI',
       'DX_bl_SMC', 'DX_bl_nan', 'Ventricles_bl', 'Hippocampus_bl',
       'WholeBrain_bl', 'Entorhinal_bl', 'Fusiform_bl', 'MidTemp_bl',
       'ICV_bl', 'FDG_bl', 'AV45_bl', 'DX_bl_AD', 'DX_bl_CN',
       'DX_bl_EMCI', 'DX_bl_LMCI', 'DX_bl_SMC', 'DX_bl_nan'])
 
    X_tr, X_v, y_tr, y_v = get_file_splits()       
    X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val = split_train_val(X_tr, X_v, y_tr, y_v, df)
    
    return X_train, X_val, y_adas_train, y_adas_val, y_mmse_train, y_mmse_val, input_path, target_path, csv_path, df
    
def get_file_splits(subset='all'):

    if subset == 'all':
        paths = ['data/X_train_flist.data', 'data/y_train_flist.data', 'data/X_val_flist.data', 'data/y_val_flist.data']

    if subset == 'current':
        paths = ['fnames_split/train_curr.data', 'fnames_split/y_train_curr.data',
                 'fnames_split/valid_curr.data', 'fnames_split/y_valid_curr.data']

    if subset == '<=2':
        paths = ['fnames_split/train_2years.data', 'fnames_split/y_train_2years.data',
                 'fnames_split/valid_2years.data', 'fnames_split/y_valid_2years.data']

    if subset == '<=4':
        paths = ['fnames_split/train_4years.data', 'fnames_split/y_train_4years.data',
                 'fnames_split/valid_4years.data', 'fnames_split/y_valid_4years.data']

    if subset == '4+':
        paths = ['fnames_split/train_4plyears.data', 'fnames_split/y_train_4plyears.data',
                 'fnames_split/valid_4plyears.data', 'fnames_split/y_valid_4plyears.data']

    with open(paths[0], 'rb') as filehandle:
        X_tr = pickle.load(filehandle)
    with open(paths[1], 'rb') as filehandle:
        y_tr = pickle.load(filehandle)
    with open(paths[2], 'rb') as filehandle:
        X_v = pickle.load(filehandle)
    with open(paths[3], 'rb') as filehandle:
        y_v = pickle.load(filehandle)

    return X_tr, X_v, y_tr, y_v
    


def get_ds_dl(subset='all', batch_size=10, num_workers=16):
    
    _, _, _, _, _, _, input_path, target_path, csv_path, df = initialize_data()
    X_tr, X_v, y_tr, y_v = get_file_splits(subset=subset) 
    ds_train = CogDataset3d(input_path, target_path, X_tr, y_tr, df, transform=True, crop = (128,128))
    ds_val = CogDataset3d(input_path, target_path, X_v, y_v, df, transform=False, crop = (128,128))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    return ds_train, ds_val, dl_train, dl_val

def tab_predict(pipe, X_train, y_train, X_val, y_val, name = 'Model'):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    train_preds = pipe.predict(X_train)

    print(f"{f'{name} Train Loss'}: {round(mean_squared_error(y_train, train_preds),3)}")
    print(f"{f'{name}  Train R2  '}: {round(r2_score(y_train, train_preds),3)}\n")
    print(f"{f'{name}  Valid Loss'}: {round(mean_squared_error(y_val, preds),3)}")
    print(f"{f'{name}  Valid R2  '}: {round(r2_score(y_val, preds),3)}\n")
    
    return train_preds, preds
