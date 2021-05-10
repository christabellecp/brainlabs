import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
import itertools
from skimage import io
import random
from pathlib import Path
from random import randint
from volumentations import *
import nibabel as nib
import matplotlib.pyplot as plt
import pickle


class CogDataset3d(torch.utils.data.Dataset):
   
    """
    Class for getting individual transformations and data
    Args:
        input_dir = path of input images
        target_dir = path of target images
        input = list of filenames for input
        target = list of filenames for target
        transform = Images transformation (default: False)
        crop = crop size
        df = dataframe for cognitive scores
    Output:
        Transformed input
        Transformed image target
        ADAS11 score
        MMSE score
        filename
        
    """
    
    def __init__(self, input_dir, target_dir, input_files, target_files, df, transform=False, crop = (128,128,128)):
        self.input_dir = input_dir 
        self.target_dir = target_dir
        self.input = sorted(input_files)   
        self.target = sorted(target_files) 
        self.transform = transform
        self.crop = crop
        self.df = df
        
        
        patient_files = pickle.load(open( "data/patient_files.data", "rb" ))
        self.patient_files = list(set(map(lambda x: x[0], patient_files)))
        
        self.train_transforms = Compose([RandomCrop(shape = (128,128,128), always_apply=True),
                                        ElasticTransform((0, 0.20), interpolation=4, p=1),
                                         RandomRotate90((0,1), p=0.5),
                                        #RandomGamma(gamma_limit=(0.5, 1.5), p=0.8),
                                         Normalize(always_apply=True)], p=1.0)

        self.val_transforms = Compose([CenterCrop(shape = (128,128,128), always_apply=True),
                                       Normalize(always_apply=True)], p=1.0)

    def __len__(self):
        return len(self.input)
    
        
    def __getitem__(self, i):
        
        # grab the baseline images
        X_tr_pid = list(map(lambda x: x[8:16], [self.input[i]]))
        new_input = list(map(self.get_baseline_file, X_tr_pid))[0]
        new_target = new_input.split('.nii')[0]+'_seg.nii'
        
        inp = nib.load(self.input_dir + new_input).get_fdata()
        target = nib.load(self.target_dir + new_target).get_fdata()
        
        data = {'image': inp, 'mask': target}
        
        if self.transform == True:
            aug_data = self.train_transforms(**data)
            filename_df = self.input[i].split('.nii')[0]
        else:
            aug_data = self.val_transforms(**data)
            filename_df = self.input[i].split('.nii')[0]

        #checking if image has an associated cognitive score 
        files_have_cog = self.df['filenames'].values.tolist()
        a_score = filename_df in files_have_cog
        
        #returning the cognitive score if true
        y_adas_score = None
        if a_score == True:
            y_adas_score = self.df[self.df['filenames'] == filename_df]['ADAS11'].values[0]
            
        x, y_img = aug_data['image'], aug_data['mask']
        
        return x[None,], y_img, y_adas_score, self.input[i].split('.nii')[0]
    
    def get_baseline_file(self, current_file):
        for s in filter(lambda x: current_file in x, self.patient_files):
            return s

            
def visualize_slices(brain, start, stop, target=False, slice_type='sagittal'):
    """
    brain: instance of the dataset
    start: starting slice
    stop: ending slice
    target: return input or target
    slice_type: sagittal, coronal, or horizontal slices
    """
    rang = stop-start
    cols = int(rang/5)
    
    fig, ax = plt.subplots(cols, 5, figsize = (int(25),int(rang/(1.5))))
    fig.set_facecolor("black")
    ax = ax.flatten()
    start_idx = start

    for i in range(0,rang, 1):
        if slice_type == 'sagittal':            
            brain_in = brain[0][:,start+i,:,:]
            brain_out= brain[1][start+i,:,:]
        elif slice_type == 'coronal':
            brain_in = brain[0][:,:,start+i,:]
            brain_out= brain[1][:,start+i,:]
        elif slice_type == 'horizontal':
            brain_in = brain[0][:,:,:,start+i]
            brain_out= brain[1][:,:,start+i]

        shape_img = np.shape(brain_in)
        if target == False:
            ax[i].set_facecolor('black')
            ax[i].set_title(f'slice: {start_idx}')
            ax[i].imshow(brain_in.reshape(shape_img[1],shape_img[2]))
        if target == True:
            ax[i].set_facecolor('black')
            ax[i].set_title(f'slice: {start_idx}')
            ax[i].imshow(brain_out)
        start_idx+=1
    plt.tight_layout()
