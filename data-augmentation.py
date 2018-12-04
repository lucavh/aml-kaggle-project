
# coding: utf-8

# In[27]:


from __future__ import print_function, division
import os
from scipy import ndarray
import skimage as sk
import pandas as pd
from skimage import io, transform, util
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def data_augment(file_dir, image_dir, rotation, shear):
    
    training_data = pd.read_csv(file_dir+'train_onelabel.csv')
    IDs_train = training_data['image'].tolist()
    
    c = 0

    for ID in IDs_train:
        img_name = os.path.join(image_dir+ID)
        X = io.imread(img_name)

        c += 1
        if c > 2:
            break

        for deg in rotation:
            X_rot = sk.transform.rotate(X, deg, resize=True, mode='constant', cval=1.0)

            for sh in shear: 
                if deg==0 and sh==0:
                    continue
                else:
                    tform = sk.transform.AffineTransform(rotation=np.deg2rad(0), shear=np.deg2rad(sh))
                    X_shear = sk.transform.warp(X_rot, tform, mode='constant', cval=1.0)

                    new_file_name = ID[:-4]+'_'+str(deg)+'_'+str(sh)+'.jpg'
                    sk.io.imsave(new_file_name, X_shear)
        
        if c % 100 == 0:
            print('Progress:',c)

data_augment(file_dir='', image_dir='train_images/', rotation=[0,90,180,270], shear=[-20,0,20])

