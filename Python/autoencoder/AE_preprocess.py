#%% Importations

import numpy as np
from numba import jit

@jit(nopython=True)     # speeds it up (but requires numpy)
def patch_decomp(img,patch_size):
    """

    Parameters
    ----------
    img : Image numpy dont ont veut recuperer les patchs
    patch_size : size of the patches
    stride : stride between center of patches

    Returns
    -------
    patch_list : list of patches from img

    """
    patch_list = np.zeros(((img.shape[1]-(patch_size-1))*(img.shape[2]-(patch_size-1)),img.shape[0],patch_size,patch_size))
    for i in range(img.shape[1]-(patch_size-1)):
        for j in range(img.shape[2]-(patch_size-1)):
            patch_list[i+j*(img.shape[1]-(patch_size-1))] = img[:,i-(patch_size-1)//2:i+(patch_size-1)//2+1,j-(patch_size-1)//2:j+(patch_size-1)//2+1]
    
    return patch_list

@jit(nopython=True)     # speeds it up (but requires numpy)
def patch_recomp(patch_list,img_shape):
    """

    Parameters
    ----------
    patch_list : np.array(:,img_channels,patch_size,patch_size)
        list of patches as given by patch_decomp
    patch_size : int
        DESCRIPTION.
    img_shape : (int,int)
        shape of the img to recompose

    Returns
    -------
    img : array(img_channel,img_shape[0],img_shape[1])
        img recomposed from the patches

    """
    img = np.zeros((patch_list.shape[1],img_shape[0],img_shape[1]))
    division_factors = np.zeros((patch_list.shape[1],img_shape[0],img_shape[1]))
    
    patch_size = patch_list.shape[-1]
    
    for i in range(2,img_shape[0]-(patch_size-1)):
        for j in range(2,img_shape[1]-(patch_size-1)):
            img[:,i-(patch_size-1)//2:i+(patch_size-1)//2+1,j-(patch_size-1)//2:j+(patch_size-1)//2+1] += patch_list[i+j*img_shape[1]-(patch_size-1),:,:,:]
            division_factors[:,i-(patch_size-1)//2:i+(patch_size-1)//2+1,j-(patch_size-1)//2:j+(patch_size-1)//2+1] += 1
    
    img = img/division_factors
    
    return img


    

    
    