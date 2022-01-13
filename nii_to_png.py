import numpy as np
import os
import nibabel as nib
import imageio
import glob

"""
filepath is a list, so each element is the path of .nii
"""

def nii_to_image():
    n = 0
    for f in filepath:
        # load img
        img = nib.load(f)
        img_fdata = img.get_fdata()
        # fname as original just delete .nii(ex:ADNI_002_S_1018_MR_MPR____N3__Scaled_Br_20090106140133108_S60846_I132795)
        # fname = f.split('/')[6].replace('.nii', '')
        fname = 'CN_{}'.format(n)
        img_f_path = os.path.join(imgfile, fname)
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)

        # z is total length of img
        (x, y, z) = img.shape
        for i in range(z):
            # choose slice(COR/AXL/SAG) by [i, :, :] or [:, i, :]
            silce = img_fdata[i, :, :]
            # save png by 0.1.2.3....
            imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)
        n +=1

if __name__ == '__main__':
    # from ADNI file(AD/CN/MCI) but ADNI file have ADNI/AD/xxx/xxx/xxx/xxx/xxx.nii
    # glob function can search all son_file by using *
    filepath = glob.glob('ADNI/CN/*/*/*/*/*')

    # where to save img in this case i save it in ADNI_png file
    imgfile = 'ADNI_png'

    # create ADNI_png file
    if not os.path.exists(imgfile):
        os.mkdir(imgfile)
    # call nii_to_image function
    nii_to_image()
