from weakref import ref
import SimpleITK
import numpy as np
import json
import os
from pathlib import Path
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")

import monai
import torch
import argparse
import glob
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import dice
from bayes_opt import BayesianOptimization
from skimage.morphology import remove_small_objects, remove_small_holes
import csv

def crf(image, pred, sdims=5., schan=5., compat=10):
    image = np.transpose(image, [1,2,3,0])
    pair_energy = create_pairwise_bilateral(sdims=(sdims,)*3, schan=(schan,)*image.shape[-1], img=image, chdim=3)
    d = dcrf.DenseCRF(np.prod(image.shape[:-1]), pred.shape[0])
    U = unary_from_softmax(pred)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pair_energy, compat=compat)
    out = d.inference(5)
    out = np.asarray(out, np.float32).reshape(pred.shape)
    return out

def reslice(image, reference=None, target_spacing=[1.,1.,1.]):
    if reference is not None:
        dims = image.GetDimension()
        resample = SimpleITK.ResampleImageFilter()
        resample.SetReferenceImage(reference)
        resample.SetInterpolator(SimpleITK.sitkLinear)
        resample.SetTransform(SimpleITK.AffineTransform(dims))

        resample.SetOutputSpacing(reference.GetSpacing())
        resample.SetSize(reference.GetSize())
        resample.SetOutputDirection(reference.GetDirection())
        resample.SetOutputOrigin(reference.GetOrigin())
        resample.SetDefaultPixelValue(0)

        newimage = resample.Execute(image)
        return newimage
    else:
        orig_spacing = image.GetSpacing()
        orig_size = image.GetSize()
        target_size = [int(round(osz*ospc/tspc)) \
            for osz,ospc,tspc in zip(orig_size, orig_spacing, target_spacing)]
        return SimpleITK.Resample(image, target_size, SimpleITK.Transform(), SimpleITK.sitkLinear,
                        image.GetOrigin(), target_spacing, image.GetDirection(), 0,
                        image.GetPixelID())

# dwi_list = glob.glob('/Users/liamchalcroft/Downloads/Dropbox/isles_dwi/*.nii.gz')
# adc_list = [os.path.join('/Users/liamchalcroft/Downloads/Dropbox/isles_adc', dwi.split('/')[-1].replace('dwi','adc')) for dwi in dwi_list]
# flair_list = [os.path.join('/Users/liamchalcroft/Downloads/Dropbox/isles_flair', dwi.split('/')[-1].replace('dwi','flair')) for dwi in dwi_list]
# gt_list = [os.path.join('/Users/liamchalcroft/Downloads/Dropbox/isles_labs', dwi.split('/')[-1].replace('dwi','msk')) for dwi in dwi_list]
# pred_list = [os.path.join('/Users/liamchalcroft/Downloads/Dropbox/isles_preds', dwi.split('/')[-1]) for dwi in dwi_list]

dwi_list = glob.glob('/home/lchalcroft/mdunet/isles_dwi/*.nii.gz')
adc_list = [os.path.join('/home/lchalcroft/mdunet/isles_adc', dwi.split('/')[-1].replace('dwi','adc')) for dwi in dwi_list]
flair_list = [os.path.join('/home/lchalcroft/mdunet/isles_flair', dwi.split('/')[-1].replace('dwi','flair')) for dwi in dwi_list]
gt_list = [os.path.join('/home/lchalcroft/mdunet/isles_labs', dwi.split('/')[-1].replace('dwi','msk')) for dwi in dwi_list]
pred_list = [os.path.join('/home/lchalcroft/mdunet/isles_preds', dwi.split('/')[-1]) for dwi in dwi_list]


dice_list = []

for i,(dwi_path,adc_path,flair_path,gt_path,pred_path) in tqdm(enumerate(zip(dwi_list, adc_list, flair_list, gt_list, pred_list)), total=len(dwi_list)):
# for i,(dwi_path,adc_path,flair_path,gt_path,pred_path) in enumerate(zip(dwi_list, adc_list, flair_list, gt_list, pred_list)):
    
    dwi_image = SimpleITK.ReadImage(dwi_path)
    adc_image = SimpleITK.ReadImage(adc_path)
    flair_image = SimpleITK.ReadImage(flair_path)

    gt_image = SimpleITK.ReadImage(gt_path)
    pred_image = SimpleITK.ReadImage(pred_path)

    dwi_image_rs = reslice(dwi_image, reference=flair_image)
    adc_image_rs = reslice(adc_image, reference=flair_image)
    flair_image_rs = reslice(flair_image, reference=flair_image)
    dwi_image_rs = reslice(dwi_image_rs, reference=dwi_image)
    adc_image_rs = reslice(adc_image_rs, reference=dwi_image)
    flair_image_rs = reslice(flair_image_rs, reference=dwi_image)

    gt_image_rs = reslice(gt_image, reference=dwi_image)
    pred_image_rs = reslice(pred_image, reference=dwi_image)

    dwi_image_data = SimpleITK.GetArrayFromImage(dwi_image_rs)
    adc_image_data = SimpleITK.GetArrayFromImage(adc_image_rs)
    flair_image_data = SimpleITK.GetArrayFromImage(flair_image_rs)
    gt_image_data = SimpleITK.GetArrayFromImage(gt_image_rs)
    pred_image_data = SimpleITK.GetArrayFromImage(pred_image_rs)

    img = np.stack([adc_image_data, dwi_image_data, flair_image_data])

    img = monai.transforms.NormalizeIntensity(nonzero=True,channel_wise=True)(img)
    orig_shape = img.shape[1:]

    # plt.figure()
    # plt.imshow(img[0,...,50])
    # plt.show()

    img = monai.transforms.ToTensor(dtype=torch.float32)(img).cpu().detach().numpy()

    img_crf = img
    img_crf = img_crf - img_crf.min()
    img_crf = 255 * (img_crf / img_crf.max())
    img_crf[img_crf < 0] = 0
    img_crf[img_crf > 255] = 255
    img_crf = np.asarray(img_crf, np.uint8)
    pred_crf = np.asarray(pred_image_data, np.float32)
    pred_crf = np.stack([1.-pred_crf, pred_crf])
    final_pred = pred_crf
    # final_pred = crf(img_crf, pred_crf, (0.1,)*3, (0.1,), 3)
    prediction = final_pred[1]
        
    prediction[prediction > 1] = 0

    prediction = (prediction > 0.5)

    # prediction = remove_small_holes(prediction, 10, 1)
    # prediction = remove_small_objects(prediction, 10, 1)

    prediction = prediction.astype(int)

    dice_val = dice(prediction.flatten(), gt_image_data.flatten())

    dice_list.append(dice_val)

    # print(np.mean(dice_list))

mean_dice = np.nanmean(dice_list)

print('Baseline dice (i.e. no CRF): ', mean_dice)
