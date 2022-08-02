from weakref import ref
import SimpleITK
import numpy as np
import json
import os
from pathlib import Path
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/")
DEFAULT_ALGORITHM_OUTPUT_FILE_PATH = Path("/output/results.json")

import torch
from nnunet.nn_unet import NNUnet
import monai
from skimage.transform import resize
import argparse
import tempfile
import pyrobex
from pyrobex.errors import PyRobexError
import glob
import subprocess
from types import SimpleNamespace
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from tqdm import tqdm
from scipy.special import softmax

"""ONLY FOR EVALUATION STAGE - WANT TO MAKE SURE DOCKER FOLLOWS SAME ALGO"""

def _find_robex_dir() -> str:
    """finds the ROBEX source code directory"""
    file_path = Path(pyrobex.__file__).resolve()
    pyrobex_dir = file_path.parent
    robex_dist = pyrobex_dir / "ROBEX"
    return str(robex_dist)


def _find_robex_script() -> str:
    """finds the ROBEX shell script"""
    robex_dist = Path(_find_robex_dir())
    robex_script = robex_dist / "runROBEX.sh"
    if not robex_script.is_file():
        raise PyRobexError("Could not find `runROBEX.sh` script.")
    return str(robex_script)


# todo change with your team-name
class ploras():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = True  # False for running the docker!
        if self.debug:
            self._input_path = Path('/home/lchalcroft/Data/ATLAS_R2.0/Testing/')
            self._output_path = Path('/home/lchalcroft/mdunet/atlas-eval/output/')
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = self._output_path / 'results.json'
            self._case_results = []

        else:
            self._input_path = input_path
            self._output_path = output_path
            self._algorithm_output_path = self._output_path / 'stroke-lesion-segmentation'
            self._output_file = DEFAULT_ALGORITHM_OUTPUT_FILE_PATH
            self._case_results = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tta = True

        args = SimpleNamespace(exec_mode='train', data='/data', 
                                results='/results', config='../docker-atlas/config/config.pkl', logname='ploras', 
                                task='16', gpus=1, nodes=1, learning_rate=0.0002, gradient_clip_val=1.0, negative_slope=0.01, 
                                tta=tta, tb_logs=False, wandb_logs=True, wandb_project='isles', brats=False, deep_supervision=True, 
                                more_chn=False, invert_resampled_y=False, amp=True, benchmark=False, focal=False, save_ckpt=False, 
                                nfolds=5, seed=1, skip_first_n_eval=500, val_epochs=10, ckpt_path=None, 
                                ckpt_store_dir='../docker-atlas/checkpoints/', fold=0, patience=100, 
                                batch_size=4, val_batch_size=4, momentum=0.99, weight_decay=0.0001, save_preds=False, dim=3, 
                                resume_training=False, num_workers=8, epochs=2000, warmup=5, norm='instance', nvol=4, depth=5, 
                                min_fmap=4, deep_supr_num=2, res_block=False, filters=None, num_units=2, md_encoder=True, 
                                md_decoder=False, shape=False, paste=0, data2d_dim=3, oversampling=0.4, overlap=0.5, 
                                affinity='unique_contiguous', scheduler=False, optimizer='adam', blend='gaussian', 
                                train_batches=0, test_batches=0)

        self.model_paths = [
            '../docker-atlas/checkpoints/0/best.ckpt', '../docker-atlas/checkpoints/1/best.ckpt', 
            '../docker-atlas/checkpoints/2/best.ckpt', '../docker-atlas/checkpoints/3/best.ckpt', 
            '../docker-atlas/checkpoints/4/best.ckpt'
            ]
        for i,pth in enumerate(self.model_paths):
            ckpt = torch.load(pth, map_location=self.device)
            ckpt['hyper_parameters']['args'] = args
            ckpt['hyper_parameters']['args'].ckpt_store_dir += str(i)
            torch.save(ckpt, pth)
        self.models = [NNUnet.load_from_checkpoint(path, map_location=self.device) for path in self.model_paths]
        for model in self.models:
            model.to(self.device)
            model.eval()
            model.freeze()
            model.model.training = False

    def reslice(self, image, reference=None, target_spacing=[1.,1.,1.]):
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
            resample.SetDefaultPixelValue(image.GetPixelIDValue())

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

    def robex(self, image):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            robex_script = _find_robex_script()
            tmp_img_fn = tdp / "img.nii"
            SimpleITK.WriteImage(image, str(tmp_img_fn))
            stripped_fn = tdp / "stripped.nii"
            mask_fn = tdp / "mask.nii"
            args = [robex_script, tmp_img_fn, stripped_fn, mask_fn, 0]
            str_args = list(map(str, args))
            out = subprocess.run(str_args, capture_output=True)
            stripped = SimpleITK.ReadImage(str(stripped_fn))
            mask = SimpleITK.ReadImage(str(mask_fn))
        return stripped, mask

    def n4(self, image, mask=None):
        mask = mask if mask is not None else SimpleITK.OtsuThreshold(image, 0, 1, 200)
        mask = SimpleITK.Cast(mask, SimpleITK.sitkUInt8)
        corrector = SimpleITK.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(image, mask)
        bias = corrector.GetLogBiasFieldAsImage(image)
        corrected_hr = image / SimpleITK.Exp(bias)
        return corrected_hr

    def crf(self, image, pred):
        image = np.transpose(image, [1,2,3,0])
        pair_energy = create_pairwise_bilateral(sdims=(5.0,)*3, schan=(5.0,)*image.shape[-1], img=image, chdim=3)
        d = dcrf.DenseCRF(np.prod(image.shape[:-1]), pred.shape[0])
        U = unary_from_softmax(pred)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(pair_energy, compat=10)
        out = d.inference(5)
        out = np.asarray(out, np.float32).reshape(pred.shape)
        return out

    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  't1w_image' , 't1w_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        t1w_image = input_data['t1w_image']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################

        if not self.preprocessed:
            t1w_image_1mm = self.reslice(t1w_image)

            t1w_ss, t1w_mask = self.robex(t1w_image_1mm)
            t1w_image_n4ss = self.n4(t1w_ss, t1w_mask)

            t1w_image_data = SimpleITK.GetArrayFromImage(t1w_image_n4ss)

        else:
            t1w_image_data = SimpleITK.GetArrayFromImage(t1w_image)

        img = t1w_image_data[None]

        img = monai.transforms.NormalizeIntensity(nonzero=True,channel_wise=True)(img)
        orig_shape = img.shape[1:]
        bbox = monai.transforms.utils.generate_spatial_bounding_box(img, channel_indices=-1)
        img = monai.transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(img)
        meta = np.vstack([bbox, orig_shape, img.shape[1:]])
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img[0,...,100])
        # plt.show()

        pred = []
        img = monai.transforms.ToTensor(dtype=torch.float32, device=self.device)(img)
        img = img.permute(0,2,3,1)[None]
        with torch.no_grad():
            for m in list(self.models):
                pred.append(softmax(m._forward(img).squeeze(0).cpu().detach().numpy(), axis=0))
        pred = np.mean(np.stack(pred, axis=0), axis=0)

        # img_crf = img[0].cpu().detach().numpy()
        # img_crf = img_crf - img_crf.min()
        # img_crf = 255 * (img_crf / img_crf.max())
        # img_crf[img_crf < 0] = 0
        # img_crf[img_crf > 255] = 255
        # img_crf = np.asarray(img_crf, np.uint8)
        # pred_crf = np.asarray(pred, np.float32)
        # pred = self.crf(img_crf, pred_crf)

        pred = np.transpose(pred, [0,3,1,2])

        min_d, max_d = meta[0,0], meta[1,0]
        min_h, max_h = meta[0,1], meta[1,1]
        min_w, max_w = meta[0,2], meta[1,2]

        n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]

        if not all(cropped_shape == pred.shape[1:]):
            resized_pred = np.zeros((n_class, *cropped_shape))
            for i in range(n_class):
                resized_pred[i] = resize(
                    pred[i], cropped_shape, order=3, mode='edge', cval=0, clip=True, anti_aliasing=False
                )
            pred = resized_pred
        final_pred = np.zeros((n_class, *original_shape))
        final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred

        prediction = final_pred[1].astype(np.float32)

        prediction = SimpleITK.GetImageFromArray(prediction)
        if self.preprocessed:
            prediction.SetOrigin(t1w_image.GetOrigin()), prediction.SetSpacing(t1w_image.GetSpacing()), prediction.SetDirection(t1w_image.GetDirection())
        else:
            prediction.SetOrigin(t1w_image_n4ss.GetOrigin()), prediction.SetSpacing(t1w_image_n4ss.GetSpacing()), prediction.SetDirection(t1w_image_n4ss.GetDirection())

            prediction = self.reslice(prediction, reference=t1w_image)

        prediction = SimpleITK.GetArrayFromImage(prediction)

        prediction = (prediction > 0.5)

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return prediction.astype(int)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['t1w_image'].GetOrigin(),\
                                     input_data['t1w_image'].GetSpacing(),\
                                     input_data['t1w_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename.replace('T1w', 'label-L_mask')
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name))],
                           "inputs": [dict(type="Image", slug="t1w-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()


    def get_all_cases(self):
        t1w_image_paths = list(glob.glob(str(\
            self._input_path \
                / 'R*' / 'sub-r*s*' / 'ses-*' / 'anat' /\
                    'sub-r*s*_ses-*_space-MNI152NLin2009aSym_T1w.nii.gz')))
        t1w_image_paths = [Path(p) for p in t1w_image_paths]

        return t1w_image_paths


    def load_isles_case(self, t1w_image_path):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        preproc_path = str(t1w_image_path).split('/')
        preproc_path[-1] = 'n4_stripped.nii.gz'
        preproc_path = '/'.join(preproc_path)
        if os.path.exists(preproc_path):
            input_data = {'t1w_image': SimpleITK.ReadImage(str(preproc_path))}
            self.preprocessed = True
        else:
            input_data = {'t1w_image': SimpleITK.ReadImage(str(t1w_image_path))}
            self.preprocessed = False

        # Set input information.
        input_filename = str(t1w_image_path).split('/')[-1]
        return input_data, input_filename

    def get_file_path(self, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            file_list = list((self._input_path / "images" / slug).glob("*.mha"))
        elif filetype == 'json':
            file_list = list(self._input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results, f)

    def process(self):
        t1w_image_paths = self.get_all_cases()
        for t1w_image_path in tqdm(t1w_image_paths, total=len(t1w_image_paths)):
            input_data, input_filename = self.load_isles_case(t1w_image_path)
            self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    ploras().process()
