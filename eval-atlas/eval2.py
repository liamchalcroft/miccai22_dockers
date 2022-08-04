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
from data_preprocessing.preprocessor import Preprocessor
import json
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from utils.utils import make_empty_dir, set_cuda_devices, set_granularity, verify_ckpt_path
from data_loading.data_module import DataModule
from copy import deepcopy
import shutil

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

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
            self._output_path = Path('/home/lchalcroft/mdunet/eval-atlas/output/')
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

        args = SimpleNamespace(exec_mode='predict', data='/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data/16_3d/test', 
                                results='/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/results', config='../docker-atlas/config/config.pkl', logname='ploras', 
                                task='16', gpus=1, nodes=1, learning_rate=0.0002, gradient_clip_val=1.0, negative_slope=0.01, 
                                tta=tta, tb_logs=False, wandb_logs=False, wandb_project='isles', brats=False, deep_supervision=True, 
                                more_chn=False, invert_resampled_y=False, amp=True, benchmark=False, focal=False, save_ckpt=False, 
                                nfolds=5, seed=1, skip_first_n_eval=500, val_epochs=10, ckpt_path=None, 
                                ckpt_store_dir='../docker-atlas/checkpoints/', fold=0, patience=100, 
                                batch_size=4, val_batch_size=4, momentum=0.99, weight_decay=0.0001, save_preds=True, dim=3, 
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
        self.args = []
        for i,pth in enumerate(self.model_paths):
            ckpt = torch.load(pth, map_location=self.device)
            ckpt['hyper_parameters']['args'] = deepcopy(args)
            ckpt['hyper_parameters']['args'].ckpt_store_dir = '../docker-atlas/checkpoints/' + str(i)
            ckpt['hyper_parameters']['args'].ckpt_path = '../docker-atlas/checkpoints/' + str(i) + '/best.ckpt'
            ckpt['hyper_parameters']['args'].fold = i
            ckpt['hyper_parameters']['args'].gpus = 1 if torch.cuda.is_available() else 0
            torch.save(ckpt, pth)
            self.args.append(ckpt['hyper_parameters']['args'])

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

    def nnunet_preprocess(self, image):
        os.makedirs('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data/ATLAS2022_ss/imagesTs/', exist_ok=True)
        SimpleITK.WriteImage(image, str('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data/ATLAS2022_ss/imagesTs/ATLAS2022_ss_0001.nii.gz'))
        data_desc = {
                    "description": "Stroke Lesion Segmentation",
                    "labels": {
                        "0": "Background",
                        "1": "Lesion"
                    },
                    "licence": "BLANK",
                    "modality": {
                        "0": "T1"
                    },
                    "name": "ATLAS2022_ss",
                    "numTest": 1,
                    "numTraining": 0,
                    "reference": "BLANK",
                    "release": "BLANK",
                    "tensorImageSize": "4D",
                    "test": [
        "/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data/ATLAS2022_ss/imagesTs/ATLAS2022_ss_0001.nii.gz",
                    ],
                    "training": []
        }
        with open('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data/ATLAS2022_ss/dataset.json', 'w') as f:
            json.dump(data_desc, f)
        args = SimpleNamespace(data='/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data', results='/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data', exec_mode='test',
                                ohe=False, verbose=False, task='16', dim=3, n_jobs=1)
        Preprocessor(args).run()

    def nnunet_infer(self, args):
        data_module = DataModule(args)
        data_module.setup()
        ckpt_path = verify_ckpt_path(args)
        model = NNUnet(args)
        callbacks = [RichProgressBar(), ModelSummary(max_depth=2)]
        logger = False
        trainer = Trainer(
            logger=logger,
            default_root_dir=args.results,
            benchmark=True,
            deterministic=False,
            max_epochs=args.epochs,
            precision=16 if args.amp else 32,
            gradient_clip_val=args.gradient_clip_val,
            enable_checkpointing=args.save_ckpt,
            # callbacks=callbacks,
            callbacks=None,
            num_sanity_val_steps=0,
            accelerator="gpu",
            devices=args.gpus,
            num_nodes=args.nodes,
            strategy="ddp" if args.gpus > 1 else None,
            limit_train_batches=1.0 if args.train_batches == 0 else args.train_batches,
            limit_val_batches=1.0 if args.test_batches == 0 else args.test_batches,
            limit_test_batches=1.0 if args.test_batches == 0 else args.test_batches,
            check_val_every_n_epoch=args.val_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        save_dir = os.path.join('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/prediction', str(args.fold))
        model.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        model.args = args
        trainer.test(model, dataloaders=data_module.test_dataloader(), ckpt_path=ckpt_path, verbose=False)

    def nnunet_ensemble(self, paths, ref):
        preds = [np.load(f) for f in paths]
        pred = np.mean(preds, 0)[1]
        print()
        print(pred.shape)
        pred = pred.transpose(2,1,0)
        print(pred.shape)
        pred_image = SimpleITK.GetImageFromArray(pred)
        pred_image.SetOrigin(ref.GetOrigin())
        pred_image.SetSpacing(ref.GetSpacing())
        pred_image.SetDirection(ref.GetDirection())
        return pred_image

    def setup(self):
        os.makedirs('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data', exist_ok=True)
        os.makedirs('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/results', exist_ok=True)
        os.makedirs('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/prediction', exist_ok=True)

    def cleanup(self):
        shutil.rmtree('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/data')
        shutil.rmtree('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/results')
        shutil.rmtree('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/prediction')

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

        self.setup()

        if not self.preprocessed:
            t1w_image_1mm = self.reslice(t1w_image)
            t1w_ss, t1w_mask = self.robex(t1w_image_1mm)
            t1w_image_n4ss = self.n4(t1w_ss, t1w_mask)
            self.nnunet_preprocess(t1w_image_n4ss)

        else:
            self.nnunet_preprocess(t1w_image)

        for args_ in self.args:
            self.nnunet_infer(args_)

        paths = [os.path.join('/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/prediction',str(i),'ATLAS2022_ss_0001.npy') for i in range(len(self.args))]
        prediction = self.nnunet_ensemble(paths, ref=t1w_image if self.preprocessed else t1w_image_n4ss)

        pred_crf = SimpleITK.GetArrayFromImage(prediction)
        # pred_crf = np.stack([1.-pred_crf, pred_crf])
        img_crf = SimpleITK.GetArrayFromImage(t1w_image if self.preprocessed else t1w_image_n4ss)
        print(pred_crf.shape, img_crf.shape)
        # img_crf = img_crf - img_crf.min()
        # img_crf = 255 * (img_crf / img_crf.max())
        # img_crf[img_crf < 0] = 0
        # img_crf[img_crf > 255] = 255
        # img_crf = np.asarray(img_crf, np.uint8)
        # pred_crf = np.asarray(pred_crf, np.float32)
        # prediction = self.crf(img_crf, pred_crf)
        # prediction = prediction[1]
        # prediction = SimpleITK.GetImageFromArray(prediction)

        if self.preprocessed:
            prediction.SetOrigin(t1w_image.GetOrigin()), prediction.SetSpacing(t1w_image.GetSpacing()), prediction.SetDirection(t1w_image.GetDirection())
        else:
            prediction.SetOrigin(t1w_image_n4ss.GetOrigin()), prediction.SetSpacing(t1w_image_n4ss.GetSpacing()), prediction.SetDirection(t1w_image_n4ss.GetDirection())

            prediction = self.reslice(prediction, reference=t1w_image)

        prediction = SimpleITK.GetArrayFromImage(prediction)

        prediction = (prediction > 0.5)

        self.cleanup()

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
