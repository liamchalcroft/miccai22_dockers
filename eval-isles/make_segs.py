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
from types import SimpleNamespace
import glob
from tqdm import tqdm
from scipy.special import softmax

"""ONLY FOR EVALUATION STAGE - WANT TO MAKE SURE DOCKER FOLLOWS SAME ALGO"""



# todo change with your team-name
class ploras():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = True  # False for running the docker!
        if self.debug:
            self._input_path = Path('/home/lchalcroft/Data/ISLES/2022/')
            self._output_path = Path('/home/lchalcroft/mdunet/isles-eval/raw_segs/')
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

        tta = False

        args = SimpleNamespace(exec_mode='train', data='/data', 
                                results='/results', config='config/config.pkl', logname='ploras', 
                                task='15', gpus=1, nodes=1, learning_rate=0.0002, gradient_clip_val=1.0, negative_slope=0.01, 
                                tta=tta, tb_logs=False, wandb_logs=True, wandb_project='isles', brats=False, deep_supervision=True, 
                                more_chn=False, invert_resampled_y=False, amp=True, benchmark=False, focal=False, save_ckpt=False, 
                                nfolds=5, seed=1, skip_first_n_eval=500, val_epochs=10, ckpt_path=None, 
                                ckpt_store_dir='../docker-isles/checkpoints/', fold=0, patience=100, 
                                batch_size=4, val_batch_size=4, momentum=0.99, weight_decay=0.0001, save_preds=False, dim=3, 
                                resume_training=False, num_workers=8, epochs=2000, warmup=5, norm='instance', nvol=4, depth=5, 
                                min_fmap=4, deep_supr_num=2, res_block=False, filters=None, num_units=2, md_encoder=True, 
                                md_decoder=False, shape=False, paste=0, data2d_dim=3, oversampling=0.4, overlap=0.5, 
                                affinity='unique_contiguous', scheduler=False, optimizer='adam', blend='gaussian', 
                                train_batches=0, test_batches=0)

        self.model_paths = [
            '../docker-isles/checkpoints/0/best.ckpt', '../docker-isles/checkpoints/1/best.ckpt', 
            '../docker-isles/checkpoints/2/best.ckpt', '../docker-isles/checkpoints/3/best.ckpt', 
            '../docker-isles/checkpoints/4/best.ckpt'
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


    def predict(self, input_data):
        """
        Input   input_data, dict.
                The dictionary contains 3 images and 3 json files.
                keys:  't1w_image' , 't1w_json'

        Output  prediction, array.
                Binary mask encoding the lesion segmentation (0 background, 1 foreground).
        """
        # Get all image inputs.
        dwi_image, adc_image, flair_image = input_data['dwi_image'],\
                                            input_data['adc_image'],\
                                            input_data['flair_image']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################

        dwi_image_rs = self.reslice(dwi_image, reference=flair_image)
        adc_image_rs = self.reslice(adc_image, reference=flair_image)
        flair_image_rs = self.reslice(flair_image, reference=flair_image)
        dwi_image_1mm = self.reslice(dwi_image_rs)
        adc_image_1mm = self.reslice(adc_image_rs)
        flair_image_1mm = self.reslice(flair_image_rs)

        dwi_image_data = SimpleITK.GetArrayFromImage(dwi_image_1mm)
        adc_image_data = SimpleITK.GetArrayFromImage(adc_image_1mm)
        flair_image_data = SimpleITK.GetArrayFromImage(flair_image_1mm)

        img = np.stack([adc_image_data, dwi_image_data, flair_image_data])
        img = np.transpose(img, (0,3,2,1))

        img = monai.transforms.NormalizeIntensity(nonzero=True,channel_wise=True)(img).astype(np.float32)
        orig_shape = img.shape[1:]
        bbox = monai.transforms.utils.generate_spatial_bounding_box(img, channel_indices=-1)
        img = monai.transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(img)
        meta = np.vstack([bbox, orig_shape, img.shape[1:]])

        pred = []
        img = monai.transforms.ToTensor(dtype=torch.float32, device=self.device)(img)
        img = img[None]
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

        min_d, max_d = meta[0,0], meta[1,0]
        min_h, max_h = meta[0,1], meta[1,1]
        min_w, max_w = meta[0,2], meta[1,2]

        n_class, original_shape, cropped_shape = pred.shape[0], meta[2], meta[3]

        final_pred = np.zeros((n_class, *original_shape))
        final_pred[:, min_d:max_d, min_h:max_h, min_w:max_w] = pred

        final_pred = np.transpose(final_pred, (0,3,2,1))
        prediction = final_pred[1]

        prediction = SimpleITK.GetImageFromArray(prediction)
        prediction.SetOrigin(dwi_image_1mm.GetOrigin()), prediction.SetSpacing(dwi_image_1mm.GetSpacing()), prediction.SetDirection(dwi_image_1mm.GetDirection())

        prediction = self.reslice(prediction, reference=dwi_image_rs)
        prediction = self.reslice(prediction, reference=dwi_image)

        prediction = SimpleITK.GetArrayFromImage(prediction)

        prediction[prediction > 1] = 0

        # prediction = (prediction > 0.5)

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        # return prediction.astype(int)
        return prediction

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['dwi_image'].GetOrigin(),\
                                     input_data['dwi_image'].GetSpacing(),\
                                     input_data['dwi_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        # Write segmentation to output location.
        if not self._algorithm_output_path.exists():
            os.makedirs(str(self._algorithm_output_path))
        output_image_path = self._algorithm_output_path / input_filename
        SimpleITK.WriteImage(output_image, str(output_image_path))

        # Write segmentation file to json.
        if output_image_path.exists():
            json_result = {"outputs": [dict(type="Image", slug="stroke-lesion-segmentation",
                                                 filename=str(output_image_path.name))],
                           "inputs": [dict(type="Image", slug="dwi-brain-mri",
                                           filename=input_filename)]}

            self._case_results.append(json_result)
            self.save()


    def get_all_cases(self):
        dwi_image_paths = list(glob.glob(str(\
            self._input_path \
                / '*' / 'rawdata' / 'sub-strokecase*' / 'ses-*' /\
                    'sub-strokecase*_ses-*_dwi.nii.gz')))
        image_paths = [[
            Path(p), Path(p.replace('dwi', 'adc')), Path(p.replace('dwi', 'flair'))
            ] for p in dwi_image_paths
            ]

        return image_paths


    def load_isles_case(self, img_paths):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        dwi_image_path, adc_image_path, flair_image_path = img_paths

        input_data = {'dwi_image': SimpleITK.ReadImage(str(dwi_image_path)), 
                      'adc_image': SimpleITK.ReadImage(str(adc_image_path)), 
                      'flair_image': SimpleITK.ReadImage(str(flair_image_path))}

        # Set input information.
        input_filename = str(dwi_image_path).split('/')[-1]
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
        image_paths = self.get_all_cases()
        for image_path in tqdm(image_paths, total=len(image_paths)):
            input_data, input_filename = self.load_isles_case(image_path)
            self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    ploras().process()
