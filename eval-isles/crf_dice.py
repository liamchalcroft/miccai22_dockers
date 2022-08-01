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
import monai
from skimage.transform import resize
import argparse
import tempfile
import glob
import subprocess
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from tqdm import tqdm

# TODO: allow user input for config in order to use gridsearch or bayesian opt
# parser = argparse.ArgumentParser()
# for k, v in kwargs.items():
#     parser.add_argument('--' + k, default=v)
# args = parser.parse_args()

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
        seg_image = input_data['seg_image']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################

        t1w_image_data = SimpleITK.GetArrayFromImage(t1w_image)
        seg = SimpleITK.GetArrayFromImage(seg_image)[None]

        img = t1w_image_data[None]

        img = monai.transforms.NormalizeIntensity(nonzero=True,channel_wise=True)(img)
        orig_shape = img.shape[1:]
        bbox = monai.transforms.utils.generate_spatial_bounding_box(img, channel_indices=-1)
        img = monai.transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(img)
        seg = monai.transforms.SpatialCrop(roi_start=bbox[0], roi_end=bbox[1])(seg)[0]
        meta = np.vstack([bbox, orig_shape, img.shape[1:]])
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img[0,...,100])
        # plt.show()

        img_crf = img[0].cpu().detach().numpy()
        img_crf = img_crf - img_crf.min()
        img_crf = 255 * (img_crf / img_crf.max())
        img_crf[img_crf < 0] = 0
        img_crf[img_crf > 255] = 255
        img_crf = np.asarray(img_crf, np.uint8)
        pred_crf = np.asarray(seg, np.float32)
        pred = self.crf(img_crf, pred_crf)

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

        prediction = final_pred[1]

        prediction = (prediction > 0.5)

        #################################### End of your prediction method. ############################################
        ################################################################################################################

        return prediction.astype(np.uint8)

    def process_isles_case(self, input_data, input_filename):
        # Get origin, spacing and direction from the DWI image.
        origin, spacing, direction = input_data['t1w_image'].GetOrigin(),\
                                     input_data['t1w_image'].GetSpacing(),\
                                     input_data['t1w_image'].GetDirection()

        # Segment images.
        prediction = self.predict(input_data) # function you need to update!

        # Build the itk object.
        output_image = SimpleITK.GetImageFromArray(prediction)
#        output_image = SimpleITK.Cast(output_image, SimpleITK.sitkInt8)
        output_image.SetOrigin(origin), output_image.SetSpacing(spacing), output_image.SetDirection(direction)

        dice = DiceScore(prediction, ground_truth)

        return dice


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
        dice = []
        for t1w_image_path in tqdm(t1w_image_paths, total=len(t1w_image_paths)):
            input_data, input_filename = self.load_isles_case(t1w_image_path)
            dice.append(self.process_isles_case(input_data, input_filename))


if __name__ == "__main__":
    # todo change with your team-name
    ploras().process()
