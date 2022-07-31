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
import subprocess
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
from scipy.special import softmax

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
class PLORAS():
    def __init__(self,
                 input_path: Path = DEFAULT_INPUT_PATH,
                 output_path: Path = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH):

        self.debug = True  # False for running the docker!
        if self.debug:
            self._input_path = Path('/home/lchalcroft/mdunet/miccai22_dockers/docker-atlas/test/')
            self._output_path = Path('/home/lchalcroft/mdunet/miccai22_dockers/docker-atlas/output/')
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

        kwargs = {
            'data':None, 'dim':3, 'learning_rate':1e-9, 
            'brats':False, 'paste':0, 'focal':False, 
            'shape':False, 'exec_mode':'test', 'benchmark':False, 
            'filters':None, 'md_encoder':True, 'task':16, 
            'min_fmap':4, 'tta':True, 'deep_supervision':True, 
            'config':'config/config.pkl', 'depth':5, 'deep_supr_num':2,
            'res_block':False, 'num_units':2, 'md_decoder':False,
            'val_batch_size':1, 'overlap':0.5, 'blend':'gaussian',
            'training':False
            }

        parser = argparse.ArgumentParser()
        for k, v in kwargs.items():
            parser.add_argument('--' + k, default=v)
        args = parser.parse_args()

        self.model_paths = [
            'checkpoints/0/best.ckpt', 'checkpoints/1/best.ckpt', 
            'checkpoints/2/best.ckpt', 'checkpoints/3/best.ckpt', 
            'checkpoints/4/best.ckpt'
            ]
        self.models = [NNUnet(args).to(self.device) for _ in self.model_paths]
        self.models = [model.load_from_checkpoint(path) for model,path in zip(self.models, self.model_paths)]
        for model in self.models:
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

        # Get all json inputs.
        t1w_json = input_data['t1w_json']

        ################################################################################################################
        #################################### Beginning of your prediction method. ######################################

        t1w_image_1mm = self.reslice(t1w_image)

        t1w_ss, t1w_mask = self.robex(t1w_image_1mm)
        t1w_image_n4ss = self.n4(t1w_ss, t1w_mask)

        t1w_image_data = SimpleITK.GetArrayFromImage(t1w_image_n4ss)
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

        img_crf = img[0].cpu().detach().numpy()
        img_crf = img_crf - img_crf.min()
        img_crf = 255 * (img_crf / img_crf.max())
        img_crf[img_crf < 0] = 0
        img_crf[img_crf > 255] = 255
        img_crf = np.asarray(img_crf, np.uint8)
        pred_crf = np.asarray(pred, np.float32)
        pred = self.crf(img_crf, pred_crf)

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


    def load_isles_case(self):
        """ Loads the 6 inputs of ISLES22 (3 MR images, 3 metadata json files accompanying each MR modality).
        Note: Cases missing the metadata will still have a json file, though their fields will be empty. """

        # Get MR data paths.
        t1w_image_path = self.get_file_path(slug='t1w-brain-mri', filetype='image')

        # Get MR metadata paths.
        t1w_json_path = self.get_file_path(slug='t1w-mri-acquisition-parameters', filetype='json')

        input_data = {'t1w_image': SimpleITK.ReadImage(str(t1w_image_path)), 
                    't1w_json': json.load(open(t1w_json_path)),}

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
        input_data, input_filename = self.load_isles_case()
        self.process_isles_case(input_data, input_filename)


if __name__ == "__main__":
    # todo change with your team-name
    PLORAS().process()
