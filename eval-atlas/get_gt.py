import glob
import shutil
import os

files = glob.glob(
    "/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/output-makesegs/stroke-lesion-segmentation/*_mask.nii.gz"
)

for fil in files:
    fname = files.split("/")[-1][:-11]
    shutil.copyfile(
        glob.glob(
            os.path.join(
                "/home/lchalcroft/Data/ATLAS_R2.0/Training/**", fname + "*mask.nii.gz"
            )
        )[0],
        os.path.join(
            "/home/lchalcroft/mdunet/miccai22_dockers/eval-atlas/output-makesegs/stroke-lesion-segmentation/",
            fname="gt.nii.gz",
        ),
    )
