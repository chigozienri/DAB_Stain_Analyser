# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import tempfile
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import numpy.ma as ma
from cog import BasePredictor, Input, Path

from DAB_Analysis_Functions import DAB


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        image: Path = Input(description="input image"),
        asyn_LMean: float = Input(default=38.35),
        asyn_aMean: float = Input(default=27.75),
        asyn_bMean: float = Input(default=24.9),
        asyn_thres: float = Input(default=15),
        cell_LMean: float = Input(default=75.4),
        cell_aMean: float = Input(default=5.5),
        cell_bMean: float = Input(default=-3.4),
        cell_thres: float = Input(default=6),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        tempdir = tempfile.mkdtemp()
        filename = os.path.basename(image)
        in_path = os.path.join(tempdir, filename)
        if os.path.exists(in_path):
            os.remove(in_path)
        shutil.copy(image, in_path)

        DAB_A = DAB()

        img = cv2.imread(in_path)
        RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab_Image = cv2.cvtColor(DAB_A.im2double(RGBimage), cv2.COLOR_RGB2LAB)
        default_asyn_params = np.array([asyn_LMean, asyn_aMean, asyn_bMean, asyn_thres])
        default_cell_params = np.array([cell_LMean, cell_aMean, cell_bMean, cell_thres])

        image_mask_asyn, _ = DAB_A.colourFilterLab(lab_Image, default_asyn_params)
        image_mask_nuclei, _ = DAB_A.colourFilterLab(
            lab_Image, default_cell_params, rate=[1, 2]
        )

        fig = plt.figure(frameon=False)
        dpi = 100
        fig.set_size_inches(RGBimage.shape[1] / dpi, RGBimage.shape[0] / dpi)
        fig.set_dpi(dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.add_axes(ax)
        ax.imshow(RGBimage, aspect="auto")  # I would add interpolation='none'
        ax.imshow(
            image_mask_asyn,
            cmap="Blues",
            alpha=0.8 * (image_mask_asyn > 0),
            aspect="auto",
        )  # interpolation='none'
        ax.imshow(
            image_mask_nuclei,
            cmap="Reds",
            alpha=0.8 * (image_mask_nuclei > 0),
            aspect="auto",
        )  # interpolation='none'
        combined_out_path = os.path.join(tempdir, "combined_output.png")
        fig.savefig(combined_out_path)

        asyn_out_path = os.path.join(tempdir, "asyn_output.png")
        cv2.imwrite(asyn_out_path, image_mask_asyn.astype(np.uint8) * 255)

        nuclei_out_path = os.path.join(tempdir, "nuclei_output.png")
        cv2.imwrite(nuclei_out_path, image_mask_nuclei.astype(np.uint8) * 255)

        return [Path(combined_out_path), Path(asyn_out_path), Path(nuclei_out_path)]