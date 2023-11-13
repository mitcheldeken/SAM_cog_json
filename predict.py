# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
sys.path.append("..")
import cv2
import numpy as np
import imutils
import json

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        device = "cuda"
        model_type = "vit_l"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
    
    def predict(
        self,
        image: Path = Input(description="Input image"),
        resize_width: int = Input(default=0, description="The width to resize the image to before running inference. Use 0 if you don't want resizing"),
        points_per_side: int = Input(default = 32, description= "The number of points to be sampled along one side of the image. The total number of points is points_per_side**2. If None, point_grids must provide explicit point sampling."),
        pred_iou_thresh: float = Input(default=0.88,description="A filtering threshold in [0,1], using the model's predicted mask quality."),
        stability_score_thresh: float = Input(default=0.95, description="A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions."),
        stability_score_offset: float = Input(default=1.0, description = "The amount to shift the cutoff when calculated the stability score."),
        box_nms_thresh: float = Input(default=0.7, description= "The box IoU cutoff used by non-maximal suppression to filter duplicate masks."),
        crop_n_layers: int = Input(default=0, description="If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops"),
        crop_nms_thresh: float = Input(default=0.7, description="The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops."),
        crop_overlap_ratio: float = Input(default=512 / 1500, description= "Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap."),
        crop_n_points_downscale_factor: int = Input(default=1, description= "The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n."),
        min_mask_region_area: int = Input(default=0, description="If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area."),
        output_mode: str = Input(default="uncompressed_rle", description="output format. Can be 'uncompressed_rle', 'coco_rle' or 'binary_mask'")
        ) -> Path:
        """Run a single prediction on the model"""
        args = locals()
        del args["self"]
        del args["image"]
        del args["resize_width"]

        mask_generator = SamAutomaticMaskGenerator(self.sam, **args)

        image = cv2.imread(str(image))
        if resize_width>0:
            image = imutils.resize(image, width=resize_width)
        masks = mask_generator.generate(image)
   
        with open('masks.json', 'w') as file:
            json.dump(masks, file)

        return Path('masks.json')