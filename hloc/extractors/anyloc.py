import torch
import torchvision.transforms as tvf
from torchvision.transforms import InterpolationMode

from ..utils.base_model import BaseModel


class EigenPlaces(BaseModel):
    default_conf = {
        "backbone": "DINOv2",
        "domain": "urban",  # indoor, urban, aerial, structured, unstructured, global,
    }
    required_inputs = ["image"]

    def _init(self, conf):

        self.net  = torch.hub.load("AnyLoc/DINO", "get_vlad_model",
                               domain="indoor", backbone=conf["backbone"]).eval()

        self.transform = tvf.Resize((224, 224), InterpolationMode.BICUBIC)

    def _forward(self, data):
        image = self.transform(data["image"])

        # Result: VLAD descriptors of shape [1, 49152]
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
