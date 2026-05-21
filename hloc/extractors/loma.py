import torch
import torch.nn.functional as F
from loma.descriptor.dedode import DeDoDeDescriptor
from loma.detector.dad import DaD
from loma.device import device

from ..utils.base_model import BaseModel

"""
loma_B
loma_B128
# loma_Bturbo
# loma_Bturbov2
loma_G
loma_L
loma_R
"""

class LoMaExtractor(BaseModel):
    default_conf = {
        "max_keypoints": 2048,
        "compile": False,
        "model_name": "loma_B"
    }
    required_inputs = ["image"]

    def _init(self, conf):
        # DaD weights loaded by default
        self.detector = DaD(DaD.Cfg(compile=conf["compile"])).eval()

        descriptor_arch = "dedode_b" if conf['model_name'] == "loma_B128" else "dedode_g"
        # Descriptor weights need to be manually loaded
        self.descriptor = DeDoDeDescriptor(
            DeDoDeDescriptor.Cfg(compile=conf["compile"], arch=descriptor_arch)
        ).eval()
        weights = torch.hub.load_state_dict_from_url(
            f"https://github.com/davnords/storage/releases/download/loma/{conf['model_name']}.pt",
            map_location=device,
        )
        weights = {k: v for k, v in weights.items() if k.startswith("_descriptor.")}
        weights = {k[len("_descriptor.") :]: v for k, v in weights.items()}
        self.descriptor.load_state_dict(weights, strict=True)

    def preprocess_image(self, image, H=784, W=784):
        image = F.interpolate(
            image,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0]
        return image[None].to(device)

    def detect_and_describe(self, batch: dict[str, torch.Tensor]):
        H, W = batch["image"].shape[2:]

        detections = self.detector.detect(
            batch, num_keypoints=self.conf["max_keypoints"]
        )
        keypoints = detections["keypoints"]

        description = self.descriptor.describe_keypoints(
            self.preprocess_image(batch["image"]),
            keypoints,
        )
        keypoints = self.detector.to_pixel_coords(keypoints, H, W)
        keypoints = keypoints - 0.5  # be consistent with hloc
        keypoints[..., 0] = keypoints[..., 0].clamp(0.5, W - 1.5)
        keypoints[..., 1] = keypoints[..., 1].clamp(0.5, H - 1.5)
        return {
            "keypoints": [keypoints[0]],
            "descriptors": [description["descriptions"].transpose(-1, -2)[0]],
            "scores": [detections["keypoint_probs"][0]],
        }

    def _forward(self, data):
        return self.detect_and_describe(data)
