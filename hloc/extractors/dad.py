import dad
import torch

from ..utils.base_model import BaseModel

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


class DaD(BaseModel):
    default_conf = {
        "model_name": "dad",
        "max_keypoints": 8 * 1024,
        "detection_threshold": 0.2,
        "nms_radius": 3,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        conf.pop("name")
        self.model = dad.load_DaD()

    @staticmethod
    def imgnet_normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - imagenet_mean[:, None, None].to(x.device)) / (
            imagenet_std[:, None, None].to(x.device)
        )

    def _forward(self, data):
        B, C, H, W = data["image"].shape
        features = self.model.detect(
            {"image": self.imgnet_normalize(data["image"])},
            num_keypoints=self.conf["max_keypoints"],
            return_dense_probs=False)
        self.model.to_pixel_coords(features["keypoints"], H, W)
        return {
            "keypoints": [features["keypoints"][0]],
            "keypoint_scores": [features["keypoint_probs"][0]],
            # "descriptors": [f.t() for f in features["descriptors"]],
        }
