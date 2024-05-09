import torch

from ..utils.base_model import BaseModel


class XFeat(BaseModel):
    default_conf = {
        "weights": "depth",
        "max_keypoints": 4096,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=conf["max_keypoints"])

    def _forward(self, data):
        image = data["image"]
        features = self.xfeat.detectAndCompute(image, top_k=None)

        features_ = {
            "keypoints": [f["keypoints"] for f in features],
            "keypoint_scores": [f["scores"] for f in features],
            "descriptors": [f["descriptors"].t() for f in features],
        }
        return features_
