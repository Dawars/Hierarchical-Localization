import kornia

from ..utils.base_model import BaseModel


class DeDoDe(BaseModel):
    default_conf = {
        "detector": "L-C4-v2",
        "descriptor": "G-upright",
        "max_keypoints": 10_000,
        "pad_if_not_divisible": True,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.model = kornia.feature.DeDoDe.from_pretrained(detector_weights=conf["detector"],
                                                           descriptor_weights=conf["descriptor"])

    def _forward(self, data):
        image = data["image"]
        features = self.model(
            image,
            n=self.conf["max_keypoints"],
            pad_if_not_divisible=self.conf["pad_if_not_divisible"],
        )
        return {
            "keypoints": [f.keypoints for f in features],
            "keypoint_scores": [f.detection_scores for f in features],
            "descriptors": [f.descriptors.t() for f in features],
        }
