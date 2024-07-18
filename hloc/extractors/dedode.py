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
        keypoints, scores, descriptors = self.model(
            image,
            n=self.conf["max_keypoints"],
            pad_if_not_divisible=self.conf["pad_if_not_divisible"],
        )
        return {
            "keypoints": [f for f in keypoints],  # list[N,2]
            "keypoint_scores": [f for f in scores],  # list[N]
            "descriptors": [f.t() for f in descriptors]  # list[D,N]
        }
