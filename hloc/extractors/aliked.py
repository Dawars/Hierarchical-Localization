from lightglue import ALIKED as ALIKED_

from ..utils.base_model import BaseModel


class ALIKED(BaseModel):
    default_conf = {
        "model_name": "aliked-n16",
        "max_num_keypoints": -1,
        "detection_threshold": 0.2,
        "nms_radius": 2,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        self.model = ALIKED_(max_num_keypoints=self.conf["max_keypoints"]).eval().cuda()

    def _forward(self, data):
        features = self.model(data)
        return {
            "keypoints": [f.keypoints for f in features],
            "keypoint_scores": [f.detection_scores for f in features],
            "descriptors": [f.descriptors.t() for f in features],
        }
