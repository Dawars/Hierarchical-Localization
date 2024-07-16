'''
Extracting DINOv2 features for retrieval.
Source: https://www.kaggle.com/code/asarvazyan/imc-understanding-the-baseline?scriptVersionId=169973704&cellId=9
'''

import torch
import torchvision.transforms as tvf
import torch.nn.functional as F
import kornia as K

from transformers import AutoImageProcessor, AutoModel

from ..utils.base_model import BaseModel


class DinoV2(BaseModel):
    default_conf = {
        'variant': 'facebook/dinov2-base',
    }
    required_inputs = ['image']

    def _init(self, conf):
        model_name = conf["variant"]
        self.device = K.utils.get_cuda_device_if_available(0)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    def _forward(self, data):
        image = data['image']

        with torch.inference_mode():
            inputs = self.processor(images=image.clip(0.0, 1.0), return_tensors="pt", do_rescale=False).to(self.device)
            outputs = self.model(**inputs)  # last_hidden_state and pooled

            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            desc = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=-1, p=2)
        return {
            'global_descriptor': desc,
        }
