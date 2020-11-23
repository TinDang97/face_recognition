import torch

from models.model_irse import IR_50, IR_SE_152, IR_SE_101, IR_SE_50, IR_152, IR_101

MODELS = {
    "IR_50": IR_50,
    "IR_SE_50": IR_SE_50,
    "IR_152": IR_152,
    "IR_SE_152": IR_SE_152,
    "IR_101": IR_101,
    "IR_SE_101": IR_SE_101
}

IR_50 = "IR_50"
IR_SE_50 = "IR_SE_50"
IR_152 = "IR_152"
IR_SE_152 = "IR_SE_152"
IR_101 = "IR_101"
IR_SE_101 = "IR_SE_101"


def remove_prefix(state_dict, prefix):
    f = (lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x)
    return {f(key): value for key, value in state_dict.items()}


class InsightFace(object):
    def __init__(self, backbone=IR_50, face_size=(112, 112)):
        self.backbone = MODELS[backbone](face_size)
        self.output_size = self.backbone.output_layer[-1].num_features

    def load_model(self, pretrained_path, device):
        assert pretrained_path, "Pre-trained model is not found!"

        pretrained_dict = torch.load(pretrained_path)

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')

        self.backbone.load_state_dict(pretrained_dict, strict=False)
        self.backbone.to(device)
        self.backbone.eval()

    def get_feature(self, face):
        assert isinstance(face, torch.Tensor)
        return self.backbone(face)

    __call__ = get_feature


__all__ = ['InsightFace', 'IR_50', 'IR_SE_50', 'IR_101', 'IR_SE_101', 'IR_152', 'IR_SE_152']
