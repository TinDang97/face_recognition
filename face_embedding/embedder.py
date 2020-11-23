import torch

from dpsutil.cv.transform import Transform, resize, crop_center, flip, normalize
from dpsutil.log import write_info
from dpsutil.media.constant import FLIP_HORIZONTAL
from util import normalize_L2

from torchvision.transforms.functional import to_tensor

from face_embedding.insight_face import InsightFace, IR_50


class FaceEmbedder(object):
    def __init__(self, model_path, backbone=IR_50, face_size=(112, 112), device='cpu'):
        # Load embedding model
        self.embedder = InsightFace(backbone, face_size)
        self.embedder.load_model(model_path, device)

        write_info(f'Loaded face embedding from {model_path}')

        # normalize face
        self.normalize = Transform([
            (lambda x: x * 0.5 + 0.5),
            normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            to_tensor
        ])

        # Crop center
        self.crop_transform = Transform([
            resize((128, 128)),
            crop_center((112, 112))
        ])

        # hflip face
        self.hflip_transform = Transform([
            flip(FLIP_HORIZONTAL)
        ])
        self.device = device

    def get_features(self, faces):
        """
        Extract features of faces.
        :param faces: face list
        :return: feature list
        """
        faces_transformed = torch.zeros(len(faces) * 2, 3, 112, 112, dtype=torch.float32)
        for idx, face in enumerate(faces):
            # pre-progress
            face_cropped = self.crop_transform(face)
            face_mirror = self.hflip_transform(face_cropped)

            # normalize and to tensor
            face_cropped = self.normalize(face_cropped)
            face_mirror = self.normalize(face_mirror)

            # put twice together
            faces_transformed[idx * 2] = face_cropped
            faces_transformed[idx * 2 + 1] = face_mirror

        faces_transformed = faces_transformed.to(self.device)
        # get features
        features = self.embedder(faces_transformed)
        features = features.detach().cpu()

        # clear cache
        del faces_transformed
        torch.cuda.empty_cache()

        # +
        features = features.detach().numpy()
        features = features[:len(features):2] + features[1:len(features):2]

        # normalize all features
        features = normalize_L2(features)
        return features

    __call__ = get_features


__all__ = ['FaceEmbedder']
