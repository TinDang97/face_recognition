import torch
import numpy
from dpsutil.log import write_info

from face import Face
from models.face_detector import py_cpu_nms, decode, decode_landm, PriorBox, RetinaNet, BACKBONE_RESNET, BACKBONE_MBV1
from dpsutil.vector.distance import euclidean_distance


def remove_prefix(state_dict, prefix):
    f = (lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x)
    return {f(key): value for key, value in state_dict.items()}


class FaceDetector(object):
    """
    FaceDetector implement RetinaFace
    Parameters
    ----------
    backbone_type: str
        (Default: BACKBONE_MBV1) BACKBONE_RESNET or BACKBONE_MBV1

    backbone_path: str
        Path of backbone pre-trained model

    device: str
        (Default: cpu) CPU or GPU ('cuda[:gpu_id]')
    """
    def __init__(self, backbone_path, backbone_type=BACKBONE_MBV1, device='cpu'):
        assert backbone_type in [BACKBONE_MBV1, BACKBONE_RESNET]

        self.model = RetinaNet(backbone_type, backbone_path)
        self.model_config = self.model.cfg
        self.priorbox = PriorBox(self.model_config)
        self.device = device

        self.image_size = None
        self.prior_data = None
        self.landmark_scale = None
        self.box_scale = None

    def load_model(self, pretrained_path):
        """
        Load pretrained model of RetinaFace

        Parameters
        ----------
        pretrained_path: str
            Path of backbone pre-trained model
        """
        write_info(f"Detector loaded: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path)

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')

        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def update_prior(self, image_size):
        if self.image_size == image_size:
            return

        priors = self.priorbox.forward(image_size, self.device)
        self.prior_data = priors.data
        self.image_size = image_size
        self.landmark_scale = torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0],
             image_size[1], image_size[0], image_size[1], image_size[0],
             image_size[1], image_size[0]],
            dtype=torch.float32
        ).to(self.device)
        self.box_scale = torch.tensor([image_size[1], image_size[0], image_size[1], image_size[0]], dtype=torch.float32).to(self.device)

    def decode(self, loc, conf, landms, threshold, top=500):
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.model_config['variance'])
        boxes = boxes * self.box_scale
        scores = conf.squeeze(0).data[:, 1]

        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.model_config['variance'])
        landms = landms * self.landmark_scale

        inds = scores > threshold

        # ignore low scores
        boxes = boxes[inds].cpu().detach().numpy()
        landms = landms[inds].cpu().detach().numpy()
        scores = scores[inds].cpu().detach().numpy()

        # empty gpu cached memory if model is working in GPU
        if "cuda" in self.device:
            torch.cuda.empty_cache()

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = numpy.hstack((boxes, scores[:, numpy.newaxis])).astype(numpy.float32, copy=False)
        keep = py_cpu_nms(dets, 0.3)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:top, :]
        landms = landms[:top, :]

        faces = numpy.concatenate((dets, landms), axis=1)

        # sort by box size
        if len(faces) > 1:
            order = numpy.argsort([euclidean_distance(face[:2], face[2:4]) for face in faces])[::-1]
            faces = faces[order]

        # remove box out of image's boundary
        faces = faces[[numpy.any((face[:4] > 0) & (face[:4] < 5000)) for face in faces]]
        return faces

    def detect(self, img, threshold, top=500):
        # update decoder.
        self.update_prior(img.shape[:2])

        # transform image
        transformed_img = img.astype(numpy.float32)
        transformed_img -= (104, 117, 123)
        transformed_img = transformed_img.transpose(2, 0, 1)  # reshape
        transformed_img = torch.from_numpy(transformed_img).unsqueeze(0).to(self.device)

        # forward
        loc_encoded, score_encoded, landms_encoded = self.model(transformed_img)
        faces_decoded = self.decode(loc_encoded, score_encoded, landms_encoded, threshold, top)

        # numpy -> Face
        faces = []
        for face in faces_decoded:
            box = face[:4].copy()
            land_mark = face[5:].reshape(-1, 2).copy()
            faces.append(Face(box, land_mark=land_mark))
        return faces

    __call__ = detect
