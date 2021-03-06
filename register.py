import os
from argparse import ArgumentParser

import cv2
import numpy

from capture.capture import VideoCapture
from face_detector import FaceDetection
from face_embedding import FaceEmbedding
from util import cosine_similarity, draw_square, show_image

RED = (255, 0, 0)
GREEN = (0, 255, 0)


def main():
    args = ArgumentParser()
    args.add_argument('-c', '--camera_url', default=0, type=str, help='0 - local camera')
    args.add_argument('-dt', '--detect_threshold', default=0.975, type=float, help="Threshold of face detection")
    args.add_argument('-rf', '--register_threshold', default=0.8, type=float, help="Threshold of face recognition")
    args.add_argument('--device', default='cuda:0', type=str, help="Device run model. `cuda:<id>` or `cpu`")
    args.add_argument('--detect_face_model', default='data/pretrained/mobilenet_header.pth',
                      type=str, help="Face detector model path")
    args.add_argument('--detect_face_backbone', default='data/pretrained/mobile_backbone.tar',
                      type=str, help="Face detector backbone path")
    args.add_argument('--recognized_model', default='data/pretrained/embedder_resnet50_asia.pth'
                      , type=str, help="Face embedding model path")
    args.add_argument('--model_registered', default='model_faces.npy', type=str, help="Model contain face's vectors")
    args.add_argument('--model_ids', default='model_face_ids.npy', type=str, help="Model contain face's ids")
    args.add_argument('--register_name', required=True, type=str, help="(Required) Register's name!")
    args = args.parse_args()

    try:
        args.camera_url = int(args.camera_url)
    except:
        pass

    if not (os.path.isfile(args.model_registered) and os.path.isfile(args.model_ids)):
        face_model = numpy.zeros((0, 512), dtype=numpy.float32)
        ids_model = []
    else:
        face_model = numpy.load(args.model_registered, allow_pickle=True)
        ids_model = numpy.load(args.model_ids, allow_pickle=True).tolist()

    detector = FaceDetection(args.detect_face_model, args.detect_face_backbone, scale_size=480, device=args.device)
    embedder = FaceEmbedding(args.recognized_model, device=args.device)

    # recognize
    video = VideoCapture(args.camera_url)

    for frame in video:
        faces = detector(frame)
        faces = embedder(faces)

        if len(faces):
            face = faces[0]
            txt = args.register_name
            color = GREEN

            score = -1
            if numpy.any(face_model):
                scores = cosine_similarity(face.embedding.reshape(1, 512), face_model, skip_normalize=True)
                score = numpy.max(scores)

            if score < args.register_threshold:
                ids_model.append(txt)
                face_model = face_model.tolist()
                face_model.append(face.embedding)
                face_model = numpy.array(face_model, dtype=numpy.float32)
                print(f"Record: {txt}")

            frame = draw_square(frame, face.box.astype(numpy.int), color=color)
            frame = cv2.putText(frame, f"EID: {txt}",
                                (int(face.box[0]), int(face.box[1] - 20)), cv2.FONT_HERSHEY_PLAIN, 1,
                                GREEN)

        if not show_image(frame, 'Face Recognition'):
            break

    video.stop()
    numpy.save(args.model_registered, face_model)
    numpy.save(args.model_ids, ids_model)


if __name__ == '__main__':
    main()
