from face import Face
from face_embedding.embedder import FaceEmbedder


class FaceEmbedding(FaceEmbedder):
    def get_features(self, faces):
        """

        Parameters
        ----------
        faces: list[Face]
            Detected faces.
        """
        if not faces:
            return []

        cropped_faces = [face.cropped_image for face in faces]
        face_features = super().get_features(cropped_faces)

        for face, feature in zip(faces, face_features):
            face.embedding = feature

        return faces

    __call__ = get_features


def main():
    import cv2
    import numpy
    from face_detector import FaceDetection
    from configuration.environments import env
    from dpsutil.media.video import VideoCapture
    from dpsutil.media.image import draw_square, draw_text
    from dpsutil.vector.distance import cosine_similarity
    from dpsutil.media.tool import show_image
    from dpsutil.media.constant import FHD_RESOLUTION
    detector = FaceDetection(env.detector_model_path, env.detector_backbone_path, env.detector_backbone_type,
                             env.face_detect_threshold, scale_size=720, device=env.device)
    embed = FaceEmbedding(env.embedder_model_path, backbone=env.embedder_backbone, device=env.device)
    capture = VideoCapture(0)
    base_emb = None
    distance = None

    with capture.read() as reader:
        for idx, frame in enumerate(reader):
            if idx in range(100):
                continue

            # get largest face.
            frame = frame.decode()
            faces = detector(frame)[:1]
            if faces.__len__() > 0:
                embed(faces)

                if base_emb is not None:
                    distance = cosine_similarity(base_emb, faces[0].embedder)
                else:
                    # recorded first face and use it to compare
                    base_emb = faces[0].embedder

                frame = draw_text(frame, f"L2: {distance}", (10, 30))
                frame = draw_square(frame, faces[0].box)

            if not show_image(frame, 'a', FHD_RESOLUTION):
                break


if __name__ == '__main__':
    main()
