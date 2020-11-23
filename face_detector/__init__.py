from dpsutil.media.image import resize


__all__ = ['FaceDetection']


class FaceDetection(object):
    """
    FaceDetection merging FaceDetector -> FaceAligner -> FaceFilter -> faces

    Parameters
    ----------
    backbone_path: str
        Path of backbone pre-trained model

    backbone_type: str
        (Default: BACKBONE_MBV1) BACKBONE_RESNET or BACKBONE_MBV1

    detect_threshold: float
        Threshold of confidence score of detector

    face_size: int
        Face's output square.

    face_angle_threshold: int
        Max deep angle face of FaceFilter

    small_face_threshold: int
        Min face's size of FaceFilter

    scale_size: int
        Scale size input image, which used to detect face.

    device: str
        (Default: cpu) CPU or GPU ('cuda[:gpu_id]')

    """

    def __init__(self, model_path, backbone_path, backbone_type='mobilenet',
                 detect_threshold=0.975, face_size=112, face_angle_threshold=15, small_face_threshold=56,
                 scale_size=480, device='cpu'):
        from face_detector.detector import FaceDetector
        from face_detector.filter import FaceFilter
        from dpsutil.cv.face_align import FaceAligner

        # prepare face detector
        self.face_detector = FaceDetector(backbone_path, backbone_type=backbone_type, device=device)
        self.face_detector.load_model(model_path)

        # prepare face aligner
        self.face_aligner = FaceAligner(output_size=face_size)

        # prepare face aligner
        self.face_filter = FaceFilter(face_angle_threshold, small_face_threshold)

        self.face_size = face_size
        self.scale_size = scale_size
        self.detect_threshold = detect_threshold

    def process(self, image):
        # scaling source -> speed up face detect process
        origin_size = image.shape[:2][::-1]
        scale = origin_size[1] / self.scale_size
        image_scaled = image

        # resize to scale size
        if scale != 1.:
            image_scaled = resize(image, (-1, self.scale_size), keep_ratio=True)

        detected_faces = self.face_detector(image_scaled, self.detect_threshold)

        for face in detected_faces:
            # scale
            box = face.box * scale
            land_mark = face.land_mark * scale

            # face align
            face_aligned, lm_aligned = self.face_aligner(image, land_mark)

            # overwrite data
            face.box = box
            face.land_mark = land_mark
            face.cropped_image = face_aligned
            face.cropped_land_mark = lm_aligned

        filtered_faces = self.face_filter(detected_faces)
        return filtered_faces

    __call__ = process


# test model
def main():
    import cv2

    from configuration.environments import env
    from dpsutil.media.video import VideoCapture
    from dpsutil.media.image import draw_square
    from dpsutil.media.tool import show_image
    from dpsutil.media.constant import FHD_RESOLUTION

    detector = FaceDetection(env.detector_model_path, env.detector_backbone_path, env.detector_backbone_type,
                             env.face_detect_threshold, scale_size=240, device=env.device)
    capture = VideoCapture(0)
    reader = capture.read()

    for frame in reader:
        frame = frame.decode()

        for idx, face in enumerate(detector(frame)[:1]):
            frame = draw_square(frame, face.box)

            for pos in face.land_mark:
                cv2.circle(frame, tuple(pos), 1, (255, 255, 255))

            if not show_image(face.cropped_image, f"face {idx}"):
                break

        if not show_image(frame, "detector_debug", windows_size=FHD_RESOLUTION):
            break
    print(f"FPS: {reader.stop()}")


if __name__ == '__main__':
    main()
