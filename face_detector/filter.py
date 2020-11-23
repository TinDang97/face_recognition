from dpsutil.vector.distance import l2_distance

from face import Face

MAX_DEEP_FACE_ANGLE = 30


def frontal_face(face: Face, face_angle_threshold=15):
    """

    bool = percent_pixel_in_cropped_image > threshold_percent

    To convert degree to percent that be calculated by below formula.
    threshold_percent = max_value_output - (angle_input/max_angle) * adjust_range_output -> output value in [0.25, 0.35]
                                            -----------
                             the closer max_angle angle_input is, the smaller value x is.

    """
    lm_aligned = face.cropped_land_mark
    return l2_distance(*lm_aligned[:2]) / 112 > (0.35 - face_angle_threshold/MAX_DEEP_FACE_ANGLE * 0.1)


def large_face(face: Face, min_size_threshold):
    box = face.box
    return l2_distance(box[:2], box[2:]) > min_size_threshold


class FaceFilter(object):
    def __init__(self, face_angle_threshold=25, small_face_threshold=56, custom_filters=None):
        if custom_filters is None:
            custom_filters = []

        self.filters = [
            lambda face: frontal_face(face, face_angle_threshold),
            lambda face: large_face(face, small_face_threshold),
            *custom_filters
        ]

    def filter(self, faces):
        """
        Filter faces in image

        Parameters
        ---------
        faces: list[Face]
        """
        faces_filtered = []
        for face in faces:
            keep = True
            for filter_func in self.filters:
                keep *= filter_func(face)
                if not keep:
                    break

            if not keep:
                continue
            faces_filtered.append(face)
        return faces_filtered

    __call__ = filter


