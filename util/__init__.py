import cv2
import numpy


def normalize_L2(x):
    assert isinstance(x, numpy.ndarray)
    return x / numpy.sqrt(numpy.sum((x ** 2), keepdims=True, axis=1))


def cosine_similarity(x1, x2, skip_normalize=False):
    if type(x1) is list:
        x1 = numpy.array(x1)

    if type(x2) is list:
        x2 = numpy.array(x2)

    assert type(x1) is numpy.ndarray or type(x2) is numpy.ndarray
    assert x1.shape[-1] == x2.shape[-1]
    assert len(x1.shape) == 2

    if not skip_normalize:
        x1 = normalize_L2(x1)
        x2 = normalize_L2(x2)
    return numpy.dot(x1, x2.T)


def draw_square(img, position, color=(0, 255, 0)) -> numpy.ndarray:
    """
    Draw text at position in image.
    - position: top-left, bottom-right of square
    - color: support tuple and hex_color
    :return:
    """
    if not isinstance(position, tuple):
        position = tuple(position)

    return cv2.rectangle(img, position[0:2], position[2:4], color, 2)


def show_image(img, windows_name, windows_size=(1280, 720), windows_mode=cv2.WINDOW_NORMAL, wait_time=1,
               key_press_exit="q"):
    """
    Show image in RGB format

    Parameters
    ----------
    img: numpy.ndarray
        image array

    windows_name: str
        Title of window

    windows_size: tuple[int, int]
        (Default: SD_RESOLUTION) size of window

    windows_mode: int
        (Default: cv2.WINDOW_NORMAL) Mode of window

    wait_time: int
        Block time. (-1: infinite)

    key_press_exit: str
        Key stop event.

    Returns
    -------
    bool
        True - Stop event from user
    """
    cv2.namedWindow(windows_name, windows_mode)
    cv2.imshow(windows_name, img[:, :, ::-1])
    cv2.resizeWindow(windows_name, *windows_size)

    if cv2.waitKey(wait_time) & 0xFF == ord(key_press_exit):
        cv2.destroyWindow(windows_name)
        return False
    return True
