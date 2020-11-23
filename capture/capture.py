import time

import cv2
from threading import Thread


class VideoCapture:
    def __init__(self, src):
        self.frame = None
        self.src = src
        self.capture = cv2.VideoCapture(src)
        self.thread = Thread(target=self._read_frames)

    def __iter__(self):
        self.start()
        time.sleep(3)
        return self

    def __next__(self):
        frame = self.get_frame()

        if frame is None:
            raise StopIteration

        return frame

    def _read_frames(self):
        while 1:
            ret, frame = self.capture.read()

            if not ret:
                break

            self.frame = frame[:, :, ::-1]

    def start(self):
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.src)

        self.thread.start()

    def stop(self):
        self.capture.release()

    def get_frame(self):
        if not self.capture.isOpened():
            return None
        return self.frame.copy()
