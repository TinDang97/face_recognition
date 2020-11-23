class Face(object):
    """
    Store face information.
    Contain box, land_mark and face's image (in RGB format)
    """

    def __init__(self, box, land_mark=None, cropped_image=None, cropped_land_mark=None, embedding=None):
        self.id = None
        self.land_mark = land_mark
        self.box = box
        self.cropped_image = cropped_image
        self.cropped_land_mark = cropped_land_mark
        self.embedding = embedding
