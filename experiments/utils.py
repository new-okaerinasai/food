import json

class Config:
    def __init__(self, fpath):
        with open(fpath) as f:
            self.__dict__ = json.load(f)

