# -*- coding = utf-8 -*-
# @time:2024/9/3 14:10
# Author:david yuan
# @File:handler_factory.py
# @Software:VeSync


# file: factory.py

class Factory:
    _instances = {}

    @classmethod
    def get_instance(cls, class_):
        if class_ not in cls._instances:
            cls._instances[class_] = class_()
        return cls._instances[class_]


