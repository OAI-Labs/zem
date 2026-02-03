#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import base64
import datetime
import io
import json
import os
import pickle
import time
import uuid
import logging
import copy
from enum import Enum, IntEnum

from . import file_utils

def read_config(conf_name=None):
    # Simplified: return empty config as we don't use RagFlow conf files
    return {}

CONFIGS = read_config()

def get_base_config(key, default=None):
    if key is None:
        return None
    return CONFIGS.get(key, default)

class BaseType:
    def to_dict(self):
        return dict([(k.lstrip("_"), v) for k, v in self.__dict__.items()])

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.date):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, datetime.timedelta):
            return str(obj)
        elif issubclass(type(obj), Enum) or issubclass(type(obj), IntEnum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif issubclass(type(obj), BaseType):
            return obj.to_dict()
        else:
            return json.JSONEncoder.default(self, obj)

def json_dumps(src, byte=False, indent=None):
    dest = json.dumps(src, indent=indent, cls=CustomJSONEncoder)
    if byte:
        dest = dest.encode(encoding="utf-8")
    return dest

def json_loads(src):
    if isinstance(src, bytes):
        src = src.decode(encoding="utf-8")
    return json.loads(src)

def current_timestamp():
    return int(time.time() * 1000)

def get_uuid():
    return uuid.uuid1().hex

def datetime_format(date_time: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(date_time.year, date_time.month, date_time.day,
                             date_time.hour, date_time.minute, date_time.second)
