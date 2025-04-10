import os
import time

import numpy as np
import pandas as pd

import audb
import audiofile
import opensmile


db = audb.load(
    "emodb",
    version="1.1.1",
    format="wav",
    mixdown=True,
    sampling_rate=16000,
    media="wav/03a01.*",  # load subset
    full_path=False,
    verbose=False,
)