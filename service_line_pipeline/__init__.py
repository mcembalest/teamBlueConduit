#!/usr/bin/env python3

import numpy as np
import logging

log = logging.getLogger(__name__)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# TODO Create project wide config singleton
# config = Config(**client_configs, **environment_configs)
