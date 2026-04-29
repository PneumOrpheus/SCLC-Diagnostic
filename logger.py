# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
import datetime 
from termcolor import colored


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Create a timestamped filename. Include minutes + PID so two runs of the
    # same model started within one hour (or even one minute, on a fast kick-off)
    # do not collide into a single interleaved log. PID is the tiebreaker for
    # truly simultaneous launches; `mode='w'` keeps each file's contents
    # scoped to exactly one run so nothing can silently append to a prior log.
    now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_filename = f'{name}_{now_str}_pid{os.getpid()}_logs.txt'

    file_handler = logging.FileHandler(os.path.join(output_dir, log_filename), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
