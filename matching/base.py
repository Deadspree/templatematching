#!/usr/bin/env python3
##
# @file calibrator.py
#
# @brief calibrator abstract module
#
# @section Author(s)
# - Created by minhc613@gmail.com (2025/12/25)
#
# Copyright (c) 2025 HACHIX. All rights reserved.

# Standard library
import os
import json
from abc import ABC, abstractmethod
from typing import Tuple, Generator

# External library
import cv2
import numpy as np

class ITM(ABC):
    """
    !Abstract class for invariant template matching
    """
    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================
    def __init__(
            self,
            raw_images_directory: str,
            output_directory: str,
    ):
        """
        !Intiallize invariant template matching class
        @param raw_images_directory<str>: Directory path of raw images
        @param output_directory<str>: Directory path of output results
        """
        self.raw_images_directory = raw_images_directory
        self.output_directory = output_directory
    
    @abstractmethod
    def calculate_offset