# Standard library
from abc import ABC, abstractmethod

# Extenral library
import cv2
import numpy as np

# Internal library
from devices.usb_camera_interface import USBCameraInterface


class Controller(ABC):
    """
    ! Base Controller class
    """

    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================
    def __init__(
            self,
            camera_interface: USBCameraInterface = None,
    ):
        """
        ! Initialized the Controller base class
        @param camera_interface<USBCameraInterface>: Camera interface
        """
        self._camera = camera_interface

    @abstractmethod
    def start(self):
        """! Abstract method to start the controller."""