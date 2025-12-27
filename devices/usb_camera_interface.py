#!/usr/bin/env python3
##
# @file tis_interface.py
#
# @brief TIS device interface module
#
# @section author
# - Converted by Tran Viet Thanh (2025/12/10)
# - Modifed by Le Quang Minh (2025/12/10)
#
# Copyright (c) 2025 HACHIX. All rights reserved.

# Standard library
import threading

# External library
import cv2


class USBCameraInterface:
    """! USB Camera device interface class."""

    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================
    def __init__(self):
        """! Initialize the USB Camera Interface class."""
        self.is_running = False
        self.frame = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def open(self):
        """! Open the USB Camera device connection."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open USB camera device.")
        self.is_running = True
        self.thread.start()

    def close(self):
        """! Close the TIS device connection."""
        self.is_running = False
        self.thread.join()

    def capture(self):
        """! Capture an image from the TIS device.
        @return<np.ndarray>: Captured image.
        """
        return self.frame

    # =====================================================================
    # PRIVATE METHODS
    # =====================================================================
    def _loop(self):
        """! Thread loop to continuously capture images from the USB camera."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            self.frame = frame
        self.cap.release()
