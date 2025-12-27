#!/usr/bin/env python3
##
# @file matching_controller.py
#
# @brief base controller module
#
# @section Author(s)
# - Created by t.tran@hachi-x.com (2025/12/25)
#
# Copyright (c) 2025 HACHIX. All rights reserved.

# Standard library
import os

# Internal library
from controllers.base import Controller
from devices.usb_camera_interface import USBCameraInterface
from matching.invariant_template_matching import InvariantEdgeMatching


class MatchingController(Controller):
    """
    ! Matching Controller class
    """

    # =====================================================================
    # PUBLIC METHODS
    # =====================================================================
    def __init__(
        self,
        folder_path: str = None,
        camera_interface: USBCameraInterface = None,
    ):
        """
        ! Initialized the Matching Controller class
        @param folder_path<str>: Path to the folder containing images
        @param camera_interface<USBCameraInterface>: Camera interface
        """
        super().__init__(camera_interface=camera_interface)
        template_dir = os.path.join(folder_path, "templates")
        os.makedirs(template_dir, exist_ok=True)
        baseimage_dir = os.path.join(folder_path, "baseimages")
        os.makedirs(baseimage_dir, exist_ok=True)
        coordinate_dir = os.path.join(folder_path, "coordinates")
        os.makedirs(coordinate_dir, exist_ok=True)
        offset_dir = os.path.join(folder_path, "offsets")
        os.makedirs(offset_dir, exist_ok=True)
        threshold = 0.3
        angle_range = (-45, 45)
        angle_step_coarse = 5
        angle_step_fine = 3
        max_matches = 1
        pyramid_levels = 3
        self._matching = InvariantEdgeMatching(
            threshold=threshold,
            angle_range=angle_range,
            angle_step_coarse=angle_step_coarse,
            angle_step_fine=angle_step_fine,
            max_matches=max_matches,
            pyramid_levels=pyramid_levels,
            template_dir=template_dir,
            baseimage_dir=baseimage_dir,
            coordinate_dir=coordinate_dir,
            offset_dir=offset_dir,
        )
        self._count = 0

    def start(self):
        """! Start the matching controller."""
        helper = "Please move the robot so camera can see the target, \n"
        helper += "Press 'n' to enter the template GUI mode, \n"
        helper += "Press 'd' to start invariant edge matching, \n"
        helper += "Press 'q' to quit.\n"

        while True:
            command = input(helper)
            if command.lower() == "q":
                print("Exiting Matching Controller...")
                break
            if command.lower() == "n":
                image = self._camera.capture()
