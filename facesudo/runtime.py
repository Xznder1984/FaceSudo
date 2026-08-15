"""Shared runtime construction (engine + landmark predictor)."""

from __future__ import annotations

import dlib


def build_predictor():
    from face_recognition_models import pose_predictor_model_location

    return dlib.shape_predictor(pose_predictor_model_location())


def build_engine():
    from .recognition import FaceEngine

    return FaceEngine()
