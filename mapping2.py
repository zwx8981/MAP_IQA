"""Logistic score calibration for the supported quality models."""

import torch


QUALITY_MAPPING_PARAMETERS = {
    'ms_liqe': (100.0, 0.0, 1.984470232985137, 0.939839104077352),
    'dbcnn': (100.0, 0.0, 0.395322502452253, 0.173202751372368),
    'liqe': (100.0, 0.0, 2.077636606942305, 1.101934158051146),
    'hyperiqa': (100.0, 0.0, 0.329198721746821, 0.12451105280977),
    'musiq': (100.0, 0.0, 43.53914016357543, 16.786801106500807),
    'clipiqa+': (100.0, 0.0, 0.444621382768446, 0.157848474223641),
    'niqe': (100.0, 0.0, -9.355054431855839, 3.621900821816348),
    'paq2piq': (100.0, 0.0, 61.959934463847304, 15.24244467862268),
    'unique': (100.0, 0.0, -0.933762731495808, 1.352320478533792),
}
SUPPORTED_QUALITY_MODELS = tuple(QUALITY_MAPPING_PARAMETERS)


def logistic_mapping(x, model):
    try:
        A, B, C, S = QUALITY_MAPPING_PARAMETERS[model]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported quality model {model!r}; choose from {SUPPORTED_QUALITY_MODELS}"
        ) from exc
    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B
