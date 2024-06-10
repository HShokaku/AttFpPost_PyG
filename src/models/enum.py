from enum import Enum


class DensityType(Enum):
    RADIAL = "radial_flow"
    IAF = "iaf_flow"


class Parametrization(Enum):
    EPS = "eps"
    MEAN = "mean"


class Architecture(Enum):
    AttFp = "attfp"
    AttFpPost = "attfppost"


class SamplingMode(Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
