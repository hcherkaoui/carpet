""" Carpet. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

from .lista_synthesis import ListaLASSO, CoupledIstaLASSO, StepIstaLASSO
from .lista_analysis import (StepSubGradTV, ListaTV, OrigChambolleTV,
                             CoupledChambolleTV, StepChambolleTV,
                             CoupledCondatVu, StepCondatVu,
                             LpgdTautString)

__version__ = '0.1.dev0'


_ALL_ALGO_ = dict(
    origista=ListaLASSO,
    coupledista=CoupledIstaLASSO,
    stepista=StepIstaLASSO,
    stepsubgradient=StepSubGradTV,
    origtv=ListaTV,
    origchambolle=OrigChambolleTV,
    coupledchambolle=CoupledChambolleTV,
    stepchambolle=StepChambolleTV,
    coupledcondatvu=CoupledCondatVu,
    stepcondatvu=StepCondatVu,
    lpgd_taut_string=LpgdTautString
)


def LearnTVAlgo(**kwargs):
    algo_type = kwargs.pop('algo_type')
    return _ALL_ALGO_[algo_type](**kwargs)


__all__ = [
    'LearnTVAlgo',
]
