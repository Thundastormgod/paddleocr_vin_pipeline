"""
Domain errors shared by every hyperparameter tuner in this repository.

One definition, imported everywhere. This repository's audit found the same
concept implemented in multiple drifted copies (charset maps, CTC decodes, F1
scorers); a shared exception type is the same discipline applied to control
flow: an ``except TrialExecutionError`` clause in one tuner must catch exactly
what the other tuner raises, which is only guaranteed when there is a single
class object.
"""


class TrialExecutionError(RuntimeError):
    """
    A tuning trial did not produce a measurement.

    Raised instead of returning a fabricated score. A crashed or unmeasured
    trial is an absence of data, not an observation of zero accuracy: scoring
    it 0.0 feeds a fabricated observation into the TPE sampler's surrogate
    model, which then steers the remaining search away from that region of
    hyperparameter space for a reason that never happened.

    ``study.optimize(..., catch=(TrialExecutionError,))`` records such a trial
    as FAILED - visible, and excluded from the sampler's observations - while
    letting the rest of the study continue.
    """
