"""
Domain errors for model evaluation.

The multi-model evaluator previously converted every failure into the
prediction ``("", 0.0)``: a crashed OCR call, an unreadable image, a model
that failed to initialise and a dispatch string that matched nothing were
all recorded as "the model predicted an empty string with zero confidence"
- indistinguishable, inside the accuracy denominator, from a model that
genuinely read nothing. A fine-tuned DeepSeek export was tabulated at 0%
for months because its dispatch key never matched and the else-branch
scored it image by image.

These exceptions separate the three situations that must never be
conflated with a measurement:
"""


class EvaluationError(RuntimeError):
    """Base class for evaluation failures that are not measurements."""


class ModelExecutionError(EvaluationError):
    """
    One image could not be evaluated (OCR call crashed, image unreadable).

    Recorded per-image as an evaluation error and excluded from the metric
    denominators; the rest of the dataset continues.
    """


class ModelUnavailableError(EvaluationError):
    """
    The model cannot run at all (initialisation failed, unknown dispatch
    type, missing runtime). The model is reported as NOT EVALUATED rather
    than as scoring 0%.
    """
