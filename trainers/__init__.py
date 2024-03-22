from .iql import IQL_Trainer
from .mfq import MFQ_Trainer

TRAINER_REGISTRY = {
    'IQL': IQL_Trainer,
    'MFQ': MFQ_Trainer
}
