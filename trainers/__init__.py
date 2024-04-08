from .iql import IQL_Trainer
from .mfq import MFQ_Trainer
from .mappo import MAPPO_Trainer
from .ippo import IPPO_Trainer
from .isac import ISAC_Trainer

TRAINER_REGISTRY = {
    'IQL': IQL_Trainer,
    'MFQ': MFQ_Trainer,
    'MAPPO': MAPPO_Trainer,
    'IPPO': IPPO_Trainer,
    'ISAC': ISAC_Trainer
}
