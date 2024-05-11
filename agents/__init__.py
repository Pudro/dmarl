from .iql import IQL_Agent
from .mfq import MFQ_Agent
from .mappo import MAPPO_Agent
from .ippo import IPPO_Agent
from .isac import ISAC_Agent
from .masac import MASAC_Agent
from .qmix import QMIX_Agent
from .random import Random_Agent

AGENT_REGISTRY = {
    "IQL": IQL_Agent,
    "MFQ": MFQ_Agent,
    "MAPPO": MAPPO_Agent,
    "IPPO": IPPO_Agent,
    "ISAC": ISAC_Agent,
    "MASAC": MASAC_Agent,
    "QMIX": QMIX_Agent,
    "Random": Random_Agent
}
