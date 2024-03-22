from .iql import IQL_Agent
from .mfq import MFQ_Agent

AGENT_REGISTRY = {
    "IQL": IQL_Agent,
    "MFQ": MFQ_Agent
}
