from dataclasses import dataclass
from typing import Literal

AgentName = Literal["cuota_tae", "coste_ratio"]


@dataclass(frozen=True)
class RouteDecision:
    agent: AgentName
    reason: str = ""