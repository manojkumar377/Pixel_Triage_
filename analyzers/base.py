from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

@dataclass
class AnalysisResult:
    agent_id: str
    agent_name: str
    score: float  # 0.0 = Real / Natural, 1.0 = AI Generated / Synthetic
    confidence: float  # 0.0 = Low, 1.0 = High
    findings: List[str] = field(default_factory=list)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    preview_image: Optional[str] = None  # Base64 data URL for heatmap / spectrum visualization

class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name

    @abstractmethod
    async def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResult:
        """Run analysis on raw image bytes and return AnalysisResult."""
        pass
