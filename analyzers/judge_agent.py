from typing import List
from .base import AnalysisResult

AGENT_WEIGHTS = {
    "c2pa": 1.00,
    "metadata": 0.90,
    "noise_fft": 0.75,
    "ela": 0.65,
    "deep_feature": 0.50
}

class JudgeAgent:
    def __init__(self):
        self.agent_id = "judge"
        self.agent_name = "Judge Consensus & Reasoning Agent"

    def synthesize(self, results: List[AnalysisResult]) -> dict:
        weighted_score_sum = 0.0
        weighted_conf_sum = 0.0
        agent_breakdown = {}
        all_findings = []

        deterministic_ai_override = False
        deterministic_real_override = False
        override_reason = ""

        for res in results:
            w = AGENT_WEIGHTS.get(res.agent_id, 0.5)
            c = max(res.confidence, 0.1)

            weighted_score_sum += res.score * w * c
            weighted_conf_sum += w * c

            # Store breakdown for radar chart / UI
            agent_breakdown[res.agent_id] = {
                "name": res.agent_name,
                "score": round(res.score * 100, 1),
                "confidence": round(res.confidence * 100, 1),
                "findings": res.findings,
                "preview_image": res.preview_image
            }

            all_findings.extend(res.findings)

            # Check for deterministic signatures
            if res.agent_id in ["metadata", "c2pa"]:
                if res.score >= 0.95 and res.confidence >= 0.95:
                    deterministic_ai_override = True
                    override_reason = res.findings[0] if res.findings else "Hard AI metadata detected."
                elif res.score <= 0.05 and res.confidence >= 0.90:
                    deterministic_real_override = True
                    override_reason = res.findings[0] if res.findings else "Verified physical camera hardware capture."

        # Calculate final aggregated score (0.0 to 1.0)
        if deterministic_ai_override:
            final_score = 0.985
            overall_confidence = 0.99
        elif deterministic_real_override:
            final_score = 0.025
            overall_confidence = 0.98
        else:
            final_score = weighted_score_sum / max(weighted_conf_sum, 0.001)
            overall_confidence = min(weighted_conf_sum / sum(AGENT_WEIGHTS.values()), 0.95)

        ai_percentage = round(final_score * 100, 1)
        real_percentage = round((1.0 - final_score) * 100, 1)

        # Classification label decision
        if ai_percentage >= 60.0:
            verdict = "AI Generated Image"
            verdict_code = "AI_GENERATED"
            risk_level = "HIGH"
        elif ai_percentage <= 40.0:
            verdict = "Real Photograph / Natural Capture"
            verdict_code = "REAL_IMAGE"
            risk_level = "LOW"
        else:
            verdict = "Uncertain / Mixed Forensic Indicators"
            verdict_code = "UNCERTAIN"
            risk_level = "MEDIUM"

        # Formulate executive reasoning summary
        reasoning_bullets = []
        if override_reason:
            reasoning_bullets.append(f"🔒 DETERMINISTIC SIGNATURE: {override_reason}")

        for res in results:
            if res.findings:
                bullet_icon = "⚠️" if res.score >= 0.6 else ("✅" if res.score <= 0.4 else "🔍")
                reasoning_bullets.append(f"{bullet_icon} [{res.agent_name}]: {res.findings[0]}")

        return {
            "verdict": verdict,
            "verdict_code": verdict_code,
            "risk_level": risk_level,
            "ai_confidence_percentage": ai_percentage,
            "real_confidence_percentage": real_percentage,
            "overall_confidence": round(overall_confidence * 100, 1),
            "reasoning_summary": reasoning_bullets,
            "agent_breakdown": agent_breakdown
        }
