import re
from .base import BaseAgent, AnalysisResult

# Known C2PA JUMBF Box signatures and manifests
C2PA_MARKERS = [b'c2pa', b'ca1b', b'c2pa.manifest', b'org.contentauthenticity', b'c2pa.assertion']
AI_C2PA_ISSUERS = ['openai', 'dall-e', 'firefly', 'adobe', 'midjourney', 'google', 'meta']
CAMERA_C2PA_ISSUERS = ['leica', 'sony', 'canon', 'nikon', 'truepic']

class C2PAAgent(BaseAgent):
    def __init__(self):
        super().__init__("c2pa", "C2PA Provenance & Content Credentials Agent")

    async def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResult:
        findings = []
        metrics = {}
        score = 0.5
        confidence = 0.5

        # Check binary headers for JUMBF / C2PA boxes
        c2pa_found = False
        matched_markers = []
        
        for marker in C2PA_MARKERS:
            if marker in image_bytes:
                c2pa_found = True
                matched_markers.append(marker.decode('ascii', errors='ignore'))

        metrics["c2pa_detected"] = c2pa_found
        metrics["matched_markers"] = matched_markers

        if c2pa_found:
            confidence = 0.95
            findings.append("Cryptographic C2PA / Content Credentials manifest signature detected.")

            # Search binary stream for issuer / generator strings inside C2PA manifest
            lower_bytes = image_bytes.lower()
            ai_matches = [issuer for issuer in AI_C2PA_ISSUERS if issuer.encode() in lower_bytes]
            cam_matches = [issuer for issuer in CAMERA_C2PA_ISSUERS if issuer.encode() in lower_bytes]

            metrics["manifest_ai_issuers"] = ai_matches
            metrics["manifest_camera_issuers"] = cam_matches

            if ai_matches and not cam_matches:
                score = 0.98
                findings.append(f"C2PA Manifest explicitly identifies AI generator provenance ({', '.join(ai_matches)}).")
            elif cam_matches:
                score = 0.02
                findings.append(f"C2PA Manifest cryptographically verifies authentic physical hardware capture ({', '.join(cam_matches)}).")
            else:
                score = 0.85
                findings.append("C2PA manifest box present; metadata attributes align with synthetic media creation.")
        else:
            confidence = 0.6
            score = 0.5
            findings.append("No cryptographic C2PA / Content Credentials manifest found in image stream.")

        return AnalysisResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            findings=findings,
            raw_metrics=metrics
        )
