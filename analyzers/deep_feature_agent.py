import io
import numpy as np
from PIL import Image, ImageStat
from .base import BaseAgent, AnalysisResult

class DeepFeatureAgent(BaseAgent):
    def __init__(self):
        super().__init__("deep_feature", "Visual Anomaly & Feature Agent")

    async def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResult:
        findings = []
        metrics = {}
        score = 0.5
        confidence = 0.7

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img.thumbnail((512, 512), Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32)

            # 1. Color Channel Correlation & Entropy Analysis
            r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
            
            # Compute channel variance & cross-correlation
            rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
            rb_corr = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
            gb_corr = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])
            mean_corr = (rg_corr + rb_corr + gb_corr) / 3.0

            metrics["channel_correlation"] = round(mean_corr, 4)

            # 2. Saturation & Luminance Gamut Distribution
            hsv_img = img.convert("HSV")
            hsv_arr = np.array(hsv_img, dtype=np.float32)
            sat = hsv_arr[:, :, 1]
            val = hsv_arr[:, :, 2]

            sat_mean = float(np.mean(sat))
            sat_std = float(np.std(sat))
            val_std = float(np.std(val))

            metrics["sat_mean"] = round(sat_mean, 2)
            metrics["sat_std"] = round(sat_std, 2)

            # 3. Micro-Edge Laplacian Sharpness Distribution
            # Compute simple 3x3 Laplacian edge magnitude
            gray = np.array(img.convert("L"), dtype=np.float32)
            laplacian = (
                -4 * gray[1:-1, 1:-1] +
                gray[0:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, 0:-2] + gray[1:-1, 2:]
            )
            edge_variance = float(np.var(laplacian))
            metrics["edge_variance"] = round(edge_variance, 2)

            # Anomaly rules
            if sat_mean > 165.0 and sat_std < 42.0:
                score = 0.74
                findings.append(f"Hyper-saturated color distribution (mean saturation: {sat_mean:.1f}). Common in digital AI art and stylized diffusion prompts.")
            elif edge_variance < 150.0:
                score = 0.68
                findings.append(f"Unusually low micro-edge Laplacian variance ({edge_variance:.1f}). Indicates synthetic softness or lack of optical micro-texture.")
            elif mean_corr > 0.965:
                score = 0.65
                findings.append(f"Hyper-correlated RGB color channels ({mean_corr:.4f}). Exceeds typical natural optical chromatic aberration range.")
            else:
                score = 0.30
                findings.append(f"Natural visual feature metrics (edge variance: {edge_variance:.1f}, channel corr: {mean_corr:.4f}). Consistent with organic photography.")

        except Exception as e:
            findings.append(f"Visual feature inspection notice: {str(e)}")
            confidence = 0.4

        return AnalysisResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            findings=findings,
            raw_metrics=metrics
        )
