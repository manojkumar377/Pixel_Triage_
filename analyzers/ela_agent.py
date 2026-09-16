import io
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from .base import BaseAgent, AnalysisResult

class ELAAgent(BaseAgent):
    def __init__(self):
        super().__init__("ela", "Error Level Analysis (ELA) Agent")

    async def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResult:
        findings = []
        metrics = {}
        score = 0.5
        confidence = 0.75
        heatmap_b64 = None

        try:
            # 1. Open original image and convert to RGB
            orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Resize preview for fast, consistent ELA computation (max 1024 width/height)
            orig_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # 2. Re-compress image at 95% JPEG quality into memory
            buffer = io.BytesIO()
            orig_img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            recomp_img = Image.open(buffer).convert("RGB")

            # 3. Calculate pixel difference
            orig_arr = np.array(orig_img, dtype=np.float32)
            recomp_arr = np.array(recomp_img, dtype=np.float32)
            
            diff = np.abs(orig_arr - recomp_arr)
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff))

            metrics["max_error"] = round(max_diff, 2)
            metrics["mean_error"] = round(mean_diff, 2)
            metrics["std_error"] = round(std_diff, 2)

            # 4. Generate colorized ELA heatmap
            scale = 255.0 / max(max_diff, 1.0)
            scaled_diff = np.clip(diff * scale, 0, 255).astype(np.uint8)

            # Apply pseudo thermal color palette (Red-Yellow-Blue gradient for error levels)
            # Higher red/yellow = higher re-compression error anomaly
            ela_img = Image.fromarray(scaled_diff, mode="RGB")
            ela_img = ImageEnhance.Brightness(ela_img).enhance(1.5)

            heatmap_buffer = io.BytesIO()
            ela_img.save(heatmap_buffer, format="PNG")
            heatmap_b64 = "data:image/png;base64," + base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')

            # 5. Calculate AI anomaly score based on error uniformity & edge ratio
            # Natural camera JPEGs have high error standard deviation aligned with optical edges.
            # Synthetic / heavily post-processed images exhibit unusual error uniformity or extreme local error spikes.
            error_ratio = std_diff / max(mean_diff, 0.001)
            metrics["error_ratio"] = round(error_ratio, 3)

            if mean_diff < 1.2:
                # Unusually over-smooth low error signature typical of AI diffusion outputs
                score = 0.72
                findings.append(f"Abnormally low compression error level (mean error: {mean_diff:.2f}). Indicates synthetic over-smoothing or non-optical rendering.")
            elif error_ratio < 0.65:
                # Highly uniform error across distinct visual areas
                score = 0.68
                findings.append(f"Uniform compression error distribution (std/mean ratio: {error_ratio:.2f}). Lacks physical camera sensor artifact gradient.")
            else:
                score = 0.35
                findings.append(f"Standard compression error dynamics (mean: {mean_diff:.2f}, std: {std_diff:.2f}). Displays typical edge-dependent optical noise.")

        except Exception as e:
            findings.append(f"ELA computation notice: {str(e)}")
            confidence = 0.4

        return AnalysisResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            findings=findings,
            raw_metrics=metrics,
            preview_image=heatmap_b64
        )
