import io
import re
from PIL import Image, ExifTags
from .base import BaseAgent, AnalysisResult

AI_SOFTWARE_KEYWORDS = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "comfyui",
    "automatic1111", "invokeai", "novelai", "firefly", "starryai",
    "leonardo.ai", "flux.1", "runway", "foocus", "civitai"
]

AI_METADATA_KEYS = [
    "prompt", "parameters", "generation_data", "workflow",
    "positive_prompt", "negative_prompt", "steps", "sampler", "seed"
]

CAMERA_EXIF_KEYS = ["Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"]

class MetadataAgent(BaseAgent):
    def __init__(self):
        super().__init__("metadata", "Metadata & Software Fingerprint Agent")

    async def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResult:
        findings = []
        metrics = {}
        score = 0.5  # Neutral default
        confidence = 0.7

        try:
            img = Image.open(io.BytesIO(image_bytes))
            fmt = (img.format or "").upper()
            metrics["format"] = fmt
            metrics["size"] = img.size

            # 1. Inspect PNG text chunks / info dictionary
            info = img.info or {}
            detected_ai_keys = []
            ai_prompt_found = False

            for k, v in info.items():
                k_str = str(k).lower()
                v_str = str(v).lower() if isinstance(v, (str, bytes)) else ""
                
                # Check for explicit generator metadata keys
                if any(ai_key in k_str for ai_key in AI_METADATA_KEYS):
                    detected_ai_keys.append(str(k))
                    ai_prompt_found = True

                # Check for AI tool names in text values
                for kw in AI_SOFTWARE_KEYWORDS:
                    if kw in v_str or kw in k_str:
                        findings.append(f"AI Generator software marker found: '{kw}' in metadata chunk '{k}'.")
                        score = 0.98
                        confidence = 0.99
                        break

            if detected_ai_keys and score < 0.9:
                findings.append(f"Generative AI metadata keys detected: {', '.join(detected_ai_keys[:4])}.")
                score = 0.95
                confidence = 0.98

            # 2. Inspect EXIF Data
            exif_data = img.getexif()
            has_camera_exif = False
            camera_details = []
            software_tag = None

            if exif_data:
                metrics["has_exif"] = True
                for tag_id, val in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    val_str = str(val)

                    if tag_name in CAMERA_EXIF_KEYS:
                        has_camera_exif = True
                        camera_details.append(f"{tag_name}: {val_str[:30]}")

                    if tag_name == "Software":
                        software_tag = val_str
                        metrics["software"] = software_tag
                        for kw in AI_SOFTWARE_KEYWORDS:
                            if kw in val_str.lower():
                                findings.append(f"EXIF Software tag explicitly matches AI tool: '{val_str}'.")
                                score = 0.99
                                confidence = 0.99

            if has_camera_exif and score < 0.8:
                findings.append(f"Valid camera EXIF parameters found ({', '.join(camera_details[:3])}). Strongly indicates physical camera capture.")
                score = min(score, 0.15)
                confidence = max(confidence, 0.85)

            # 3. Geometric Resolution & Aspect Ratio Heuristics
            w, h = img.size
            if (w, h) in [(512, 512), (1024, 1024), (768, 768), (1152, 896), (896, 1152), (1024, 576)]:
                metrics["square_standard_ai_res"] = True
                if not has_camera_exif and score < 0.7:
                    findings.append(f"Image dimensions ({w}x{h}) match exact default resolution output of AI models (e.g. Stable Diffusion / Midjourney).")
                    score = max(score, 0.65)

            if not findings:
                if not has_camera_exif:
                    findings.append("No camera EXIF metadata present. (Common in web images and AI outputs).")
                    score = 0.55
                else:
                    findings.append("Clean metadata structure with no AI generator signatures detected.")
                    score = 0.2

        except Exception as e:
            findings.append(f"Metadata extraction warning: {str(e)}")
            confidence = 0.3

        return AnalysisResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            findings=findings,
            raw_metrics=metrics
        )
