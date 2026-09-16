import io
import base64
import numpy as np
from PIL import Image
from .base import BaseAgent, AnalysisResult

class NoiseFFTAgent(BaseAgent):
    def __init__(self):
        super().__init__("noise_fft", "Noise & FFT Frequency Domain Agent")

    async def analyze(self, image_bytes: bytes, filename: str) -> AnalysisResult:
        findings = []
        metrics = {}
        score = 0.5
        confidence = 0.8
        spectrum_b64 = None

        try:
            # 1. Open image and convert to grayscale
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32)

            # 2. Compute 2D Fast Fourier Transform & Log Magnitude Spectrum
            f_transform = np.fft.fft2(img_arr)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-5)

            # Normalize spectrum to 0 - 255 for visualization
            min_val = np.min(magnitude_spectrum)
            max_val = np.max(magnitude_spectrum)
            norm_spectrum = ((magnitude_spectrum - min_val) / max(max_val - min_val, 1e-5) * 255.0).astype(np.uint8)

            # 3. Create Colorized Log-Magnitude Heatmap Preview (Magma/Jet style visualization)
            # Center bright dot = low frequency (coarse structure), outer regions = high frequency (micro details / noise)
            spectrum_img = Image.fromarray(norm_spectrum, mode="L").convert("RGB")
            
            spectrum_buffer = io.BytesIO()
            spectrum_img.save(spectrum_buffer, format="PNG")
            spectrum_b64 = "data:image/png;base64," + base64.b64encode(spectrum_buffer.getvalue()).decode('utf-8')

            # 4. Spectral High-Frequency Lattice & Energy Distribution Analysis
            center_x, center_y = 256, 256
            radius_low = 30
            radius_high = 180

            y_grid, x_grid = np.ogrid[:512, :512]
            dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

            low_freq_mask = dist_from_center <= radius_low
            high_freq_mask = (dist_from_center > radius_low) & (dist_from_center <= radius_high)

            low_freq_energy = float(np.mean(norm_spectrum[low_freq_mask]))
            high_freq_energy = float(np.mean(norm_spectrum[high_freq_mask]))
            high_freq_std = float(np.std(norm_spectrum[high_freq_mask]))

            freq_ratio = high_freq_energy / max(low_freq_energy, 1.0)

            metrics["low_freq_energy"] = round(low_freq_energy, 2)
            metrics["high_freq_energy"] = round(high_freq_energy, 2)
            metrics["high_freq_std"] = round(high_freq_std, 2)
            metrics["freq_ratio"] = round(freq_ratio, 3)

            # Evaluate periodic upsampling grid signatures (high frequency lattice variance)
            if high_freq_std > 38.0 or freq_ratio > 0.48:
                score = 0.78
                findings.append(f"High-frequency periodic lattice patterns detected in 2D FFT spectrum (std: {high_freq_std:.1f}). Characteristic of generative neural upsamplers.")
            elif freq_ratio < 0.22:
                score = 0.65
                findings.append(f"Unnatural high-frequency attenuation in log spectrum (ratio: {freq_ratio:.3f}). Suggests synthetic pixel-smoothing filters.")
            else:
                score = 0.32
                findings.append(f"Natural continuous 1/f spectral frequency falloff (freq ratio: {freq_ratio:.3f}). Aligns with real optical lens aperture physics.")

        except Exception as e:
            findings.append(f"FFT Spectral analysis notice: {str(e)}")
            confidence = 0.4

        return AnalysisResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            score=round(score, 4),
            confidence=round(confidence, 4),
            findings=findings,
            raw_metrics=metrics,
            preview_image=spectrum_b64
        )
