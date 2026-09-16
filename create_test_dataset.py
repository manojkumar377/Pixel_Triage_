import os
import io
import numpy as np
from PIL import Image, PngImagePlugin

def create_dataset():
    dataset_dir = "dataset"
    real_dir = os.path.join(dataset_dir, "real")
    ai_dir = os.path.join(dataset_dir, "ai")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    print("Generating benchmark test images...")

    # 1. Generate Real Test Image 1 (Natural photo simulation)
    # Natural image with organic noise & camera EXIF representation
    arr_real1 = np.zeros((800, 1200, 3), dtype=np.float32)
    # Add continuous gradient & optical noise
    for y in range(800):
        for x in range(1200):
            arr_real1[y, x] = [
                (x / 1200.0) * 180 + np.random.normal(0, 10),
                (y / 800.0) * 160 + np.random.normal(0, 10),
                ((x + y) / 2000.0) * 200 + np.random.normal(0, 10)
            ]

    img_real1 = Image.fromarray(np.clip(arr_real1, 0, 255).astype(np.uint8))
    # Add EXIF tags
    exif = img_real1.getexif()
    exif[271] = "Canon"  # Make
    exif[272] = "Canon EOS R5"  # Model
    exif[305] = "Adobe Photoshop 2023"  # Software
    img_real1.save(os.path.join(real_dir, "real_camera_photo_1.jpg"), exif=exif, quality=92)

    # 2. Generate Real Test Image 2 (Landscape optical photo)
    arr_real2 = np.random.randint(50, 220, (600, 900, 3), dtype=np.uint8)
    img_real2 = Image.fromarray(arr_real2)
    exif2 = img_real2.getexif()
    exif2[271] = "Sony"
    exif2[272] = "ILCE-7RM4"
    img_real2.save(os.path.join(real_dir, "real_camera_photo_2.jpg"), exif=exif2, quality=95)

    # 3. Generate AI Test Image 1 (Automatic1111 / Stable Diffusion PNG)
    arr_ai1 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Create periodic upsampling lattice pattern (transposed convolution artifact in FFT)
    y_idx, x_idx = np.ogrid[:1024, :1024]
    lattice = (np.sin(x_idx / 4.0) * np.cos(y_idx / 4.0) * 40).astype(np.uint8)
    for c in range(3):
        arr_ai1[:, :, c] = 128 + lattice

    img_ai1 = Image.fromarray(arr_ai1)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("parameters", "cyberpunk street at night, neon reflections, 8k resolution, highly detailed, photorealistic master piece\nNegative prompt: blurry, bad anatomy\nSteps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 39482910, Size: 1024x1024, Model: sd_xl_base_1.0")
    img_ai1.save(os.path.join(ai_dir, "ai_diffusion_sdxl_1.png"), pnginfo=png_info)

    # 4. Generate AI Test Image 2 (Midjourney v6 style image)
    arr_ai2 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    # Hyper-saturated uniform smooth color gradient with high-frequency periodic noise
    for y in range(1024):
        arr_ai2[y, :, 0] = int(240 * (y / 1024.0))
        arr_ai2[y, :, 1] = int(30 * (y / 1024.0))
        arr_ai2[y, :, 2] = int(180 * (1.0 - y / 1024.0))
    
    # Add artificial grid lattice
    arr_ai2[::8, :, :] = np.clip(arr_ai2[::8, :, :] + 25, 0, 255)
    img_ai2 = Image.fromarray(arr_ai2)
    png_info2 = PngImagePlugin.PngInfo()
    png_info2.add_text("Software", "Midjourney v6.0")
    img_ai2.save(os.path.join(ai_dir, "ai_midjourney_v6_2.png"), pnginfo=png_info2)

    print("Test dataset created successfully in ./dataset/")

if __name__ == "__main__":
    create_dataset()
