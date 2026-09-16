import os
import sys
import io
import time
import json
import asyncio
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from analyzers.metadata_agent import MetadataAgent
from analyzers.c2pa_agent import C2PAAgent
from analyzers.ela_agent import ELAAgent
from analyzers.noise_fft_agent import NoiseFFTAgent
from analyzers.deep_feature_agent import DeepFeatureAgent
from analyzers.judge_agent import JudgeAgent

metadata_agent = MetadataAgent()
c2pa_agent = C2PAAgent()
ela_agent = ELAAgent()
noise_fft_agent = NoiseFFTAgent()
deep_feature_agent = DeepFeatureAgent()
judge_agent = JudgeAgent()

async def evaluate_single_image(image_path: str):
    start_time = time.perf_counter()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    filename = os.path.basename(image_path)
    
    t0 = time.perf_counter()
    results = await asyncio.gather(
        metadata_agent.analyze(image_bytes, filename),
        c2pa_agent.analyze(image_bytes, filename),
        ela_agent.analyze(image_bytes, filename),
        noise_fft_agent.analyze(image_bytes, filename),
        deep_feature_agent.analyze(image_bytes, filename)
    )
    pipeline_time_ms = (time.perf_counter() - t0) * 1000

    verdict_report = judge_agent.synthesize(results)
    verdict_report["latency_ms"] = round(pipeline_time_ms, 2)
    return verdict_report

async def run_benchmark(dataset_dir: str):
    real_dir = os.path.join(dataset_dir, "real")
    ai_dir = os.path.join(dataset_dir, "ai")

    if not os.path.exists(real_dir) or not os.path.exists(ai_dir):
        print(f"[ERROR] Benchmark dataset folder structure missing.")
        print(f"Please create subfolders:")
        print(f"  - {real_dir}")
        print(f"  - {ai_dir}")
        print("Add sample JPEG/PNG images into each subfolder and run again.")
        return

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    ai_files = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    total_images = len(real_files) + len(ai_files)
    if total_images == 0:
        print("[WARNING] No test images found in dataset subfolders.")
        return

    print(f"\n=======================================================")
    print(f"  AI Image Forensics Benchmark & Accuracy Evaluator")
    print(f"=======================================================")
    print(f"Dataset Target: {dataset_dir}")
    print(f"Real Photos Count: {len(real_files)}")
    print(f"AI Generated Count: {len(ai_files)}")
    print(f"Total Test Suite: {total_images} images\n")

    tp, fp, tn, fn = 0, 0, 0, 0
    latencies = []

    # Evaluate Real Images (Ground Truth = Real, Label 0)
    print("Evaluating Real Photographs...")
    for img_path in real_files:
        try:
            res = await evaluate_single_image(img_path)
            latencies.append(res["latency_ms"])
            is_ai_pred = res["ai_confidence_percentage"] >= 50.0

            if is_ai_pred:
                fp += 1  # False Positive (Real flagged as AI)
            else:
                tn += 1  # True Negative (Real correctly identified)
        except Exception as e:
            print(f"  Failed {os.path.basename(img_path)}: {e}")

    # Evaluate AI Images (Ground Truth = AI, Label 1)
    print("Evaluating AI Generated Images...")
    for img_path in ai_files:
        try:
            res = await evaluate_single_image(img_path)
            latencies.append(res["latency_ms"])
            is_ai_pred = res["ai_confidence_percentage"] >= 50.0

            if is_ai_pred:
                tp += 1  # True Positive (AI correctly identified)
            else:
                fn += 1  # False Negative (AI missed as Real)
        except Exception as e:
            print(f"  Failed {os.path.basename(img_path)}: {e}")

    # Compute Forensic Evaluation Metrics
    accuracy = (tp + tn) / max(total_images, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1_score = (2 * precision * recall) / max(precision + recall, 1e-5)
    fpr = fp / max(fp + tn, 1) * 100
    fnr = fn / max(tp + fn, 1) * 100

    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0

    print("\n=======================================================")
    print("                 BENCHMARK RESULTS REPORT              ")
    print("=======================================================")
    print(f"Overall Accuracy    : {accuracy:.2f}%")
    print(f"Precision           : {precision:.2f}%")
    print(f"Recall (Sensitivity): {recall:.2f}%")
    print(f"F1-Score            : {f1_score:.2f}%")
    print(f"False Positive Rate : {fpr:.2f}% (Real flagged as AI)")
    print(f"False Negative Rate : {fnr:.2f}% (AI missed as Real)")
    print("-------------------------------------------------------")
    print("Confusion Matrix:")
    print(f"  True Positives  (AI -> AI)   : {tp}")
    print(f"  True Negatives  (Real -> Real): {tn}")
    print(f"  False Positives (Real -> AI) : {fp}")
    print(f"  False Negatives (AI -> Real) : {fn}")
    print("-------------------------------------------------------")
    print("Performance & Speed Benchmarks:")
    print(f"  Average Pipeline Latency : {avg_latency:.2f} ms / image")
    print(f"  95th Percentile Latency  : {p95_latency:.2f} ms / image")
    print("=======================================================\n")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    asyncio.run(run_benchmark(target_dir))
