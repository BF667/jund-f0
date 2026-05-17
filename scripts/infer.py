#!/usr/bin/env python3
"""
Inference script for JUND-F0

Usage:
    # Extract F0 from single file
    python -m jund_f0.scripts.infer --model_path runs/jund-f0/best_model.pt --input audio.wav

    # Batch processing
    python -m jund_f0.scripts.infer --model_path runs/jund-f0/best_model.pt --input_dir ./audio/ --output_dir ./f0_output/

    # Export F0 as CSV
    python -m jund_f0.scripts.infer --model_path runs/jund-f0/best_model.pt --input audio.wav --format csv
"""

import argparse
import sys
import numpy as np
from pathlib import Path

from jund_f0.infer import JUNDF0Inference


def main():
    parser = argparse.ArgumentParser(description="JUND-F0 Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, help="Input audio file path")
    parser.add_argument("--input_dir", type=str, help="Input directory for batch processing")
    parser.add_argument("--output_dir", type=str, default="./f0_output", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--format", type=str, default="npy", choices=["npy", "csv", "txt"], help="Output format")

    args = parser.parse_args()

    # Load model
    print(f"Loading JUND-F0 model from {args.model_path}...")
    inferencer = JUNDF0Inference(model_path=args.model_path, device=args.device)

    if args.input:
        # Single file
        print(f"Extracting F0 from {args.input}...")
        result = inferencer.extract_f0(args.input, return_vuv=True)

        # Save output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem

        if args.format == "npy":
            np.save(output_dir / f"{stem}_f0.npy", result["f0"])
            np.save(output_dir / f"{stem}_vuv.npy", result["vuv"])
        elif args.format == "csv":
            import csv
            with open(output_dir / f"{stem}_f0.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "f0", "voiced"])
                for t, f0, v in zip(result["time"], result["f0"], result["vuv"]):
                    writer.writerow([f"{t:.4f}", f"{f0:.2f}", int(v)])
        elif args.format == "txt":
            with open(output_dir / f"{stem}_f0.txt", "w") as f:
                for t, f0, v in zip(result["time"], result["f0"], result["vuv"]):
                    f.write(f"{t:.4f}\t{f0:.2f}\t{int(v)}\n")

        print(f"Output saved to {output_dir}/")
        print(f"  F0 range: {result['f0'][result['vuv']].min():.1f} - {result['f0'][result['vuv']].max():.1f} Hz")
        print(f"  Voiced frames: {result['vuv'].sum()}/{len(result['vuv'])}")

    elif args.input_dir:
        # Batch processing
        input_dir = Path(args.input_dir)
        audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.flac")) + list(input_dir.glob("*.mp3"))

        if not audio_files:
            print(f"No audio files found in {args.input_dir}")
            sys.exit(1)

        print(f"Processing {len(audio_files)} files...")
        results = inferencer.extract_f0_batch(
            [str(f) for f in audio_files],
            output_dir=args.output_dir,
        )
        print(f"Done! Output saved to {args.output_dir}/")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
