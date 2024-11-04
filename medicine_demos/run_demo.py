"""Run MEDICINE on an example dataset and display summary plots.

Usage:
$ python run_demo.py

This script runs MEDICINE on the data in ./example_dataset. It runs with fewer
training steps than default to speed up the demo. The output plots are both
displayed and saved to ./medicine_output.
"""

import sys

sys.path.insert(0, "..")

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import medicine

CURRENT_DIR = Path(__file__).parent


def main():
    """Run MEDICINE on example dataset."""
    # Load data
    example_dataset_dir = CURRENT_DIR / "example_dataset"
    peak_amplitudes = np.load(example_dataset_dir / "peak_amplitudes.npy")
    peak_depths = np.load(example_dataset_dir / "peak_depths.npy")
    peak_times = np.load(example_dataset_dir / "peak_times.npy")

    # Run MEDICINE and display output plots
    medicine.run_medicine(
        peak_amplitudes=peak_amplitudes,
        peak_depths=peak_depths,
        peak_times=peak_times,
        output_dir=CURRENT_DIR / "medicine_output",
        training_steps=2000,
        motion_noise_steps=1000,
    )
    plt.show()


if __name__ == "__main__":
    main()
