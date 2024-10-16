"""Run MEDICINE on an example dataset and display summary plots."""

import sys

sys.path.insert(0, "..")

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from medicine import run as medicine_run


def main():
    """Run MEDICINE on example dataset."""
    # Load data
    dataset_dir = Path("./example_dataset")
    peak_amplitudes = np.load(dataset_dir / "peak_amplitudes.npy")
    peak_depths = np.load(dataset_dir / "peak_depths.npy")
    peak_times = np.load(dataset_dir / "peak_times.npy")

    # Run MEDICINE and display output plots
    medicine_run.run_medicine(
        peak_amplitudes=peak_amplitudes,
        peak_depths=peak_depths,
        peak_times=peak_times,
        output_dir="medicine_output",
        training_steps=2000,
        motion_noise_steps=1000,
    )
    plt.show()


if __name__ == "__main__":
    main()
