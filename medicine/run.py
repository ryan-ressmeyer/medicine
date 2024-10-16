"""Run motion correction."""

import json
from pathlib import Path

import numpy as np
import torch

from medicine import model, plotting


def run_medicine(
    peak_times,
    peak_depths,
    peak_amplitudes,
    output_dir,
    motion_bound=800,
    temporal_resolution=1,
    temporal_kernel_width=30,
    motion_corrector_hidden_features=(256, 256),
    num_depth_bins=2,
    depth_smoothing=None,
    batch_size=4096,
    training_steps=10000,
    initial_motion_noise=0.1,
    motion_noise_steps=2000,
    optimizer=torch.optim.Adam,
    learning_rate=0.0005,
    epsilon=1e-3,
    plot_figures=True,
):
    """Run motion correction."""

    # Create and clear output_dir
    print(f"\nCreating output_dir {output_dir}")
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Warning: {output_dir} already exists")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save parameters
    parameters = dict(
        output_dir=str(output_dir),
        motion_bound=motion_bound,
        temporal_resolution=temporal_resolution,
        temporal_kernel_width=temporal_kernel_width,
        motion_corrector_hidden_features=motion_corrector_hidden_features,
        num_depth_bins=num_depth_bins,
        depth_smoothing=depth_smoothing,
        batch_size=batch_size,
        training_steps=training_steps,
        initial_motion_noise=initial_motion_noise,
        motion_noise_steps=motion_noise_steps,
        optimizer=f"{optimizer.__module__}.{optimizer.__name__}",
        learning_rate=learning_rate,
        epsilon=epsilon,
        plot_figures=plot_figures,
    )
    parameters_path = output_dir / "medicine_parameters.json"
    print(f"\nSaving parameters to {output_dir}")
    json.dump(parameters, open(parameters_path, "w"))

    # Plot raster and amplitudes if necessary
    if plot_figures:
        plotting.plot_raster_and_amplitudes(
            peak_times, peak_depths, peak_amplitudes, figure_dir=output_dir
        )

    # Create dataset
    dataset = model.Dataset(
        times=peak_times,
        depths=peak_depths,
        amplitudes=peak_amplitudes,
    )

    # Create motion_predictor
    motion_bound_normalized = (
        0.5 * motion_bound / (dataset.depth_range[1] - dataset.depth_range[0])
    )
    time_range = (
        np.min(peak_times) - epsilon,
        np.max(peak_times) + epsilon,
    )
    motion_predictor = model.MotionPredictor(
        bound_normalized=motion_bound_normalized,
        time_range=time_range,
        time_bin_size=temporal_resolution,
        time_kernel_width=temporal_kernel_width,
        num_depth_bins=num_depth_bins,
        depth_smoothing=depth_smoothing,
    )

    # Create motion_corrector
    motion_corrector = model.MotionCorrector(
        motion_predictor=motion_predictor,
        distribution_predictor=model.DistributionPredictor(
            hidden_features=motion_corrector_hidden_features,
        ),
    )

    # Create trainer
    trainer = model.Trainer(
        dataset,
        motion_corrector=motion_corrector,
        batch_size=batch_size,
        training_steps=training_steps,
        initial_motion_noise=initial_motion_noise,
        motion_noise_steps=motion_noise_steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
    )

    # Run trainer
    trainer()

    # Plot motion correction results if necessary
    if plot_figures:
        plotting.run_post_motion_correction_plots(
            figure_dir=output_dir,
            trainer=trainer,
        )

    return trainer
