"""Read motion correction data."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import colors as matplotlib_colors
from scipy import interpolate as scipy_interpolate


def _correct_motion_on_peaks(
    peak_times, peak_depths, motion, time_bins, depth_bins
):
    corrected_peak_depths = peak_depths.copy()
    f = scipy_interpolate.RegularGridInterpolator(
        (time_bins, depth_bins),
        motion,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    shift = f(np.c_[peak_times, peak_depths])
    corrected_peak_depths -= shift

    return corrected_peak_depths


def plot_motion_correction(
    peak_times,
    peak_depths,
    peak_amplitudes,
    time_bins,
    depth_bins,
    motion,
    stride=30,
    alpha=0.75,
    lw=1,
    colormap="winter",
    motion_color="r",
):
    """Plot raster-map before and after motion correction."""

    # Subsample
    peak_times = peak_times[::stride]
    peak_depths = peak_depths[::stride]
    peak_amplitudes = peak_amplitudes[::stride]

    # Normalize amplitudes by CDF to have uniform distribution
    amp_argsort = np.argsort(np.argsort(peak_amplitudes))
    peak_amplitudes = amp_argsort / len(peak_amplitudes)

    # Get colors and create figure
    cmap = plt.get_cmap(colormap)
    colors = cmap(peak_amplitudes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)

    def _plot_neural_activity(ax, times, depths):
        plot = ax.scatter(times, depths, s=1, c=colors, alpha=alpha)
        ax.set_xlabel("time (s)", fontsize=12)
        ax.set_ylabel("depth from probe tip ($\mu$m)", fontsize=12)
        return plot

    # Plot raw peaks
    _ = _plot_neural_activity(axes[0], peak_times, peak_depths)

    # Plot raw peaks with motion correction traces
    _ = _plot_neural_activity(axes[1], peak_times, peak_depths)
    axes[1].plot(
        time_bins,
        motion + depth_bins[None],
        alpha=alpha,
        lw=lw,
        color=motion_color,
    )
    axes[1].yaxis.set_ticks_position("both")

    # Plot motion-corrected peaks
    peak_depths_corrected = _correct_motion_on_peaks(
        peak_times,
        peak_depths,
        motion,
        time_bins,
        depth_bins,
    )
    plot = _plot_neural_activity(axes[2], peak_times, peak_depths_corrected)
    fig.colorbar(plot, ax=axes[2])
    axes[2].yaxis.set_ticks_position("both")

    return fig


def plot_raster_and_amplitudes(
    peak_times, peak_depths, peak_amplitudes, stride=10, figure_dir=None
):
    """Plot raster of raw data and a histogram of amplitudes."""

    peak_times = peak_times[::stride]
    peak_depths = peak_depths[::stride]
    peak_amplitudes = peak_amplitudes[::stride]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    min_peak_amp = min(peak_amplitudes)
    max_peak_amp = max(peak_amplitudes)

    # Scatterplot peaks
    cmap = plt.get_cmap("winter")
    norm = matplotlib_colors.Normalize(
        vmin=min_peak_amp, vmax=max_peak_amp, clip=True
    )
    colors = cmap(norm(peak_amplitudes))
    axes[0].scatter(peak_times, peak_depths, s=1, c=colors, alpha=0.5)
    axes[0].set_xlabel("time (s)", fontsize=12)
    axes[0].set_ylabel("depth from probe tip ($\mu$m)", fontsize=12)
    axes[0].set_title("Peaks over session")

    # Histogram peak amplitudes
    axes[1].hist(peak_amplitudes, bins=30)
    axes[1].set_xlabel("Amplitude")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Histogram of peak amplitudes")

    fig.tight_layout()

    # Save figure
    if figure_dir is not None:
        save_path = figure_dir / "raw_raster_and_amplitudes.png"
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path)

    return fig


def plot_training_loss(losses, figure_dir=None):
    """Plot training loss."""

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_title("Loss throughout training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")

    # Save figure
    if figure_dir is not None:
        save_path = figure_dir / "training_loss.png"
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path)

    return fig


def plot_depth_amplitude_distributions(
    medicine_model,
    dataset,
    figure_dir=None,
    grid_size=100,
    num_timepoints=5,
    data_samples_per_plot=30000,
):
    """Plot distribution of depths and amplitudes"""

    # Find timepoints to plot
    time_slice_proportions = np.linspace(0.0, 1.0, num_timepoints)[1:-1]
    dataset_times = dataset.times.numpy()
    min_time = np.min(dataset_times)
    max_time = np.max(dataset_times)
    time_slices = [
        min_time + t * (max_time - min_time) for t in time_slice_proportions
    ]

    # Create figure
    fig, (axes_model, axes_data) = plt.subplots(
        2,
        len(time_slices),
        figsize=(3 * len(time_slices), 6),
    )
    fig.suptitle("Predicted depth-amplitude distributions", y=1.05)

    # Plot model-predicted distribution
    for ax, time in zip(axes_model, time_slices):
        uniform_batch = dataset.sample_uniform(time=time, grid_size=grid_size)
        pred_distrib = medicine_model(uniform_batch).detach().numpy()
        pred_distrib = np.reshape(pred_distrib, (grid_size, grid_size))
        extent = tuple(
            list(dataset.amplitude_range) + list(dataset.depth_range)
        )
        ax.imshow(pred_distrib[::-1], extent=extent, aspect="auto")
        ax.set_title(f"Model, Time = {int(np.round(time))} sec")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Depth from probe tip ($\mu$m)")

    # Plot data distribution
    loc_range = dataset.depth_range
    amp_range = dataset.amplitude_range
    for ax, time in zip(axes_data, time_slices):
        temporal_proximity_inds = np.argsort(np.abs(dataset.times - time))
        sample_inds = temporal_proximity_inds[:data_samples_per_plot]
        depths = dataset.depths_normalized[sample_inds]
        amplitudes = dataset.amplitudes_normalized[sample_inds]
        depths = loc_range[0] + depths * (loc_range[1] - loc_range[0])
        amplitudes = amp_range[0] + amplitudes * (amp_range[1] - amp_range[0])
        try:
            sns.kdeplot(
                ax=ax,
                x=amplitudes,
                y=depths,
                fill=True,
                levels=100,
                cmap="winter",
                gridsize=grid_size,
                thresh=0.01,
            )
        except:
            print("Error happened in sns.kdeplot")
        ax.set_xlim(amp_range)
        ax.set_ylim(loc_range)
        ax.set_title(f"Data, Time = {int(np.round(time))} sec")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Depth from probe tip ($\mu$m)")

    fig.tight_layout()

    # Save figure
    if figure_dir is not None:
        save_path = figure_dir / "depth_amplitude_distributions.png"
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path)

    return fig


def _get_predicted_motion(dataset, medicine_model, num_depth_bins=15):
    # Get times to evaluate predicted motion
    time_range = medicine_model.motion_function.time_range
    num_time_bins = medicine_model.motion_function.num_time_bins
    time_bins = np.linspace(*time_range, num_time_bins)

    # Get depth bins
    depth_bins = np.linspace(0, 1, num_depth_bins)

    # Get grid of time and depth bins
    times = np.repeat(time_bins[:, None], num_depth_bins, axis=1)
    depths = np.repeat(depth_bins[None, :], num_time_bins, axis=0)

    # Convert times and depths to torch
    times_torch = torch.from_numpy(times.astype(np.float32))
    depths_torch = torch.from_numpy(depths.astype(np.float32))

    # Get predicted motion of shape [num_time_bins, num_depth_bins]
    times_torch_flat = times_torch.flatten()
    depths_torch_flat = depths_torch.flatten()
    pred_motion_flat = medicine_model.motion_function(
        times_torch_flat, depths_torch_flat
    )
    pred_motion = pred_motion_flat.reshape(num_time_bins, num_depth_bins)
    pred_motion = -1 * pred_motion.detach().numpy()
    pred_motion *= dataset.depth_range[1] - dataset.depth_range[0]

    # Un-normalize depths
    depths = dataset.depth_range[0] + depths * (
        dataset.depth_range[1] - dataset.depth_range[0]
    )

    return times, depths, pred_motion


def plot_predicted_motion(
    medicine_model, dataset, num_depth_bins=15, figure_dir=None
):
    """Plot motion over time."""

    # Get predicted motion
    times, depths, pred_motion = _get_predicted_motion(
        dataset, medicine_model, num_depth_bins=num_depth_bins
    )

    # Plot predicted motion
    fig, ax = plt.subplots(figsize=(3, 6))
    ax.plot(times, depths + pred_motion, c="r")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Predicted deviation ($\mu$m)")
    ax.set_title("Predicted motion")

    # Save figure
    if figure_dir is not None:
        save_path = figure_dir / "predicted_motion.png"
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path)

    return fig


def plot_motion_corrected_raster(
    dataset, medicine_model, num_depth_bins=15, figure_dir=None
):

    # Get predicted motion
    times, depths, pred_motion = _get_predicted_motion(
        dataset, medicine_model, num_depth_bins=num_depth_bins
    )

    # Plot new motion correction
    fig = plot_motion_correction(
        peak_times=dataset.times,
        peak_depths=dataset.depths_raw,
        peak_amplitudes=dataset.amplitudes_raw,
        time_bins=times[:, 0],
        depth_bins=depths[0],
        motion=pred_motion,
    )
    _ = fig.suptitle("Motion correction", fontsize=24)

    # Save figure
    if figure_dir is not None:
        save_path = figure_dir / "corrected_motion_raster.png"
        print(f"Saving figure to {save_path}")
        fig.savefig(save_path)

    return fig


def run_post_motion_estimation_plots(figure_dir, trainer):
    # Create figure_dir if necessary
    if not figure_dir.exists():
        figure_dir.mkdir(parents=True)

    # Extract needed attributes from trainer
    medicine_model = trainer.medicine_model
    dataset = trainer.dataset
    losses = trainer.losses

    # Run and save plots
    _ = plot_training_loss(losses, figure_dir=figure_dir)

    _ = plot_depth_amplitude_distributions(
        medicine_model, dataset, figure_dir=figure_dir
    )
    _ = plot_predicted_motion(medicine_model, dataset, figure_dir=figure_dir)
    _ = plot_motion_corrected_raster(
        dataset, medicine_model, figure_dir=figure_dir
    )
