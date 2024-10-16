"""Motion correction algorithm.

The general idea of this algorithm is to fit two things to a recording:
    (i) The motion of the brain relative to a probe in depth.
    (ii) An activity distribution of the brain along the probe.
These are fit based on peaks (essentially threshold crossings) extracted from
neural recording. The peaks have two parameters: depth and amplitude.

The algorithm fits motion and activity simultaneously. They are parameterized as
follows:
    Motion: The motion is parameterized as a discretized timeseries across the
        entire session. The discretization has bin size, e.g. 1 second, in which
        case motion is a vector of length number of seconds in the session,
        where values are the motion in microns at that point in time.
        Importantly, we smooth this motion vector with a triangular kernel
        (~30-second support tends to work well), to prevent the motion from
        being too jumpy and noise-sensitive. In the future, a Gaussian process
        prior might be more principled.
    Activity distribution: The activity distribution in [depth, amplitude] space
        is parameterized by a neural network taking a (depth, amplitude) point
        and returning the probability of receiving a spike with that depth and
        amplitude. In other words, the density is parameterized implicitly. This
        leads to much better fitting than an explicitly parameterized density.

The fitting is done by gradient descent to maximize the fit of the distribution
to data across the entire session, specifically to classify with a logistic loss
real datapoints from points randomly uniformly sampled in [depth, amplitude]
space. Since the distribution does not depend on time, the motion is pressured
to facilitate this classification, i.e. to stabilize the distribution over time.
"""

import math

import numpy as np
import torch
import tqdm


class MLP(torch.nn.Module):
    """MLP model."""

    def __init__(self, in_features, layer_features, activation=None):
        """Create MLP module.
        Args:
            in_features: Number of features of the input.
            layer_features: Iterable of ints. Output sizes of the layers.
            activation: Activation function. If None, defaults to ReLU.
            bias: Bool. Whether to use bias.
        """
        super(MLP, self).__init__()

        self._in_features = in_features
        self._layer_features = layer_features
        if activation is None:
            activation = torch.nn.ReLU()
        self.activation = activation

        features_list = [in_features] + list(layer_features)
        module_list = []
        for i in range(len(features_list) - 1):
            if i > 0:
                module_list.append(activation)
            layer = torch.nn.Linear(
                in_features=features_list[i], out_features=features_list[i + 1]
            )
            module_list.append(layer)

        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x):
        """Apply MLP to input.

        Args:
            x: Tensor of shape [batch_size, ..., in_features].

        Returns:
            Output of shape [batch_size, ..., self.out_features].
        """
        return self.net(x)

    @property
    def in_features(self):
        return self._in_features

    @property
    def layer_features(self):
        return self._layer_features

    @property
    def out_features(self):
        return self._layer_features[-1]


class Dataset:
    """Dataset that gives samples of peaks."""

    def __init__(self, times, depths, amplitudes):
        """Constructor.

        Args:
            times: Array of times for each peak (in seconds).
            depths: Array of depths for each peak (in microns).
            amplitudes: Array of amplitudes for each peak.
            log_amplitudes: Bool. Whether to take logarithm of amplitudes.
        """
        print("\nConstructing Dataset object")
        self._times = torch.from_numpy(times.astype(np.float32))
        self._num_samples = len(times)
        self._depths_raw = depths
        self._amplitudes_raw = amplitudes

        # Normalize depths and amplitudes to [0, 1]
        depths_normalized, self._depth_range = self._normalize(depths)
        amplitudes_normalized, self._amplitude_range = self._normalize(
            amplitudes
        )
        self._depths_normalized = torch.from_numpy(
            depths_normalized.astype(np.float32)
        )
        self._amplitudes_normalized = torch.from_numpy(
            amplitudes_normalized.astype(np.float32)
        )

    def _normalize(self, x):
        x_range = (np.min(x), np.max(x))
        x_normalized = (x - x_range[0]) / (x_range[1] - x_range[0])
        return x_normalized, x_range

    def sample_real(self, batch_size):
        """Sample batch of real data."""
        indices = np.random.randint(self._num_samples, size=batch_size)
        data = {
            "times": self._times[indices],
            "depths": self._depths_normalized[indices],
            "amplitudes": self._amplitudes_normalized[indices],
        }
        return data

    def sample_fake(self, batch_size, motion_bound):
        """Sample batch of fake data."""
        indices = np.random.randint(self._num_samples, size=batch_size)
        data = {
            "times": self._times[indices],
            "depths": torch.FloatTensor(batch_size).uniform_(
                -motion_bound, 1 + motion_bound
            ),
            "amplitudes": torch.FloatTensor(batch_size).uniform_(0, 1),
        }
        return data

    def sample_uniform(self, time, grid_size):
        """Sample grid of data."""
        amplitudes, depths = np.meshgrid(
            np.linspace(0, 1, grid_size),
            np.linspace(0, 1, grid_size),
        )
        data = {
            "times": time * np.ones(grid_size * grid_size),
            "depths": np.ravel(depths),
            "amplitudes": np.ravel(amplitudes),
        }
        data = {
            key: torch.from_numpy(value.astype(np.float32))
            for key, value in data.items()
        }
        return data

    @property
    def times(self):
        return self._times

    @property
    def times_numpy(self):
        return self.times.numpy()

    @property
    def depths_raw(self):
        return self._depths_raw

    @property
    def amplitudes_raw(self):
        return self._amplitudes_raw

    @property
    def depths_normalized(self):
        return self._depths_normalized

    @property
    def amplitudes_normalized(self):
        return self._amplitudes_normalized

    @property
    def depth_range(self):
        return self._depth_range

    @property
    def amplitude_range(self):
        return self._amplitude_range


class MotionPredictor(torch.nn.Module):
    """MotionPredictor class.

    This class takes in a batch of timestamps and returns a batch of predicted
    motions.
    """

    def __init__(
        self,
        bound_normalized,
        time_range,
        time_bin_size,
        time_kernel_width,
        num_depth_bins=2,
        depth_smoothing=None,
        epsilon=1e-3,
    ):
        """Constructor.

        Args:
            bound_normalized: Scalar. Bound on maximum absolute motion.
            time_range: Tuple (min_time, max_time) for data.
            time_bin_size: Scalar. Discretization of the motion function in
                units of time.
            time_kernel_width: Scalar. Width of the smoothing kernel in units of
                time.
            num_depth_bins: Int. Number of depth bins.
            depth_smoothing: Scalar. Width of depth smoothing.
            epsilon: Small value to prevent division by zero.
        """
        super(MotionPredictor, self).__init__()
        print(
            "\nConstructing MotionPredictor object with parameters:\n"
            f"    bound_normalized = {bound_normalized}\n"
            f"    time_range = {time_range}\n"
            f"    time_bin_size = {time_bin_size}\n"
            f"    time_kernel_width = {time_kernel_width}\n"
            f"    num_depth_bins = {num_depth_bins}\n"
            f"    depth_smoothing = {depth_smoothing}"
        )

        self._bound_normalized = bound_normalized
        self._time_range = time_range
        self._time_bin_size = time_bin_size
        self._num_depth_bins = num_depth_bins
        self._epsilon = epsilon * epsilon
        if depth_smoothing is None:
            depth_smoothing = 1.0 / max(1, num_depth_bins - 1)
        self._depth_smoothing = depth_smoothing + epsilon

        # Construct motion matrix time kernel
        self._num_time_bins = math.ceil(
            (time_range[1] - time_range[0]) / time_bin_size
        )
        self._motion = torch.nn.Parameter(
            torch.zeros(self._num_depth_bins, self._num_time_bins),
            requires_grad=True,
        )

        # Construct depth levels
        self._depth_levels = torch.linspace(0.0, 1.0, num_depth_bins)

        # Construct time kernel
        self._time_kernel = self._get_kernel(time_bin_size, time_kernel_width)

        # Compute kernel applied to ones, which will be used to normalize kernel
        # convolution applications to remove edge effects.
        ones_input = torch.ones((1, 1, self._num_time_bins))
        self._conv_ones = torch.nn.functional.conv1d(
            ones_input, self._time_kernel, padding="same"
        )[0]
        self._tanh = torch.nn.Tanh()

    def _get_kernel(self, time_bin_size, time_kernel_width):
        """Get triangular kernel discretized by time_bin_size.

        Args:
            time_bin_size: Scalar. Width of the kernel in units of time.
            time_kernel_width: Scalar. Width of the kernel in units of time.

        Returns:
            kernel: Torch array of size [1, 1, bins_in_kernel]. Triangular
                kernel, normalized to have unit area.
        """
        if time_kernel_width < time_bin_size:
            print(
                f"time_kernel_width {time_kernel_width} is smaller than "
                f"time_bin_size {time_bin_size}, so rounding up smoothing "
                "kernel to one bin.\n"
            )
        kernel_slope = 0.5 * time_kernel_width / time_bin_size
        half_kernel = np.arange(1.0, 0.0, -1 / kernel_slope)
        kernel = np.concatenate([half_kernel[::-1], half_kernel[1:]])
        kernel /= np.sum(kernel)
        kernel = torch.from_numpy(kernel.astype(np.float32))
        kernel = kernel[None, None]
        return kernel

    def forward(self, times, depths):
        """Run on 1-dimensional tensor containing a batch of times."""
        time_bins = torch.floor(
            (times - self._time_range[0]) / self._time_bin_size
        )
        time_bins = time_bins.type(torch.int64)

        # Shape [num_depth_bins, batch_size]
        pred_motions = self.smooth_motion[:, time_bins]

        # Get coefficients for depth smoothing
        diffs = torch.abs(depths[None] - self._depth_levels[:, None])
        coeffs = torch.nn.functional.relu(self._depth_smoothing - diffs)
        coeffs /= self._epsilon + torch.sum(coeffs, dim=0, keepdim=True)

        # Apply depth smoothing to get shape [batch_size]
        pred_motions = torch.sum(pred_motions * coeffs, dim=0)

        return pred_motions

    @property
    def bound_normalized(self):
        return self._bound_normalized

    @property
    def time_range(self):
        return self._time_range

    @property
    def time_bin_size(self):
        return self._time_bin_size

    @property
    def num_time_bins(self):
        return self._num_time_bins

    @property
    def num_depth_bins(self):
        return self._num_depth_bins

    @property
    def smooth_motion(self):
        """Normalize and smooth self._motion by kernel.

        Returns smooth_motion_normalized of shape
        [self.num_time_bins, self.num_depth_bins].
        """
        # Apply nonlinearity and smoothing
        motion = self._bound_normalized * self._tanh(self._motion)
        smooth_motion = torch.nn.functional.conv1d(
            motion[:, None], self._time_kernel, padding="same"
        )[:, 0]
        smooth_motion_normalized = smooth_motion / self._conv_ones

        # Center each motion vector. We detach the mean in order to prevent the
        # model intentionally expanding dense timepoints because of the motion
        # noise.
        smooth_motion_normalized -= torch.mean(
            smooth_motion_normalized, axis=1, keepdims=True
        ).detach()

        return smooth_motion_normalized


class DistributionPredictor(torch.nn.Module):
    """DistributionPredictor class.

    This class takes in a batch of depths and amplitudes and returns an
    un-normalized probability for each of them occuring in the real dataset.
    """

    def __init__(self, hidden_features=(256, 256), activation=None):
        """Constructor."""
        super(DistributionPredictor, self).__init__()
        print(
            "\nConstructing DistributionPredictor object with parameters:\n"
            f"    hidden_features = {hidden_features}\n"
            f"    activation = {activation}"
        )

        self._feature_frequencies = [1, 2, 4, 8, 16, 32]
        in_features = 2 * (1 + len(self._feature_frequencies))
        self._net = MLP(
            in_features=in_features,
            layer_features=list(hidden_features) + [1],
            activation=activation,
        )
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, depths, amplitudes):
        """Compute likelihood in [0, 1] for depths and amplitudes."""
        features_depths = [depths] + [
            torch.sin(f * depths) for f in self._feature_frequencies
        ]
        features_amplitudes = [amplitudes] + [
            torch.sin(f * amplitudes) for f in self._feature_frequencies
        ]
        net_inputs = torch.stack(features_depths + features_amplitudes, axis=1)
        net_outputs = self._net(net_inputs)[:, 0]
        pred_distrib = self._sigmoid(net_outputs)
        return pred_distrib


class MotionCorrector(torch.nn.Module):
    """MotionCorrector class."""

    def __init__(self, motion_predictor, distribution_predictor, epsilon=1e-4):
        """Constructor.

        Args:
            motion_predictor: MotionPredictor object. Callable, takes in a batch
                of times and returns a batch of predicted motions.
            distribution_predictor: DistributionPredictor object. Callable,
                takes in a batch of depths and a batch of amplitudes and
                returns a batch of likelihoods.
            epsilon: Small value to prevent logarithms from exploding.
        """
        super(MotionCorrector, self).__init__()
        print("\nConstructing MotionCorrector object")

        self.add_module("motion_predictor", motion_predictor)
        self.add_module("distribution_predictor", distribution_predictor)
        self._epsilon = epsilon

    def forward(self, data_batch, motion_noise=0.0):
        """Predict likelihood for each sample in data_batch.

        Args:
            data_batch: Dictionary with keys 'times', 'depths', and
                'amplitudes'.

        Returns:
            pred_distrib: Torch array of shape [batch_size]. Values in [0, 1].
        """
        times = data_batch["times"]
        depths = data_batch["depths"]
        amplitudes = data_batch["amplitudes"]
        pred_motion = self.motion_predictor(times, depths)
        pred_motion += motion_noise * torch.randn_like(pred_motion)
        pred_depths = depths + pred_motion
        pred_distrib = self.distribution_predictor(pred_depths, amplitudes)

        return pred_distrib

    def loss(self, data_real, data_fake, motion_noise=0.0):
        """Compute loss on a batch of real and fake data.

        Args:
            data_real: Dictionary with keys 'times', 'depths', and
                'amplitudes'.
            data_fake: Dictionary with keys 'time16s', 'depths', and
                'amplitudes'.

        Returns:
            loss: Torch scalar. Loss on the batch.
        """
        pred_distrib_real = self.forward(data_real, motion_noise=motion_noise)
        pred_distrib_fake = self.forward(data_fake, motion_noise=motion_noise)

        # Logistic loss to pressure pred_distribution_real towards 1 and
        # pred_distribution_fake towards 0
        loss = -1 * (
            torch.mean(torch.log(self._epsilon + pred_distrib_real))
            + torch.mean(torch.log(self._epsilon + 1 - pred_distrib_fake))
        )

        return loss


class Trainer:

    def __init__(
        self,
        dataset,
        motion_corrector,
        batch_size,
        training_steps,
        initial_motion_noise=0.1,
        motion_noise_steps=2000,
        optimizer=torch.optim.Adam,
        learning_rate=0.001,
        grad_clip=1,
    ):
        """Constructor."""
        self._dataset = dataset
        self._motion_corrector = motion_corrector
        self._batch_size = batch_size
        self._training_steps = training_steps
        self._initial_motion_noise = initial_motion_noise
        self._motion_noise_steps = motion_noise_steps
        self._learning_rate = learning_rate
        self._grad_clip = grad_clip

        self._optimizer = optimizer(
            self._motion_corrector.parameters(),
            lr=learning_rate,
        )
        self._losses = None

    def __call__(self):
        """Evaluate model on dataset."""
        print("\nFitting motion correction")

        motion_bound = self._motion_corrector.motion_predictor.bound_normalized
        training_losses = []
        for step in tqdm.tqdm(range(self._training_steps)):

            # Compute motion noise
            remaining = (
                self._motion_noise_steps - step
            ) / self._motion_noise_steps
            motion_noise = max(remaining * self._initial_motion_noise, 0)

            # Compute loss and gradients
            self._optimizer.zero_grad()
            data_real = self._dataset.sample_real(batch_size=self._batch_size)
            data_fake = self._dataset.sample_fake(
                batch_size=self._batch_size, motion_bound=motion_bound
            )
            loss = self._motion_corrector.loss(
                data_real, data_fake, motion_noise=motion_noise
            )
            loss.backward()

            # Clip gradients and step optimizer
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._motion_corrector.parameters(), self._grad_clip
                )
            self._optimizer.step()
            training_losses.append(float(loss.detach()))

        self._losses = training_losses
        print("\nFinished fitting motion correction")

    @property
    def motion_corrector(self):
        return self._motion_corrector

    @property
    def dataset(self):
        return self._dataset

    @property
    def losses(self):
        return self._losses
