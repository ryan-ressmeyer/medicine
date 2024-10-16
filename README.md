# Motion Estimation by DIstributional Contrastive Inference for NEurophysioloy (MEDICINE)

Motion Estimation by DIstributional Contrastive Inference for NEurophysioloy
(MEDICINE) is a method for correcting motion in neurophysiology data.

## Introduction

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

## Installation

Please use a virtual environment before installing.

This library requires the following dependencies:
```
pip install spikeinterface
pip install torch
```

## Usage

See the notebook `run_motion_correction.ipynb` for example usage.

## Distribution

This repo is intended for use only within JazLab and is not intended for
distribution. If you have questions about this, please talk with Nick.
