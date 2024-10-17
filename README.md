# MEDICINE: Motion Estimation by DIstributional Contrastive Inference for NEurophysioloy

MEDICINE is a method for estimating motion in neurophysiology data for spike
sorting. See our [publication](https://) for a complete description of the
method and results.

## Introduction

The general idea of MEDICINE is to decompose neural activity data into two
components:
* The **motion** of the brain relative to a probe in depth.
* An **activity distribution** of the brain along the probe. These two
components are jointly optimized via gradient descent to maximize the likelihood
of a dataset of detected spikes extracted from a neural recording session. Here
is a video of this optimization process in action:

<img src="graphics/model_fitting.gif" width="900">

The red curves on the left show the motion learned by the model, the heatmap on
the right show the activity distribution learned by the model and the
scatterplots show detected spikes (colored by amplitude).

<img src="graphics/model_schematic.jpg" width="900">

## Usage

### Getting Started

We recomend using a virtual environment (e.g. conda or pipenv) to manage dependencies. Once in a virtual environment, install MEDICINE with:
```
pip install medicine-neuro
```
This will also install the necessary dependencies.

Then you can run the demo script with
```
python -m medicine_demos.run_demo
```
This will run the [`demo script`](medicine_demos/run_demo.py) and display several figures showing the results. See [medicine_demos/run_demo.py](medicine_demos/run_demo.py) for more details.

### Hyperparameters

### SpikeInterface Integration

### Kilosort Integration

# Contact and Support

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for information about support.
Please email Nick Watters at nwatters@mit.edu with questions and feedback.

# Reference

If you use MEDICINE or a derivative of it in your work, please cite it as
follows:

```
@misc{watters2024,
author = {Nicholas Watters and Mehrdad Jazayeri},
title = {MEDICINE: Motion Estimation by DIstributional Contrastive Inference for NEurophysiology},
url = {https://arxiv.org/},
journal = {arXiv preprint arXiv:},
year = {2024}
}
```

# MEDICINE Website

The [MEDICINE website](https://jazlab.github.io/medicine.github.io/) is a
[GitHub Pages](https://pages.github.com/) website with a [Slate
theme](https://github.com/pages-themes/slate). The website is generated from
this [`README.md`](README.md) with the settings in [`_config.yml`](_config.yml) and
the Ruby dependencies in [`Gemfile`](Gemfile).

If you would like to modify the website, first make sure you can test deploying
it locally by following the [GitHub Pages testing
instructions](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll).
Then modify this [`README.md`](README.md) and test deploy to view the changes
before committing.
