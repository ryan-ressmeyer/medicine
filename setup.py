"""Installation script."""

import os

from setuptools import find_packages, setup

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="medicine-neuro",
    version="1.0",
    description=(
        "MEDICINE: Motion Estimation by DIstributional Contrastive Inference "
        "for NEurophysiology."
    ),
    author="Nicholas Watters",
    url="https://github.com/jazlab/medicine.github.io",
    license="MIT license",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "neuroscience",
        "neurophysiology",
        "spike sorting",
        "machine learning",
        "motion",
        "python",
    ],
    packages=(
        ["medicine_demos", "medicine"]
        + ["medicine_demos." + x for x in find_packages("medicine_demos")]
        + ["medicine." + x for x in find_packages("medicine")]
    ),
    install_requires=[
        "torch==2.2.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
)
