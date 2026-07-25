"""Installation script for the 'humanoid_amp_mjlab' Python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    # mujoco-warp 3.5.0 still imports mjENBL_MULTICCD, which was removed in MuJoCo 3.8+.
    "mujoco==3.5.0",
    "mujoco-warp==3.5.0",
    "scipy",
]

# Installation operation
setup(
    name="humanoid_amp_mjlab",
    packages=find_packages(include=["src", "src.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
