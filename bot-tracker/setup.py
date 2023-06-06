#!/usr/bin/env python3

from setuptools import find_packages, setup
from typing import Optional, Union
import importlib
import pkg_resources
import os
import glob
import warnings

package_name = "bot-tracker"

# Pytorch will only be added if it is not already installed
# minimum pyotrch version
torch_min_version = "1.2.0"
torch_version = "1.8.2"
torchvision_version = "0.9.2"


# we need extra anyvison builds here as some cuda versions are missing in offical release
pytorch_options = {
    None: [
        f"torch=={torch_version}+cpu",
        f"torchvision=={torchvision_version}+cpu",
    ],
    # cuda 10.2 and 11.0 both use cu102
    "10.2": [
        f"torch=={torch_version}+cu102",
        f"torchvision=={torchvision_version}+cu102",
    ],
    "11.0": [
        f"torch=={torch_version}+cu102",
        f"torchvision=={torchvision_version}+cu102",
    ],
    # cuda version >= 11.1 all use cuda 11.1 wheel
    "11.x": [
        f"torch=={torch_version}+cu111",
        f"torchvision=={torchvision_version}+cu111",
    ],
}


def get_cuda_version() -> Union[str, None]:
    """Get cuda version if it is installed.
    Normally cuda is installed at /usr/local/cuda.
    details can be found here:

    https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html#ubuntu-x86_64-run

    Returns
    -------
    Union[str, None]
        Returns version of cuda(eg, 9.0, 10.2) if cuda is found.
    """

    cuda_home = os.getenv("CUDA_HOME")

    if cuda_home is None:
        cuda_path = os.popen("whereis cuda").read()[6:-1]
        if len(cuda_path) < 4:
            cuda_path = "/usr/local/cuda"
    else:
        cuda_path = cuda_home

    cuda_exists = os.path.exists(cuda_path)

    cuda_runtime_file = os.path.join(cuda_path, "lib64/libcudart.so.*")

    if cuda_exists:
        # check if libcudart.so* exists
        cuda_runtime_file_getter = glob.glob(cuda_runtime_file)

        if not cuda_runtime_file_getter:
            warnings.warn(
                "CUDA libcudart.so* not found at CUDA_HOME, fall back to system path",
                stacklevel=1,
            )

            cuda_runtime_file = "/usr/lib/x86_64-linux-gnu/libcudart.so*"
            cuda_runtime_file_getter = glob.glob(cuda_runtime_file)

            if not cuda_runtime_file_getter:
                raise Exception(
                    "libcudart.so* could not be found, please report a bug to https://github.com/AnyVisionltd/ai-module-builder/issues"
                )

        ver_string = os.popen(
            f'ls {cuda_runtime_file} | sort | tac | head -1 | rev | cut -d"." -f -3 | rev | cut -f1,2 -d"."'
        ).read()
        print("CUDA version :" + ver_string)
        return ver_string.strip()
    else:
        return None


def get_pytorch_package_name(cuda_version: Optional[str]) -> Union[list, None]:
    """Get pytorch wheel name with given cuda version.
    Parameters
    ----------
    cuda_version
        cuda version string, eg, 10.2, 11.0, or None
    Returns
    -------
    Union[list,None]
        Returns package name of cupy.
    """

    pytorch_spec = importlib.util.find_spec("torch")

    # prefer not updating pytorch
    if pytorch_spec is not None:
        current_version = pkg_resources.get_distribution("torch").version
        if pkg_resources.parse_version(current_version) >= pkg_resources.parse_version(
            torch_min_version
        ):
            return None
        else:
            print(f"Pytorch version {current_version} is too old, will be updated...")

    # if cuda version >= 11.1(!= 11.0), change it to 11.x
    if cuda_version is not None and cuda_version.startswith("11."):
        if cuda_version != "11.0":
            cuda_version = "11.x"

    torch_wheel = pytorch_options.get(cuda_version, None)

    if torch_wheel is None:
        raise RuntimeError(f"CUDA version {cuda_version} is not supported by Pytorch.")

    return torch_wheel


if __name__ == "__main__":
    # the dependency list
    dependency_list = [
        "cython",
        "numpy",
        "scipy",
        "lap",
        "motmetrics",
        "opencv-python",
        "cython_bbox==0.1.13",
        "matplotlib",
    ]

    # Check if CUDA is supported
    cuda_version = get_cuda_version()

    # get pytorch version
    torch_wheels = get_pytorch_package_name(cuda_version)

    if torch_wheels is not None:
        dependency_list = dependency_list + torch_wheels

    # final setup
    setup(
        name=package_name,
        version="0.2.0",
        python_requires=">=3.8",
        packages=find_packages(exclude=["tests*"]),
        include_package_data=True,
        description="Bot Tracker",
        long_description="BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
        install_requires=dependency_list,
        author="OostoLTD",
        author_email="xuang@oosto.com",
    )
