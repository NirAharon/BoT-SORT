#!/usr/bin/env python3

from setuptools import find_packages, setup

package_name = "bot-tracker"

if __name__ == "__main__":

    # the dependency list
    dependency_list = [
        "cython",
        "numpy",
        "scipy",
        "motmetrics",
        "cython_bbox",
        "matplotlib",
    ]

    # final setup
    setup(
        name=package_name,
        version='0.1.0',
        python_requires=">=3.8",
        packages=find_packages(exclude=['tests*']),
        include_package_data=True,
        description="Bot Tracker",
        long_description="BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
        install_requires=dependency_list,
        author="OostoLTD",
        author_email="xuang@oosto.com",
    )
