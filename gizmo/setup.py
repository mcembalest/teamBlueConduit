#!/usr/bin/env python3
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gizmo",
    version="0.0.1",
    author="Alex Chojnacki",
    author_email="ac@blueconduit.com",
    description="A utility library for machine learning at BlueConduit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BlueCondiut/gizmo",
    packages=setuptools.find_packages(),
    # NOTE Packages for gizmo come from service_line_pipeline env
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.7",
)
