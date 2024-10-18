from setuptools import setup, find_packages

setup(
    name="nanoclip",
    version="0.1.0",
    packages=find_packages(),
    author="Tom Pollak",
    author_email="tompollak1000@gmail.com",
    description="Minimal CLIP / ViT training and inference.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tom-pollak/nanovit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

