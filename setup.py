"""
VisionInspect AI — Package setup
"""
from setuptools import setup, find_packages

setup(
    name="visioninspect-ai",
    version="0.1.0",
    author="AMD Project Team",
    description="Industrial Defect Detection with Limited Training Data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.2",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "visioninspect-train=train:main",
            "visioninspect-eval=evaluate:main",
            "visioninspect-inspect=run_inspection:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
