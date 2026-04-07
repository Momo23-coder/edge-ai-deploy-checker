from setuptools import setup, find_packages

setup(
    name="edge-ai-deploy-checker",
    version="0.1.0",
    author="Moses Thlama James",
    author_email="mosesjamesthlama@gmail.com",
    description="Check if an AI model is ready to deploy on edge hardware",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Momo23-coder/edge-ai-deploy-checker",
    packages=find_packages(),
    package_data={"": ["devices/*.json"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "tflite": ["tensorflow>=2.10"],
        "onnx": ["onnx>=1.12"],
        "all": ["tensorflow>=2.10", "onnx>=1.12"],
    },
    entry_points={
        "console_scripts": [
            "edge-check=edge_check.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
