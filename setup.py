from setuptools import setup, find_packages
import os

# Read the dependencies from requirements.txt
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_file, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="your_package_name",
    version="0.1.0",
    package_dir={"": "src"},  # Specify 'src' as the root directory for packages
    packages=find_packages(where="src"),  # Find packages inside 'src'
    install_requires=read_requirements(),
    author="ThomasNgl",
    author_email="thomas.negrello@outlook.com",
    description="A package for variational autoencoders using PyTorch and PyTorch Geometric",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
