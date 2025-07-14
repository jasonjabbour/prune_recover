from setuptools import setup, find_packages

setup(
    name="prune_recover",
    version="0.1.0",
    description="A VLA prune and recover package",
    author="jasonjabbour",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)