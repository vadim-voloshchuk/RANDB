from setuptools import setup, find_packages

setup(
    name="redandblue",
    version="0.0.1",
    packages=find_packages(include=['redandblue*']),
    install_requires=["gymnasium", "pygame"],
)
