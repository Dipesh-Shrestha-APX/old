# Placeholder file
# setup.py
from setuptools import setup, find_packages

setup(
    name="vits-nepali",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Enterprise-grade VITS model for Nepali text-to-speech",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)