from setuptools import setup, find_packages
from pathlib import Path

with open('requirements.txt', 'r') as f:
    dependencies = [l.strip() for l in f]

setup(
   name='sigmafnl',
   version='0.1.0',
   author='',
   author_email='',
   packages=['sigmafnl', 'sigmafnl.architecture', 'sigmafnl.architecture.bimodal', 'sigmafnl.architecture.latin_hypercubes'],
   description='',
   install_requires=dependencies,
)
