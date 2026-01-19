from setuptools import find_packages, setup

setup(
    name='experimentalis',
    packages=find_packages(include=['experimentalis']),
    version='0.0.1',
    description='A loose collection of functions and things developed for analyzing experimental data, including basic extensible modeling and data management.',
    author='mm',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='test',
)
