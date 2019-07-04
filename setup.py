import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="baNetLearn",
    version="0.1",
    author="Carlos Rodolfo Huerta Santiago",
    author_email="carlos.huertaso@udlap.mx",
    description="A python package that implements dynamic bayesian networks models on time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charx7/DynamicBayesianNetworkds",
    #packages=setuptools.find_packages(),
    packages=setuptools.find_packages('src'), # Alternative to tell to find directoly on src
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
