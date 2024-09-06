from setuptools import find_packages, setup 

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name = 'PyEPG',
    version = '0.0.1',
    description = "Open-source package for insects EPG data analysis",
    package_dir = {"":"PyEPG"},
    packages= find_packages(where = "PyEPG"),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = "https://github.com/HySonLab/PyEPG",
    author = "DINHQuangDung1999",
    author_email = "dqdung999@gmail.com",
    license="MIT",
    classifiers= [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent" 
    ],
    extra_requires = {
        "twine >= 4.0.2"
    },
    python_requires = ">=3.10"
)