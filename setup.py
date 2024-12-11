from setuptools import find_packages, setup 

with open('package/Introduction.md', 'r', encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = 'DiscoEPG',
    version = '0.0.22',
    description = "Open-source package for insects EPG data analysis",
    package_dir = {"":"package"},
    packages= find_packages(where = "package"),
    install_requires=required,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = "https://github.com/HySonLab/ML4Insects",
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