
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r", encoding='utf-16') as fh:
    req = fh.readlines()
    requirements = []
    for line in req:
        requirements.append(line.replace("\n", "")) # \ufeff

setuptools.setup(
    name='crypt_utilities',  
    version='0.0.1',
    author="Pablo García Mesa",
    author_email="pgmesa.sm@gmail.com",
    description="Dataset utility package for UPM OCT study",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgmesa-upm/upm_oct_dataset_utils",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
 )