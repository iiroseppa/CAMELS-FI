from setuptools import setup, find_packages

setup(
    name='CAMELS-FI',  
    version='1.0.0',  
    author='Iiro Seppä',
    author_email='iielse@utu.fi', 
    description='Scripts used for CAMELS-FI, utils are a module',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD-3-Clause',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Minimum Python version required
    install_requires=[  # List of dependencies
        # 'some-dependency>=1.0.0',
    ],
)
