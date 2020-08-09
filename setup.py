# from distutils.core import setup
from os import path
from setuptools import setup, Extension

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'tinn',         
    packages = ['tinn'],   
    version = '2.01',      
    license='MIT',     
    description = 'A light weight simple, multi layer ,feedforward neural network library',   
    long_description=long_description,
    long_description_content_type='text/markdown',
    author = 'Shashank Sahu',        
    author_email = 'shashankcs083@gmail.com',    
    url = 'https://github.com/bit-shashank/tinn', 
    download_url = 'https://github.com/bit-shashank/tinn/archive/v0.2-alpha.tar.gz',  
    keywords = ['Neural', 'Deep', 'Learning' ,'Machine','Network'],   
    install_requires=[          
            'numpy',
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',     
        'Intended Audience :: Developers',      
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3',     
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)