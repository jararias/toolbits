import sys
import setuptools

find_packages = setuptools.find_packages
if sys.version_info.major > 2:
    find_packages = setuptools.find_namespace_packages

setuptools.setup(
    name='toolbits',
    version='0.1',
    author='Jose A Ruiz-Arias',
    author_email='jararias at uma.es',
    url='',
    description='A variety of tools',
    packages=find_packages(where='toolbits'),
)
