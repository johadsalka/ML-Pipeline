from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]
        
        if HYPEN_E_DOT in requirements:
          requirements.remove(HYPEN_E_DOT)


setup(name='ML-PIPELINE',
      version='0.0.1',
      description='Machine Learning Pipeline Project',
      author='Soham Rane',
      author_email='rane.soham6147@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements("requirements.txt")
     )

