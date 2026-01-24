from setuptools import find_packages,setup
from typing import List

Hyphen_e_dot='-e .'
def get_requirements(file_path:str)->List[str]:
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if Hyphen_e_dot in requirements:
            requirements.remove(Hyphen_e_dot)
    return requirements
setup(
    name="Bigmart_Stores_Sales_Prediction",
    version="1.0",
    author="Abhirup Ghosh",
    author_email="abhirup9799@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)