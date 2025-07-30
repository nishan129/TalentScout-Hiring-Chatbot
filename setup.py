from setuptools import setup, find_packages
from typing import List



def get_requirements()-> List[str]:
    """This function will return list of requirements"""
    requiremnts_list:List[str] = []
    
    try:
        # Open and read the requirements.txt file
        with open("requiremnts.txt", "r") as file:
            # Read lines from the file 
            lines = file.readlines()
           # Process each line
            for line in lines:
                # Strip whitespace and newline characters
                requirements = line.strip()
                 # Ingnore_empty lines and -e.
                if requirements and requirements != "-e .":
                    requiremnts_list.append(requirements)
    except FileNotFoundError:
         print("requirements.txt file not found.")
            
    return requiremnts_list

setup(
    name="TalentScout Hiring Chatbot",
    version= "0.0.1",
    author= "Nishant Borkar",
    author_email= "nishantborkar139@gmail.com",
    description="TalentScout Hiring Chatbot ",
    packages=find_packages(),
    install_requires=get_requirements()
)