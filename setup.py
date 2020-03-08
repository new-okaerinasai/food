from setuptools import setup, find_packages

setup(
    name="food",
    version="0.1.0",
    url="https://github.com/new-okaerinasai/food",
    author="Ruslan Khaidurov, Sonya Dymchenko, Angelina Yaroshenko, Dmitry Vypirailenko",
    author_email="rakhaydurov@edu.hse.ru",
    python_requires=">=3.6.0",
    package_dir={"": "src"},
    packages=find_packages("./src"),
    long_description=open('README.md').read(),
)