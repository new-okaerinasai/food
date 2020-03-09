from setuptools import setup, find_packages

setup(
    name="food-python",
    version="0.1.0",
    url="https://github.com/new-okaerinasai/food",
    author="Ruslan Khaidurov, Sonya Dymchenko, Angelina Yaroshenko, Dmitry Vypirailenko",
    author_email="rakhaydurov@edu.hse.ru",
    python_requires=">=3.6.0",
    package_dir={"": "src"},
    packages=find_packages("./src"),
    description="A framework for out-of-distribution and anomaly detection",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    py_modules=["food"]
)
