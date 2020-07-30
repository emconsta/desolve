import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="desolver-emconsta", # Replace with your own username
    version="0.0.1",
    author="Emil Constantinescu",
    author_email="emconsta101@gmail.com",
    description="Differential equations solver.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/desolver",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
