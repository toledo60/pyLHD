import setuptools

with open("README.md",'r') as f:
    long_description = f.read()

setuptools.setup(
    name="pyLHD",
    version="0.4.8",
    author="Jose Toledo",
    author_email="toledo60@protonmail.com",
    description="Latin Hypercube Designs for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy'],
    url="https://github.com/toledo60/pyLHD",
    project_urls={
        "Bug Tracker": "https://github.com/toledo60/pyLHD/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
