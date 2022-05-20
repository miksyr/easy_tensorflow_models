from setuptools import setup

setup(
    name="easy_tensorflow_models",
    packages=["easy_tensorflow_models"],
    version="0.1.3",
    description="Helpers for creating, training, and serving TF models",
    author="Michael Doran",
    author_email="mikrdoran@gmail.com",
    url="https://github.com/miksyr/easy_tensorflow_models",
    keywords=["tensorflow"],
    install_requires=["tensorflow==2.7.1", "tqdm>=4.38.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
