import setuptools

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

setuptools.setup(
    name="traffic_ca",
    version="0.1",
    description="Modelling traffic with cellular automata",
    author="Nadir Bašić",
    install_requires=reqs,
    author_email="basicnadir@gmail.com",
    packages=setuptools.find_packages(),
    zip_safe=False,
)
