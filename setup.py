from setuptools import setup, find_namespace_packages

setup(
    name='latalg',
    packages=find_namespace_packages(include=['latalg.*', 'lib.*']),
    version='0.1',
    install_requires=[
        'click',
        'colored_traceback',
        'colorama',
        'plotly',
        'matplotlib',
        'tqdm',
        'dacite',

        'torch',
        'torchvision',
        'lightning-bolts',
        'lightning',

        'jaxtyping<=0.2.23',
        'beartype',
        'einops',
        'wandb',

        'numpy',
        'scipy',
        'scikit-learn',
        'pdbpp',

        'siren-pytorch',
        'h5py',
        'boolean.py',
        'tabulate'
    ])
