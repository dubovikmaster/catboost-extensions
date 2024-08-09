from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='catboost_extensions',
    version='1.0',
    python_requires='>=3.7',
    packages=find_packages(),
    author='Dubovik Pavel',
    author_email='geometryk@gmail.com',
    description='Extensions for catboost models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dubovikmaster/catboost-extensions',
    install_requires=[
        'scikit-learn',
        'tqdm',
        'requests',
        'shap',
        'catboost',
        'optuna'
    ],
    platforms='any'
)
