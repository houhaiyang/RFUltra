from setuptools import setup, find_packages
import datetime

now = datetime.datetime.now()
created=now.strftime('%Y-%m-%d %H:%M:%S')

setup(
    name='randomForestUltra',
    version='0.3.0',
    python_requires='>=3.7',
    author='Haiyang Hou',
    author_email='2868582991@qq.com',
    description=f'{created}It supports multi-objective variable and multi-fold random forest, and can calculate P value through random permutation.',
    keywords='random forest, multi-fold, multi-objective, P value',
    packages=find_packages(),
    install_requires=['numpy','pandas','tqdm','seaborn','matplotlib', 'scipy', 'scikit-learn']
)

# 'scikit-learn'
