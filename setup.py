try:
    from setuptools import setup #enables develop
except ImportError:
    from distutils.core import setup

setup(name='gh_rsom',
      version='0.0.1',
      description='Growing Hierarchical Recurrent Self-Organizing Map.',
      author='Karthik kumar Santhanaraj',
      author_email='karthikkumar.s@protonmail.com',
      license='MIT',
      url='https://github.com/kar7hik/gh-rsom',
      packages=['gh_rsom'],
      install_requires=[
          'numpy',
          'matplotlib',
          'tqdm',
      ]
    )
