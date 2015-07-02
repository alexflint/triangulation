#!/usr/bin/env python
from distutils.core import setup

setup(name='triangulation',
      description='3D triangulation algorithms',
      version='0.1',
      author='Alex Flint',
      author_email='alex.flint@gmail.com',
      url='...',
      packages=['triangulation'],
      package_dir={'triangulation': '.'},
      scripts=['cmds/evaluate_algorithm', 'cmds/triangulate_dataset']
      )
