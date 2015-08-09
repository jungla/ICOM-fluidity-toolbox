#!~/source/ParaView-3.12.0-Linux-x86_64/bin/pvpython

from paraview.simple import *

data = = XMLPartitionedUnstructuredGridReader( FileName=['/tamay/mensa/fluidity/mli_250m_4/mli_0.pvtu'])

AnimationScene2 = GetAnimationScene()
