# Issue 1: "QtWidgets/QOpenGLWidget: No such file or directory"

From [here](https://github.com/openai/roboschool/issues/42)

"Installing Qt from runfile, [Qt Runfile Installation](https://wiki.qt.io/Install_Qt_5_on_Ubuntu) and setting the PKG_CONFIG_PATH."

For instance, following the installation of Qt 5.9.1:

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/:/PATH_TO_QT_ROOT_DIR/5.9.1/gcc_64/lib/pkgconfig
```

# Issue 2: "libQt5Core.so: undefined reference to ..."

Mainly from [here](https://github.com/openai/roboschool/issues/42) again.

## Step 1: 

Change line 50-54 of the ~/roboschool/cpp_household/Makefile with the following.

```bash
INC     += `$(PKG) --cflags Qt5Widgets Qt5OpenGL assimp bullet`
LIBS    += -lstdc++ `$(PKG) --libs Qt5OpenGL Qt5Widgets assimp bullet`
INC     += -I~/.forked_bullet/include -I~/.forked_bullet/include/bullet -I/usr/local/include/bullet -I/usr/include/python3.6m
LIBS    += $(RPATH) -L~/.forked_bullet/lib -lLinearMath -lBullet3Common -lBulletCollision -lBulletDynamics -lBulletInverseDynamics -lPhysicsClientC_API -lBulletSoftBody
```

## Step 2:

```bash
export  LD_LIBRARY_PATH=/PATH_TO_QT_ROOT_DIR/5.9.1/gcc_64/lib:$LD_LIBRARY_PATH
```

# Issue 3: 

At compilation time, upon the following error:

```bash 
.build-debug/render-ssao.o: In function 'SimpleRender::ContextViewport::_depthlinear_paint(int)'
cpp-household/render-ssao.cpp:75: undefined reference to `glBindMultiTextureEXT'
```

The issue is solved by exporting the following:
```bash
export ROBOSCHOOL_DISABLE_HARDWARE_RENDER=1
```

or apparently it is not sufficient...

# Issue 4: 

Upon the following error, while trying to render an environment:

```bash
QGLShaderProgram: could not create shader program
Could not create shader of type 2.
python: render-simple.cpp:250: void SimpleRender::Context::initGL(): Assertion `r0' failed.
Aborted (core dumped)
```

from [here](https://github.com/openai/roboschool/issues/15) :

"It seems this is an existing issue with the Nvidia drivers for Linux distributions - https://bugs.launchpad.net/ubuntu/+source/python-qt4/+bug/941826.

In summary, the problem comes from when python dynamically loads the required OpenGL libraries. It loads the Messa GL library as opposed to the Nvidia driver one unless PyOpenGL is loaded first.

A workaround is to import GL as so `from OpenGL import GL` within a zoo file."

# Issue 5:

Upon the following error, while trying to install boost for a virtualenv/pipenv:

```
./boost/python/detail/wrap_python.hpp:50:23: fatal error: pyconfig.h: No such file or directory
 # include <pyconfig.h>
```

One solution/tweak consists of exporting a environment variable where pyconfig.h sits:

```
export CPLUS_INCLUDE_PATH=/usr/include/python3.5m
```

# Issue 6:

Upon the following erro, while trying to compile roboschool/cpp_household:

```

```

One solution consist of changin line 8 of the Makefile with:

```
LIBS =-L/usr/lib64 -lm -lGL -L$(HOME)/.boost/lib -L$(HOME)/.forked_bullet/lib
```

# Issue 7:

Upon re-entering the virtualenv that have been previously configure, it is required to export the two environment variables specified in Issue 2 above.
Otherwise this issue occurs when trying to import roboschool:

```
>>> import roboschool
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/kevin/Development/git/roboschool/roboschool/__init__.py", line 127, in <module>
    from roboschool.gym_pendulums import RoboschoolInvertedPendulum
  File "/home/kevin/Development/git/roboschool/roboschool/gym_pendulums.py", line 1, in <module>
    from roboschool.scene_abstract import SingleRobotEmptyScene
  File "/home/kevin/Development/git/roboschool/roboschool/scene_abstract.py", line 12, in <module>
    from roboschool  import cpp_household   as cpp_household
ImportError: /home/kevin/Development/git/roboschool/roboschool/cpp_household.so: symbol _ZTI13QOpenGLWidget, version Qt_5 not defined in file libQt5Widgets.so.5 with link time reference
```