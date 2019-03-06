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

# Issue 3: ".build-debug/render-ssao.o: In function 'SimpleRender::ContextViewport::_depthlinear_paint(int)' "

Followed by: "cpp-household/render-ssao.cpp:75: undefined reference to `glBindMultiTextureEXT'"

The issue is solved by exporting the following:

```bash
export ROBOSCHOOL_DISABLE_HARDWARE_RENDER=1
```

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

