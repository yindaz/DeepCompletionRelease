This directory contains all code for the GAPS software library
required for the depth2depth application.
There are several subdirectories:

    pkgs - source and include files for all packages (software libraries).
    apps - source files for the depth2depth application.
    makefiles - unix-style make file definitions
    lib - archive library (.lib) files (created during compilation).
    bin - executable files (created during compilation).

If you are using linux or cygwin and have gcc and OpenGL development
libraries installed, or if you are using MAC OS X with the xcode
development environment, you should be able to compile all the code by
typing "make clean; make" in this directory.  For other development
platforms, you should edit the shared compilation settings in the
makefiles/Makefiles.std to meet your needs.

The software is distributed under the GNU General Public License.



