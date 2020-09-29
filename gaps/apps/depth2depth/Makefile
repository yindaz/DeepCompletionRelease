#
# Application name 
#

NAME=depth2depth



#
# With Ceres
# 

#USER_LIBS=-L/usr/local/lib -lceres -lccolamd -lcamd -lcolamd -lamd -Wl,-Bdynamic -lcxsparse -lccolamd -lcamd -lcolamd -lcholmod -lamd -llapack -lopenblas -Wl,-Bstatic -lcxsparse -Wl,-Bdynamic -lgomp -lglog -lgflags -lgfortran -lhdf5 -ldl -lm

#USER_LIBS=-L/usr/local/lib -lceres -lgflags -lglog -lcamd -lcholmod -lccolamd -lcolamd -lcamd -lamd -lmetis -lcxsparse -lsuitesparseconfig -fopenmp -lpthread -lgomp -llapack -lopenblas -lgfortran -lhdf5 -ldl -lm



#
# Without Ceres
#

USER_LIBS=-lhdf5_serial
INC=-I/usr/include/hdf5/serial
#LIB=-L/

USER_CFLAGS=-DRN_USE_CSPARSE 



#
# Source files
#

CCSRCS=$(NAME).cpp 



#
# Libraries
#

PKG_LIBS=-lR2Shapes -lRNMath -lRNBasics -ljpeg -lpng -lCSparse



#
# Include standard makefile
#

include ../../makefiles/Makefile.apps















