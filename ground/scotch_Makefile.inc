prefix		= $(SCROOT)

EXE		=
LIB		= .a
OBJ		= .o

MAKE		= make
AR		= ar
ARFLAGS		= -ruv
CAT		= cat
CCS		= gcc
CCP		= mpicc
CCD		= gcc
CFLAGS		= -fPIC -O3 -DCOMMON_FILE_COMPRESS_GZ -DCOMMON_RANDOM_FIXED_SEED -DSCOTCH_RENAME -DSCOTCH_RENAME_PARSER -Drestrict=__restrict -DIDXSIZE32 -DINTSIZE32
CLIBFLAGS	=
LDFLAGS		= -lz -lm -lrt
CP		= cp
LEX		= flex -Pscotchyy -olex.yy.c
LN		= ln
MKDIR		= mkdir
MV		= mv
RANLIB		= ranlib
YACC		= bison -pscotchyy -y -b y
