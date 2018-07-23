
manpathmunge () {
  if ! echo $MANPATH | egrep -q "(^|:)$1($|:)" ; then
    if [ "$2" = "after" ] ; then
      MANPATH=$MANPATH:$1
    else
      MANPATH=$1:$MANPATH
    fi
  fi
  export MANPATH
}
pythonpathmunge () {
  if ! echo $PYTHONPATH | egrep -q "(^|:)$1($|:)" ; then
    if [ "$2" = "after" ] ; then
      PYTHONPATH=$PYTHONPATH:$1
    else
      PYTHONPATH=$1:$PYTHONPATH
    fi
  fi
  export PYTHONPATH
}
pathmunge () {
  if ! echo $PATH | egrep -q "(^|:)$1($|:)" ; then
    if [ "$2" = "after" ] ; then
      PATH=$PATH:$1
    else
      PATH=$1:$PATH
    fi
  fi
  export PATH
}
libpathmunge () {
  if ! echo $LIBRARY_PATH | egrep -q "(^|:)$1($|:)" ; then
    if [ "$2" = "after" ] ; then
      LIBRARY_PATH=$LIBRARY_PATH:$1
    else
      LIBRARY_PATH=$1:$LIBRARY_PATH
    fi
  fi
  export LIBRARY_PATH
}
ldpathmunge () {
  if ! echo $LD_LIBRARY_PATH | egrep -q "(^|:)$1($|:)" ; then
    if [ "$2" = "after" ] ; then
      LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$1
    else
      LD_LIBRARY_PATH=$1:$LD_LIBRARY_PATH
    fi
  fi
  export LD_LIBRARY_PATH
}

pathmunge $SCROOT/soil/bin
manpathmunge $SCROOT/soil/share/man
libpathmunge $SCROOT/soil/lib
libpathmunge $SCROOT/soil/lib64
ldpathmunge $SCROOT/soil/lib
ldpathmunge $SCROOT/soil/lib64

# vim: sw=2 ts=2 tw=76 et nu ft=sh:
