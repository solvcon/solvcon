# Copyright (C) 2017 Yung-Yu Chen <yyc@solvcon.net>.

SCGROUND="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCDEP=${SCDEP:=$HOME/tmp/scdep}
mkdir -p $SCDEP
SCDEP="$(cd $SCDEP && pwd)"
SCDL="${SCDL:=$SCDEP/downloaded}"
mkdir -p $SCDL
SCDL="$(cd $SCDL && pwd)"
if [ $(uname) == Darwin ] ; then
  SCDLLEXT=dylib
  NP=${NP:=$(sysctl -n hw.ncpu)}
elif [ $(uname) == Linux ] ; then
  SCDLLEXT=so
  NP=${NP:=$(cat /proc/cpuinfo | grep processor | wc -l)}
else
  NP=${NP:=1}
fi

download () {
  loc=$1
  url=$2
  md5hash=$3
  if [ $(uname) == Darwin ] ; then
    md5="md5 -q"
  elif [ $(uname) == Linux ] ; then
    md5=md5sum
  fi
  if [ ! -e $loc ] || [ $md5hash != `$md5 $loc | cut -d ' ' -f 1` ] ; then
    mkdir -p $(dirname $loc)
    rm -f $loc
    echo "Download from $url"
    curl -sSL -o $loc $url
  fi
  if [ $md5hash != `$md5 $loc | cut -d ' ' -f 1` ] ; then
    echo "$(basename $loc) md5 hash $md5hash but got `$md5 $loc`"
  else
    echo "$(basename $loc) md5 hash $md5hash confirmed"
  fi
}

finalize () {
  echo "$1 completed"
}

namemunge () {
  if ! echo ${!1} | egrep -q "(^|:)$2($|:)" ; then
    if [ -z "${!1}" ] ; then
      eval "$1=$2"
    else
      if [ "$3" == "after" ] ; then
        eval "$1=\$$1:$2"
      else
        eval "$1=$2:\$$1"
      fi
    fi
  fi
  eval "export $1"
}

namemunge PATH $SCDEP/bin
if [ $(uname) == Darwin ] ; then
  namemunge DYLD_LIBRARY_PATH $SCDEP/lib
elif [ $(uname) == Linux ] ; then
  namemunge LD_LIBRARY_PATH $SCDEP/lib
fi

# vim: set et nobomb ff=unix fenc=utf8:
