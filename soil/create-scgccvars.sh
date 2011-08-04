#!/bin/sh
echo "export SCROOT=$SCROOT" > $SCROOT/bin/scgccvars.sh
cat scgccvars.sh >> $SCROOT/bin/scgccvars.sh
