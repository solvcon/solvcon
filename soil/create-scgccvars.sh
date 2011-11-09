#!/bin/sh
mkdir -p $SCROOT/bin
echo "export SCROOT=$SCROOT" > $SCROOT/bin/scgccvars.sh
cat scgccvars.sh >> $SCROOT/bin/scgccvars.sh
echo "setenv SCROOT $SCROOT" > $SCROOT/bin/scgccvars.csh
cat scgccvars.csh >> $SCROOT/bin/scgccvars.csh
