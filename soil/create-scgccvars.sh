#!/bin/sh
mkdir -p $SCROOT/bin
echo "export SCROOT=$SCROOT" > $SCETC/scgccvars.sh
cat scgccvars.sh >> $SCETC/scgccvars.sh
echo "setenv SCROOT $SCROOT" > $SCETC/scgccvars.csh
cat scgccvars.csh >> $SCETC/scgccvars.csh
