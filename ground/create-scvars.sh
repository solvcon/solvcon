#!/bin/sh
mkdir -p $SCETC
echo "export SCROOT=$SCROOT" > $SCETC/scvars.sh
cat scvars.sh >> $SCETC/scvars.sh
echo "setenv SCROOT $SCROOT" > $SCETC/scvars.csh
cat scvars.csh >> $SCETC/scvars.csh
