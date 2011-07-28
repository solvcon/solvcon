import os
scprefix = os.environ['SCPREFIX']
lines = open('scgccvars.sh').readlines()
lines[0] = 'export SCROOT=%s\n' % scprefix
dst = open(os.path.join(scprefix, 'bin', 'scgccvars.sh'), 'w')
for line in lines:
    dst.write(line)
dst.close()
