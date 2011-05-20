import os
scprefix = os.environ['SCPREFIX']
lines = open('scvars.sh').readlines()
lines[0] = 'export SCROOT=%s\n' % scprefix
dst = open(os.path.join(scprefix, 'bin', 'scvars.sh'), 'w')
for line in lines:
    dst.write(line)
dst.close()
