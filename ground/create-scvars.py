import os
scroot = os.environ['SCROOT']
lines = open('scvars.sh').readlines()
lines[0] = 'export SCROOT=%s\n' % scroot
dst = open(os.path.join(scroot, 'bin', 'scvars.sh'), 'w')
for line in lines:
    dst.write(line)
dst.close()
lines = open('scvars.csh').readlines()
lines[0] = 'setenv SCROOT %s\n' % scroot
dst = open(os.path.join(scroot, 'bin', 'scvars.csh'), 'w')
for line in lines:
    dst.write(line)
dst.close()
