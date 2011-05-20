import sys
for fn in sys.argv[1:]:
    fo = open(fn)
    lines = fo.readlines()
    fo.close
    lines[0] = '#!%s\n' % sys.executable
    fo = open(fn, 'w')
    for line in lines:
        fo.write(line)
    fo.close()
