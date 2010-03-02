import os
import sys

AddOption('--f90', dest='f90', type='string',
    action='store', default='gfortran',
    help='Fortran compiler (SCons tool): gfortran, ifort.')

AddOption('--download', dest='download',
    action='store_true', default=False,
    help='Flag to download external packages.')
AddOption('--extract', dest='extract',
    action='store_true', default=False,
    help='Flag to extract external packages.')
AddOption('--apply-patches', dest='patches',
    action='store', default='',
    help='Indicate matches to be applied.')

AddOption('--count', dest='count',
    action='store_true', default=False,
    help='Count line of sources.')

class Archive(object):
    """
    External package downloader/extractor.
    """

    bufsize = 1024*1024
    depdir = 'dep'

    pkgs = (
        ('http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-4.0.tar.gz',
         '0aa546419ff7ef50bd86ce1ec7f727c7'),
    )

    def __init__(self, url, md5sum, filename=None):
        import os
        from urlparse import urlparse
        self.url = url
        self.md5sum = md5sum
        if filename == None:
            up = urlparse(url)
            filename = up[2].split('/')[-1]
        self.filename = os.path.join(self.depdir, filename)
        if not os.path.exists(self.depdir):
            os.makedirs(self.depdir)

    @classmethod
    def digest(cls, f):
        import md5
        m = md5.new()
        while True:
            data = f.read(cls.bufsize)
            m.update(data)
            if len(data) < cls.bufsize: break
        return m.hexdigest()

    def download(self):
        import sys
        import os
        import urllib
        url = self.url
        fn = self.filename
        cksum = self.md5sum
        if os.path.exists(fn):
            if cksum and cksum != self.digest(open(fn, 'rb')):
                sys.stdout.write("%s checksum mismatch, delete old.\n" % fn)
                os.unlink(fn)
            else:
                sys.stdout.write("%s exists.\n" % fn)
                return False
        sys.stdout.write("Download %s from %s: " % (fn, url))
        uf = urllib.urlopen(url)
        f = open(fn, 'wb')
        sys.stdout.flush()
        while True:
            data = uf.read(self.bufsize)
            sys.stdout.write('.')
            sys.stdout.flush()
            f.write(data)
            if len(data) < self.bufsize: break
        uf.close()
        f.close()
        if cksum:
            if cksum != self.digest(open(fn, 'rb')):
                sys.stdout.write("note, %s checksum mismatch!\n" % fn)
            else:
                sys.stdout.write("%s checksum OK.\n" % fn)
        else:
            sys.stdout.write("no checksum defined for %s .\n" % fn)
        sys.stdout.write(" done.\n")

    def extract(self):
        import tarfile
        tar = tarfile.open(self.filename)
        tar.extractall(path=self.depdir)
        tar.close()

    @classmethod
    def downloadall(cls):
        for url, md5sum in cls.pkgs:
            obj = Archive(url, md5sum)
            obj.download()

    @classmethod
    def extractall(cls):
        for url, md5sum in cls.pkgs:
            obj = Archive(url, md5sum)
            obj.extract()

class LineCounter(object):
    """
    Walk given directory to count lines in source files.
    """

    def __init__(self, *args, **kw):
        self.exts = args
        self.counter = dict()
        self.testdir = kw.pop('testdir', ['tests'])
        self.testcounter = 0

    def __call__(self, path):
        import os
        from os.path import join, splitext
        for root, dirs, files in os.walk(path):
            for fname in files:
                mainfn, extfn = splitext(fname)
                if extfn not in self.exts:
                    continue
                nline = len(open(join(root, fname)).readlines())
                self.counter[extfn] = self.counter.get(extfn, 0) + nline
                if os.path.basename(root) in self.testdir:
                    self.testcounter += nline

    def __str__(self):
        keylenmax = max([len(key) for key in self.counter])
        tmpl = "%%-%ds = %%d" % keylenmax
        all = 0
        ret = list()
        for extfn in sorted(self.counter.keys()):
            ret.append(tmpl % (extfn, self.counter[extfn]))
            all += self.counter[extfn]
        ret.append(tmpl % ('All', all))
        ret.append('%d are for unittest.' % self.testcounter)
        return '\n'.join(ret)

if GetOption('download'):
    Archive.downloadall()
if GetOption('extract'):
    Archive.extractall()

patches = [token for token in GetOption('patches').split(',') if token]
for patch in patches:
    patchpath = os.path.join('patch', patch+'.patch')
    os.system('patch -p0 -i %s'%patchpath)

if GetOption('count'):
    counter = LineCounter('.py', '.f90', '.inc', '.c', '.h', 'cu')
    paths = ('solvcon', 'src', 'test')
    for path in paths:
        counter(path)
    sys.stdout.write('In directories %s:\n' % ', '.join(paths))
    sys.stdout.write(str(counter)+'\n')
    sys.exit(0)

# global tools.
tools = []
tools.append(GetOption('f90'))
if GetOption('f90') == 'gfortran':
    FLAG_FPP = '-x f95-cpp-input'
elif GetOption('f90') == 'ifort':
    FLAG_FPP = '-fpp'
if sys.platform.startswith('win'):
    tools.insert(0, 'mingw')
else:
    tools.insert(0, 'default')

# solvcon environment.
env = Environment(ENV=os.environ, tools=tools,
    F90FLAGS=' '.join(['-O2', FLAG_FPP]),
)
def build_epydoc(target, source, env):
    import sys
    sys.path.insert(0, '.')
    from solvcon.helper import generate_apidoc
    generate_apidoc()
def build_sphinx(target, source, env):
    import os
    os.system('sphinx-build doc/source doc/build')
env.Append(BUILDERS={
    'Epydoc': Builder(
        action=build_epydoc,
    ),
    'Sphinx': Builder(
        action=build_sphinx,
    ),
})

# metis environment.
metissrc = 'dep/metis-4.0/Lib'
CCFLAGS = ''
if sys.platform.startswith('win'):
    CCFLAGS = ' '.join([CCFLAGS, '-D__VC__'])
OPM = '-O3'
DBG = ''
metisenv = Environment(ENV=os.environ, tools=tools,
    CCFLAGS=' '.join([CCFLAGS, '-I%s'%metissrc, OPM, DBG]))

Export('env')
Export('metisenv')
Export('metissrc')
SConscript(['SConscript'])
# vim: set ft=python ff=unix:
