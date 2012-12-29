"""
Paper tool.
"""

import os
import glob

PAPER_BUILDERS = dict()
class PaperBuilderMeta(type):
    def __new__(cls, name, bases, namespace):
        newcls = super(PaperBuilderMeta, cls).__new__(
            cls, name, bases, namespace)
        if name != 'PaperBuilder':
            PAPER_BUILDERS[name] = newcls
        return newcls

def exists(env):
    return True

def generate(env):
    from SCons.Builder import Builder
    attrnames = [
        'emitter',
        'action',
        'suffix',
        'src_suffix',
    ]
    for name in sorted(PAPER_BUILDERS.keys()):
        pbdr = PAPER_BUILDERS[name]
        # create builder object.
        bdrkw = dict()
        for attrname in attrnames:
            if hasattr(pbdr, attrname):
                bdrkw[attrname] = getattr(pbdr, attrname)
        builder = Builder(**bdrkw)
        # modify environment for the PaperBuilder.
        pbdr.modify_environment(env)
        # append the builder to environment.
        env.Append(BUILDERS={name: builder})
    return env

class PaperBuilder(object):
    __metaclass__ = PaperBuilderMeta
    @classmethod
    def modify_environment(cls, env):
        pass

class RedoBib(PaperBuilder):
    @staticmethod
    def action(target, source, env):
        source = str(source[0])
        src = os.path.splitext(source)[0]
        os.system('bibtex %s' % src)

class CleanBib(PaperBuilder):
    suffix = '.bib'
    src_suffix = '.bib'

    @staticmethod
    def action(target, source, env):
        source = str(source[0])
        target = str(target[0])
        # set fields to be cleared.
        toclean = ('url', 'doi', 'note', 'abstract', 'keywords', 'issn')
        newlines = []
        for line in open(source).readlines():
            line = line.decode('utf8')
            tokens = line.split('=')
            if tokens[0].strip() in toclean:
                continue
            newlines.append(line)
        open(target, 'w').write((''.join(newlines)).encode('utf8'))

class Pstricks(PaperBuilder):
    src_suffix = '.tex'

    _template_ = r'''\documentclass[%sletterpaper,dvips]{article}
\usepackage[usenames]{color}
\usepackage{pst-all}
\usepackage{pst-3dplot}
\usepackage{pst-eps}
\usepackage{pst-coil}
\usepackage{pst-bar}
\usepackage{multido}
\usepackage{cmbright}
\usepackage{fancyvrb}
\begin{document}
\pagestyle{empty}
\begin{TeXtoEPS}
%s
\end{TeXtoEPS}
\end{document}'''

    @classmethod
    def action(cls, target, source, env):
        options = env.get('FONTSIZE', '')
        if len(options) > 0 and options[-1] != ',':
            options += ','
        assert len(target) == len(source)
        for dst, src in zip(target, source):
            dst = str(dst)
            src = str(src)
            tmpf = open('makeeps_tmp.tex', 'w').write(
                cls._template_%(options, open(src).read()))
            os.system('latex makeeps_tmp.tex')
            os.system('dvips makeeps_tmp.dvi -E -o %s' % dst)
            for fn in glob.glob('makeeps_tmp.*'):
                os.unlink(fn)

    @classmethod
    def emitter(cls, target, source, env):
        import os
        newtarget = list()
        for src in source:
            src = str(src)
            srcmain = os.path.split(src)[-1]
            dst = '/'.join([
                env['OUTDIR'],
                os.path.splitext(srcmain)[0]+'.eps',
            ])
            newtarget.append(dst)
        return newtarget, source

class Imconvert(PaperBuilder):
    @classmethod
    def action(cls, target, source, env):
        import sys
        import os
        import glob
        assert len(target) == len(source)
        for dst, src in zip(target, source):
            dst = str(dst)
            src = str(src)
            cmd = '%s -density %s -units PixelsPerInch %s %s' % (
                env['CONVERT'], env['DPI'], src, dst,
            )
            sys.stdout.write(cmd + '\n')
            os.system(cmd)

    @staticmethod
    def modify_environment(env):    
        import sys
        env['DPI'] = 300
        ## set convert from imagemagick.
        CONVERT = 'convert'
        if sys.platform.startswith('win'):
            for path in os.environ['PATH'].split(';'):
                if 'imagemagick' in path.lower():
                    break
            CONVERT = '"%s"' % '\\'.join([path, 'convert.exe'])
        env['CONVERT'] = CONVERT

    @classmethod
    def emitter(cls, target, source, env):
        import os
        newtarget = list()
        for src in source:
            src = str(src)
            srcmain = os.path.split(src)[-1]
            dst = '/'.join([
                env['OUTDIR'],
                os.path.splitext(srcmain)[0]+env['EXT'],
            ])
            newtarget.append(dst)
        return newtarget, source

class Diaexport(PaperBuilder):
    @classmethod
    def action(cls, target, source, env):
        import sys
        import os
        import glob
        assert len(target) == len(source)
        for dst, src in zip(target, source):
            dst = str(dst)
            src = str(src)
            cmd = 'dia %s --export=%s' % (src, dst)
            sys.stdout.write(cmd + '\n')
            os.system(cmd)

    @classmethod
    def emitter(cls, target, source, env):
        import os
        newtarget = list()
        for src in source:
            src = str(src)
            srcmain = os.path.split(src)[-1]
            dst = '/'.join([
                env['OUTDIR'],
                os.path.splitext(srcmain)[0]+env['EXT'],
            ])
            newtarget.append(dst)
        return newtarget, source

class PackZip(PaperBuilder):
    @staticmethod
    def action(target, source, env):
        from zipfile import ZipFile
        zf = ZipFile(str(target[0]), 'w')
        for src in source:
            zf.write(str(src))
        zf.close()

class PSPDF(PaperBuilder):
    suffix = '.pdf'
    src_suffix = '.ps'
    action = 'ps2pdf $SOURCE $TARGET'
