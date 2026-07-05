# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


"""
Tools to run applications
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


import code
import importlib
import inspect
import re
import rlcompleter


__all__ = [
    'environ',
    'AppEnvironment',
    'get_current_appenv',
    'get_completions',
    'get_call_tip',
    'run_code',
    'stop_code',
    'build_pilot_namespace',
    'format_banner',
    'install_pilot_namespace',
]


# A dotted identifier chain such as ``range`` or ``mgr.add3DWidget``. The
# call tip only introspects such an expression, never one that calls or
# subscripts, so evaluating it cannot run arbitrary user code.
_IDENTIFIER_CHAIN = re.compile(
    r'[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$')


# All environment objects of this process.
environ = {}


class _ConsoleInterpreter(code.InteractiveConsole):
    """
    A :class:`code.InteractiveConsole` bound to an application namespace.

    The console compiles in ``"single"`` mode, so a bare expression is
    auto-displayed and bound to ``_`` the way the standard read-eval-print
    loop does. Exceptions are formatted against the user's own input by
    :meth:`showtraceback` and :meth:`showsyntaxerror`, not the host stack.

    ``exit()`` or ``quit()`` typed in the console raises :exc:`SystemExit`,
    which would tear down the interpreter that the pilot embeds. Guard it
    here so the console reports the request without ending the process.
    """
    def push(self, line):
        try:
            return super().push(line)
        except SystemExit:
            self.write("SystemExit: use the window controls to quit.\n")
            self.resetbuffer()
            return False


class AppEnvironment:
    """
    Collects the environment for an application.

    The read-eval-print loop uses a single namespace, so :ivar:`globals`
    and :ivar:`locals` are the same dict: names bound by executed code and
    names injected by the pilot are both visible to the interpreter.

    :ivar globals:
        The namespace of the application.
    :ivar locals:
        An alias of :ivar:`globals`.
    """
    def __init__(self, name):
        namespace = {
            # Give the application an alias of the top package.
            'sc': importlib.import_module('solvcon'),
            'appenv': self,
        }
        self.globals = namespace
        self.locals = namespace
        self.name = name
        self.namespace_refreshers = []
        self.console = _ConsoleInterpreter(namespace)
        # Each run of the application appends a new environment.
        environ[name] = self

    def seed(self, **handles):
        """Install curated handles into the interpreter namespace."""
        self.globals.update(handles)

    def add_namespace_refresher(self, refresh):
        """
        Register a callback that refreshes the namespace before each command.

        The callback receives the namespace dict. Use it for handles whose
        value tracks live session state, such as the current viewer.
        """
        self.namespace_refreshers.append(refresh)

    def run_code(self, source):
        """
        Feed a possibly multi-line command to the persistent interpreter.

        Each line drives ``push()``, which returns whether more input is
        needed to complete the statement. A block left open after the last
        line is closed with a blank line, the way a blank line ends a
        compound statement in the interactive interpreter.

        :return: True when the interpreter is still waiting for more input.
        """
        for refresh in self.namespace_refreshers:
            refresh(self.globals)
        more = False
        for line in source.split('\n'):
            more = self.console.push(line)
        if more:
            more = self.console.push('')
        return more


def get_appenv(name=None):
    if None is name:
        for i in range(10):
            name = f'anonymous{i}'
            if name not in environ:
                break
        else:
            raise ValueError("hit limit of anonymous environments (10)")
    app = environ.get(name, None)
    if None is app:
        app = AppEnvironment(name)
    return app


get_appenv(name='master')


def get_current_appenv():
    if not environ:
        raise KeyError("No AppEnviron is available")
    return environ[next(reversed(environ))]


def get_completions(text):
    aenv = get_current_appenv()
    namespace = {'__builtins__': __builtins__}
    namespace.update(aenv.globals)
    namespace.update(aenv.locals)
    completer = rlcompleter.Completer(namespace)
    completions = []
    i = 0
    while True:
        c = completer.complete(text, i)
        if c is None:
            break
        completions.append(c)
        i += 1
    return completions


def get_call_tip(expr):
    """
    A signature and docstring summary for the callable named by ``expr``.

    ``expr`` is a dotted identifier chain such as ``range`` or
    ``mgr.add3DWidget``. It is resolved against the current namespace, so,
    like the introspective completion, it can touch live objects. It is
    never evaluated when it is anything other than an identifier chain, so
    a call or subscript cannot run arbitrary code. Returns an empty string
    when the expression does not resolve to a callable.
    """
    expr = expr.strip()
    if not _IDENTIFIER_CHAIN.match(expr):
        return ''
    aenv = get_current_appenv()
    namespace = {'__builtins__': __builtins__}
    namespace.update(aenv.globals)
    try:
        obj = eval(expr, namespace)
    except Exception:
        return ''
    if not callable(obj):
        return ''
    try:
        signature = str(inspect.signature(obj))
    except (TypeError, ValueError):
        signature = '(...)'
    tip = expr + signature
    doc = inspect.getdoc(obj)
    if doc:
        tip += '\n' + doc.strip().split('\n\n')[0]
    return tip


def run_code(source):
    aenv = get_current_appenv()
    return aenv.run_code(source)


def stop_code(appenvobj=None):
    if None is appenvobj:
        environ.clear()
    else:
        names = [name for name, env in environ.items() if env is appenvobj]
        for name in names:
            del environ[name]


def build_pilot_namespace(mgr):
    """
    Curated console handles and their descriptions for the pilot.

    Duck-typed on the pilot manager (``mgr``) so it can be exercised with a
    stand-in in the tests. Returns ``(handles, entries)`` where ``handles``
    is a name-to-object dict to seed the namespace and ``entries`` is an
    ordered list of ``(name, description)`` for the banner.
    """
    def show_mesh(m):
        """Open a mesh in a fresh 3D viewer and return the viewer."""
        w = mgr.add3DWidget()
        w.updateMesh(m)
        w.showAxis(True)
        return w

    def viewers():
        """List the open 3D viewers."""
        return list(mgr.list3DWidgets())

    def meshes():
        """List the meshes currently loaded in the 3D viewers."""
        return [w.mesh for w in mgr.list3DWidgets() if w.mesh is not None]

    viewer = mgr.currentR3DWidget()
    handles = {
        'mgr': mgr,
        'viewer': viewer,
        'mesh': None if viewer is None else viewer.mesh,
        'show_mesh': show_mesh,
        'viewers': viewers,
        'meshes': meshes,
    }
    entries = [
        ('mgr', 'the running pilot manager'),
        ('viewer', 'the current 3D viewer (or None)'),
        ('mesh', 'the current mesh (or None)'),
        ('show_mesh(m)', 'open a mesh in a fresh 3D viewer'),
        ('viewers()', 'list the open 3D viewers'),
        ('meshes()', 'list the loaded meshes'),
        ('sc', 'the solvcon package'),
    ]
    return handles, entries


def _refresh_pilot_namespace(mgr, namespace):
    viewer = mgr.currentR3DWidget()
    namespace['viewer'] = viewer
    namespace['mesh'] = None if viewer is None else viewer.mesh


def format_banner(entries):
    """Format ``(name, description)`` entries as an aligned banner."""
    width = max(len(name) for name, _ in entries)
    lines = ["solvcon pilot console. Handles in scope:"]
    for name, desc in entries:
        lines.append("  {}  {}".format(name.ljust(width), desc))
    return "\n".join(lines) + "\n"


def install_pilot_namespace(mgr, appenv):
    """
    Seed the console namespace from the pilot manager and keep it fresh.

    The dynamic handles (``viewer``, ``mesh``) are refreshed before each
    command so they track the focused viewer.

    :return: The banner text listing the curated handles.
    """
    handles, entries = build_pilot_namespace(mgr)
    appenv.seed(**handles)
    appenv.add_namespace_refresher(
        lambda namespace: _refresh_pilot_namespace(mgr, namespace))
    return format_banner(entries)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
