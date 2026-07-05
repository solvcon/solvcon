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
import rlcompleter


__all__ = [
    'environ',
    'AppEnvironment',
    'get_current_appenv',
    'get_completions',
    'run_code',
    'stop_code',
]


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
        self.console = _ConsoleInterpreter(namespace)
        # Each run of the application appends a new environment.
        environ[name] = self

    def run_code(self, source):
        """
        Feed a possibly multi-line command to the persistent interpreter.

        Each line drives ``push()``, which returns whether more input is
        needed to complete the statement. A block left open after the last
        line is closed with a blank line, the way a blank line ends a
        compound statement in the interactive interpreter.

        :return: True when the interpreter is still waiting for more input.
        """
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
    has_key = False
    for k in reversed(environ):
        has_key = True
        break
    if not has_key:
        raise KeyError("No AppEnviron is available")
    return environ[k]


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


def run_code(source):
    aenv = get_current_appenv()
    return aenv.run_code(source)


def stop_code(appenvobj=None):
    if None is appenvobj:
        environ.clear()
    else:
        indices = [i for i, o in enumerate(environ) if o == appenvobj]
        indices = reversed(indices)
        for i in indices:
            del environ[i]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
