# Settings

Pilot keeps its settings in `pilot.json` in solvcon's configuration directory,
resolved in this order:

1. `$SOLVCON_CONFIG_HOME`, taken as the directory itself when it is set. This
   keeps a test or a self-contained checkout from touching the settings of the
   user who runs it.
2. `$XDG_CONFIG_HOME/solvcon`, when `XDG_CONFIG_HOME` is set.
3. `~/.config/solvcon` otherwise.

Pilot starts from its defaults when the file is missing or cannot be parsed,
so removing the file or breaking a hand edit does not stop the application.

Settings are grouped by section. The `ui` section holds the state of the user
interface, one entry per part of it that is remembered:

```json
{
  "ui": {
    "window": {
      "height": 600,
      "width": 1000,
      "x": 100,
      "y": 80
    }
  }
}
```

The `window` entry is the main window's size and location, applied at startup
and written back when the application quits. A saved location on a screen that
is no longer connected is ignored, so the window cannot come back out of sight
after a monitor is detached.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
