"""
PyInstaller entry script for the standalone Compositional Suite executable.

``freeze_support()`` runs before Compositional Suite (and therefore PyQt5) is imported.
The Batch Simulator uses a ProcessPoolExecutor, and on Windows each worker is
started by re-executing this bundle; without the early call the workers would
re-run the GUI instead of the task, spawning processes without bound. Calling it
first also lets workers exit before paying to import Qt.

The executable is built without a console window, so a start-up failure would be
completely silent. Any error raised while starting is therefore written to
``crash.log`` (Windows: %LOCALAPPDATA%\\CompositionalSuite) and shown in a message box.
"""

import multiprocessing
import os
import sys
import time
import traceback


def _report_fatal(details: str, show_dialog: bool = True):
    path = None
    try:
        folder = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "CompositionalSuite")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "crash.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{details}\n")
    except Exception:
        pass
    if sys.platform == "win32":
        if not show_dialog:                       # tests and scripts must never block on a dialog
            return
        try:
            import ctypes
            text = "Compositional Suite could not start.\n\n" + details[-1200:]
            if path:
                text += f"\n\nThis message was saved to:\n{path}"
            ctypes.windll.user32.MessageBoxW(None, text, "Compositional Suite", 0x10)
        except Exception:
            pass
    else:
        print(details, file=sys.stderr)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        from compositional_suite.app import main
        main()
    except SystemExit:
        raise
    except Exception:
        _report_fatal(traceback.format_exc())
        sys.exit(1)
