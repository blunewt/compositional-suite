"""
Directly-runnable launcher for the Compositional Suite GUI.

``compositional_suite/app.py`` cannot be executed as a file: it uses
package-relative imports (``from . import analysis``), which only resolve when
Python loads it as part of the ``compositional_suite`` package. Running it directly
fails with "attempted relative import with no known parent package".

This script sits at the project root, puts that root on ``sys.path``, and then
imports the app as a package — so it works from an IDE's run button with no
installation step.

Equivalent alternatives:
    python -m compositional_suite.app
    pip install -e .   &&   compositional-suite

For the frozen executable use ``packaging/launcher.py`` instead; it calls
``multiprocessing.freeze_support()`` before Qt loads, which the Batch tab needs.
"""

import multiprocessing
import os
import sys

if __name__ == "__main__":
    # Batch runs use a ProcessPoolExecutor; on Windows each worker re-executes
    # this file, so freeze_support() must run before PyQt5 is imported.
    multiprocessing.freeze_support()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from compositional_suite.app import main

    sys.exit(main())
