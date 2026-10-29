"""
Compositional Suite: stochastic copolymerisation simulation and sequence analysis.

Nothing is imported eagerly here on purpose. ``headless`` and its dependencies
are imported by every ProcessPoolExecutor worker; pulling PyQt5, pandas or
scipy in at package import would be paid again on each spawn.

Import what you need directly::

    from compositional_suite import headless, simulation_engine   # light
    from compositional_suite import reactivity_library            # light (stdlib only)
    from compositional_suite.app import main         # pulls in Qt
"""

# Release metadata - the single place to edit. pyproject.toml reads the version
# from here; the About dialog and the Windows executable's file properties read
# the copyright lines. Placeholders are checked by tools/check_placeholders.py.
# DTU owns software created by its employees: keep DTU as the copyright holder.
__version__ = "1.0.0"
__copyright__ = "Copyright \u00a9 2026 Technical University of Denmark"
__developed_by__ = ("This version of the software (Compositional Suite 1.0.0) was developed by "
                    "Charalampos Sideris, PhD Student, Department of Health Technology, and "
                    "Anton A. A. Autzen, Associate Professor, Department of Health Technology.")
__authors__ = "Charalampos Sideris, Anton A. A. Autzen"
__url__ = "https://github.com/blunewt/compositional-suite"
