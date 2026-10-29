"""
polymer.py
----------
Defines the Polymer data model used throughout the simulation.

A Polymer is a lightweight container for a single chain — a list of monomer
IDs (1-indexed integers).  Rate constants and global simulation parameters are
intentionally kept out of this class; they belong to SimulationEngine and are
passed in at call time to avoid per-chain duplication.
"""


class Polymer:
    """
    Represents a single polymer chain as a sequence of monomer IDs.

    Rate constants are NOT stored here — they belong to the SimulationEngine
    and are passed in at call time, avoiding redundant per-chain duplication.
    The numMonomers field has been removed as it was dead state.
    """

    def __init__(self, polymerList: list):
        self.polymerList = polymerList

    def lastMonomer(self) -> int:
        """Returns the last monomer unit in the chain (1-indexed)."""
        return self.polymerList[-1]

    def asArray(self) -> list:
        """Returns the polymer sequence as a list."""
        return self.polymerList

    def len(self) -> int:
        """Returns the length of the polymer chain."""
        return len(self.polymerList)

    def append(self, monomer: int) -> None:
        """Appends a new monomer unit to the chain."""
        self.polymerList.append(monomer)
