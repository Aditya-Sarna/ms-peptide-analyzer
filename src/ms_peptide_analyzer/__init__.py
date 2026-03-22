"""
ms_peptide_analyzer
===================
In-silico peptide analysis powered by pyOpenMS nanobind/pybind11 C++ bindings.

Modules
-------
peptide      - AASequence wrappers, protein digestion
fragmentation - Theoretical b/y/a/c/z ion generation
isotopes     - Isotope distribution calculation
scorer       - Peptide property scoring
cli          - Command-line interface
"""

from ms_peptide_analyzer.peptide import digest_protein, PeptideInfo
from ms_peptide_analyzer.fragmentation import FragmentIonGenerator
from ms_peptide_analyzer.isotopes import IsotopeAnalyzer
from ms_peptide_analyzer.scorer import PeptideScorer

__all__ = [
    "digest_protein",
    "PeptideInfo",
    "FragmentIonGenerator",
    "IsotopeAnalyzer",
    "PeptideScorer",
]
