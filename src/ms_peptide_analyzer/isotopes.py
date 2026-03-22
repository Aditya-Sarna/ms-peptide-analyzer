"""
isotopes.py
-----------
Isotope distribution and envelope analysis via pyOpenMS C++ bindings.

Key pyOpenMS C++ bindings used (via nanobind/pybind11):
  - pyopenms.EmpiricalFormula              : molecular formula + isotope math
  - pyopenms.CoarseIsotopePatternGenerator : integer-resolution isotope envelope
  - pyopenms.FineIsotopePatternGenerator   : high-resolution isotope envelope
  - pyopenms.IsotopeDistribution           : container for (mass, intensity) peaks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyopenms as oms


@dataclass
class IsotopePeak:
    """A single peak in an isotope envelope."""
    mass: float
    mz: float          # computed at given charge
    intensity: float   # relative (0–1 normalized)
    isotope_index: int  # 0 = monoisotopic, 1 = M+1, ...

    def __str__(self) -> str:
        return (
            f"M+{self.isotope_index}  mass={self.mass:.4f}  "
            f"m/z={self.mz:.4f}  rel_int={self.intensity:.4f}"
        )


class IsotopeAnalyzer:
    """
    Compute isotope envelopes and related properties using pyOpenMS.

    Wraps EmpiricalFormula and CoarseIsotopePatternGenerator C++ objects
    accessed through the pyOpenMS pybind11/nanobind binding layer.

    Parameters
    ----------
    max_isotopes : int
        How many isotope peaks to include in the envelope (0 = monoisotopic
        only, 5 means M through M+5).
    use_fine : bool
        Use FineIsotopePatternGenerator (high-res, slower) instead of the
        default CoarseIsotopePatternGenerator (unit-resolution, fast).
    """

    PROTON = 1.007276

    def __init__(self, max_isotopes: int = 5, use_fine: bool = False) -> None:
        self.max_isotopes = max_isotopes
        self.use_fine = use_fine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_sequence(
        self, sequence: str, charge: int = 1
    ) -> List[IsotopePeak]:
        """
        Compute the isotope envelope for a peptide sequence.

        Parameters
        ----------
        sequence : str
            One-letter peptide string (may include OpenMS modifications).
        charge : int
            Charge state for m/z calculation.

        Returns
        -------
        list[IsotopePeak]  normalized so the most-abundant peak = 1.0.
        """
        aa_seq = oms.AASequence.fromString(sequence)
        formula: oms.EmpiricalFormula = aa_seq.getFormula(
            oms.Residue.ResidueType.Full, 0
        )
        return self._compute(formula, charge)

    def from_formula(
        self, formula_str: str, charge: int = 1
    ) -> List[IsotopePeak]:
        """
        Compute the isotope envelope from a molecular formula string.

        Parameters
        ----------
        formula_str : str
            e.g. "C100H160N28O30S2"
        charge : int
            Charge state for m/z calculation.
        """
        formula = oms.EmpiricalFormula(formula_str)
        return self._compute(formula, charge)

    def monoisotopic_mass(self, sequence: str) -> float:
        """Return the monoisotopic neutral mass of a peptide."""
        aa_seq = oms.AASequence.fromString(sequence)
        return aa_seq.getMonoWeight(oms.Residue.ResidueType.Full, 0)

    def average_mass(self, sequence: str) -> float:
        """Return the average (chemical) neutral mass of a peptide."""
        aa_seq = oms.AASequence.fromString(sequence)
        return aa_seq.getAverageWeight(oms.Residue.ResidueType.Full, 0)

    def most_abundant_mass(self, sequence: str) -> float:
        """Return the mass of the most abundant isotope peak."""
        peaks = self.from_sequence(sequence, charge=1)
        if not peaks:
            return 0.0
        return max(peaks, key=lambda p: p.intensity).mass

    def envelope_summary(self, sequence: str, charge: int = 1) -> dict:
        """
        Return a summary dict with key envelope properties.
        """
        peaks = self.from_sequence(sequence, charge)
        aa = oms.AASequence.fromString(sequence)
        formula = aa.getFormula(oms.Residue.ResidueType.Full, 0)

        most_abundant = max(peaks, key=lambda p: p.intensity)
        mono = peaks[0] if peaks else None

        return {
            "sequence": sequence,
            "formula": formula.toString(),
            "charge": charge,
            "monoisotopic_mz": mono.mz if mono else None,
            "most_abundant_mz": most_abundant.mz,
            "most_abundant_isotope": most_abundant.isotope_index,
            "peaks": [
                {"index": p.isotope_index, "mz": round(p.mz, 5), "rel_intensity": round(p.intensity, 4)}
                for p in peaks
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(
        self, formula: oms.EmpiricalFormula, charge: int
    ) -> List[IsotopePeak]:
        """Use the C++ IsotopeDistribution generator via pyOpenMS binding."""
        if self.use_fine:
            gen = oms.FineIsotopePatternGenerator()
            gen.setThreshold(1e-4)
        else:
            gen = oms.CoarseIsotopePatternGenerator()
            gen.setMaxIsotope(self.max_isotopes)

        # getIsotopeDistribution calls C++ via nanobind
        iso_dist: oms.IsotopeDistribution = formula.getIsotopeDistribution(gen)
        raw_peaks = list(iso_dist.getContainer())

        if not raw_peaks:
            return []

        max_int = max(p.getIntensity() for p in raw_peaks) or 1.0

        result: List[IsotopePeak] = []
        for idx, peak in enumerate(raw_peaks):
            mass = peak.getMZ()        # .getMZ() returns the mass on IsotopeDistribution
            rel_int = peak.getIntensity() / max_int
            mz = (mass + charge * self.PROTON) / charge
            result.append(IsotopePeak(mass=mass, mz=mz, intensity=rel_int, isotope_index=idx))

        return result

    # ------------------------------------------------------------------
    # Convenience: compare two envelopes (e.g. light vs heavy labelled)
    # ------------------------------------------------------------------

    def compare_envelopes(
        self, seq1: str, seq2: str, charge: int = 1
    ) -> List[Tuple[float, float, float]]:
        """
        Compare isotope envelopes of two peptides.

        Returns list of (mz_seq1, mz_seq2, delta_mz) for paired peaks.
        Useful for SILAC / isotope-label ratio estimation.
        """
        e1 = self.from_sequence(seq1, charge)
        e2 = self.from_sequence(seq2, charge)

        pairs = []
        for p1, p2 in zip(e1, e2):
            pairs.append((p1.mz, p2.mz, p2.mz - p1.mz))
        return pairs
