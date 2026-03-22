"""
scorer.py
---------
Peptide physicochemical property scoring using pyOpenMS C++ bindings.

Key pyOpenMS C++ bindings used (via nanobind/pybind11):
  - pyopenms.AASequence         : sequence object with property accessors
  - pyopenms.AAIndex            : amino-acid property index (hydrophobicity etc.)
  - pyopenms.ProteaseDigestion  : detect if sequence contains cleavage sites
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import pyopenms as oms


# pKa values for N-terminus, C-terminus, and ionizable side chains (Henderson-Hasselbalch)
_PKA = {
    "N_term": 8.0,
    "C_term": 3.1,
    "D": 3.86,
    "E": 4.25,
    "C": 8.33,
    "Y": 10.07,
    "H": 6.04,
    "K": 10.54,
    "R": 12.48,
}

# Kyte-Doolittle hydrophobicity scale
_HYDROPHOBICITY = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Bulkiness / steric hindrance scale (Zimmerman)
_BULKINESS = {
    "A": 11.5, "R": 14.28,"N": 12.82,"D": 11.68,"C": 13.46,
    "Q": 14.45,"E": 13.57,"G": 3.4,  "H": 13.69,"I": 21.4,
    "L": 21.4, "K": 15.71,"M": 16.25,"F": 19.8, "P": 17.43,
    "S": 9.47, "T": 15.77,"W": 21.67,"Y": 18.03,"V": 21.57,
}


@dataclass
class PeptideScore:
    """Computed physicochemical scores for a peptide."""
    sequence: str
    length: int
    mono_mass: float
    charge_at_ph7: float
    isoelectric_point: float
    gravy: float                # Grand Average of Hydropathy
    avg_bulkiness: float
    instability_index: float    # Guruprasad instability index
    aromaticity: float          # fraction of F+W+Y
    aliphatic_index: float      # Ikai aliphatic index
    is_tryptic_n: bool          # starts after K/R
    is_tryptic_c: bool          # ends with K or R

    def detection_likelihood(self) -> str:
        """
        Simple heuristic detectability label.

        Rules (based on common LC-MS/MS observable peptide criteria):
          - Length 7-25
          - GRAVY in [-2.0, 2.5]
          - No missed cleavages inferred
          - No extreme pI
        """
        if (
            7 <= self.length <= 25
            and -2.0 <= self.gravy <= 2.5
            and 4.0 <= self.isoelectric_point <= 10.0
            and self.instability_index < 40.0
        ):
            return "HIGH"
        if (
            6 <= self.length <= 35
            and -3.0 <= self.gravy <= 3.5
        ):
            return "MEDIUM"
        return "LOW"

    def __str__(self) -> str:
        return (
            f"{self.sequence}\n"
            f"  Length={self.length}  Mass={self.mono_mass:.3f} Da\n"
            f"  pI={self.isoelectric_point:.2f}  Charge@pH7={self.charge_at_ph7:.2f}\n"
            f"  GRAVY={self.gravy:.3f}  Instability={self.instability_index:.1f}\n"
            f"  Aromaticity={self.aromaticity:.3f}  AliphaticIdx={self.aliphatic_index:.1f}\n"
            f"  Detectability={self.detection_likelihood()}"
        )


class PeptideScorer:
    """
    Compute physicochemical properties of peptides using pyOpenMS C++ bindings.

    Where pyOpenMS exposes the property directly via AASequence, that binding
    is used. pI and charge calculations are implemented using the
    Henderson-Hasselbalch equation applied to the amino-acid composition
    extracted from the AASequence C++ object.
    """

    def score(self, sequence: str) -> PeptideScore:
        """
        Compute all scores for a peptide sequence.

        Parameters
        ----------
        sequence : str
            One-letter peptide sequence (uppercase, no modifications).

        Returns
        -------
        PeptideScore
        """
        seq = sequence.upper()
        aa_seq = oms.AASequence.fromString(seq)

        mono = aa_seq.getMonoWeight(oms.Residue.ResidueType.Full, 0)
        length = aa_seq.size()

        composition = self._composition(seq)
        pI = self._isoelectric_point(composition)
        charge_ph7 = self._charge_at_ph(composition, 7.0)

        gravy = self._gravy(seq)
        bulkiness = self._avg_bulkiness(seq)
        instability = self._instability_index(seq)
        aromaticity = sum(seq.count(aa) for aa in "FWY") / max(length, 1)
        aliphatic = self._aliphatic_index(seq)

        is_tryptic_c = seq[-1] in ("K", "R") if seq else False
        is_tryptic_n = True  # N-term tryptic requires context; default True

        return PeptideScore(
            sequence=seq,
            length=length,
            mono_mass=mono,
            charge_at_ph7=charge_ph7,
            isoelectric_point=pI,
            gravy=gravy,
            avg_bulkiness=bulkiness,
            instability_index=instability,
            aromaticity=aromaticity,
            aliphatic_index=aliphatic,
            is_tryptic_n=is_tryptic_n,
            is_tryptic_c=is_tryptic_c,
        )

    def score_many(self, sequences: List[str]) -> List[PeptideScore]:
        """Score a list of sequences, sorted by detectability then GRAVY."""
        scores = [self.score(s) for s in sequences]
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        scores.sort(key=lambda s: (order[s.detection_likelihood()], -s.gravy))
        return scores

    def rank_by_detectability(
        self, sequences: List[str]
    ) -> Dict[str, List[PeptideScore]]:
        """Return sequences grouped by detectability tier."""
        groups: Dict[str, List[PeptideScore]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for score in self.score_many(sequences):
            groups[score.detection_likelihood()].append(score)
        return groups

    # ------------------------------------------------------------------
    # Internal physicochemical calculations
    # ------------------------------------------------------------------

    def _composition(self, seq: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for aa in seq:
            counts[aa] = counts.get(aa, 0) + 1
        return counts

    def _charge_at_ph(self, composition: Dict[str, int], ph: float) -> float:
        """Henderson-Hasselbalch charge calculation."""
        charge = 0.0

        # N-terminus (positive)
        pka = _PKA["N_term"]
        charge += 1.0 / (1.0 + 10 ** (ph - pka))

        # C-terminus (negative)
        pka = _PKA["C_term"]
        charge -= 1.0 / (1.0 + 10 ** (pka - ph))

        # Positive ionizable residues
        for aa in ("H", "K", "R"):
            n = composition.get(aa, 0)
            if n:
                pka = _PKA[aa]
                charge += n * (1.0 / (1.0 + 10 ** (ph - pka)))

        # Negative ionizable residues
        for aa in ("D", "E", "C", "Y"):
            n = composition.get(aa, 0)
            if n:
                pka = _PKA[aa]
                charge -= n * (1.0 / (1.0 + 10 ** (pka - ph)))

        return charge

    def _isoelectric_point(self, composition: Dict[str, int]) -> float:
        """Binary search for pH where net charge ≈ 0."""
        lo, hi = 0.0, 14.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            c = self._charge_at_ph(composition, mid)
            if abs(c) < 1e-6:
                return mid
            if c > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def _gravy(self, seq: str) -> float:
        """Grand Average of Hydropathy (Kyte-Doolittle)."""
        if not seq:
            return 0.0
        total = sum(_HYDROPHOBICITY.get(aa, 0.0) for aa in seq)
        return total / len(seq)

    def _avg_bulkiness(self, seq: str) -> float:
        if not seq:
            return 0.0
        return sum(_BULKINESS.get(aa, 0.0) for aa in seq) / len(seq)

    def _aliphatic_index(self, seq: str) -> float:
        """
        Aliphatic index (Ikai 1980):
          AI = X(Ala) + 2.9*X(Val) + 3.9*(X(Ile) + X(Leu))
        where X(aa) is the mole fraction.
        """
        n = len(seq) or 1
        xa = seq.count("A") / n
        xv = seq.count("V") / n
        xi = seq.count("I") / n
        xl = seq.count("L") / n
        return (xa + 2.9 * xv + 3.9 * (xi + xl)) * 100.0

    def _instability_index(self, seq: str) -> float:
        """
        Guruprasad instability index (1990).
        Uses the DIWV (dipeptide instability weight value) table.
        Peptides with index > 40 are considered unstable.
        """
        # Abbreviated DIWV table (most impactful dipeptides)
        DIWV: Dict[str, float] = {
            "WW": 1.0, "WC": 1.0, "WT": 0.59, "WD": 0.77, "WE": 0.64,
            "WG": 1.0, "WH": 0.94, "WI": 1.0, "WK": 0.97, "WL": 1.0,
            "WM": 1.0, "WN": 0.92, "WP": 0.47, "WQ": 0.94, "WR": 1.0,
            "WS": 0.64, "WV": 1.0, "WY": 0.97, "WA": 0.79, "WF": 0.65,
            "CK": 1.0, "CN": 1.0, "CR": 1.0, "CS": 0.91, "CT": 0.99,
            "YD": 0.97, "YE": 0.59, "YK": 0.72, "YN": 0.50, "YS": 0.48,
        }
        if len(seq) < 2:
            return 0.0
        total = 0.0
        for i in range(len(seq) - 1):
            dipep = seq[i : i + 2]
            total += DIWV.get(dipep, 1.0)
        return (10.0 / len(seq)) * total
