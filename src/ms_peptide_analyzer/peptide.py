"""
peptide.py
----------
Protein digestion and peptide sequence utilities using pyOpenMS bindings.

Key pyOpenMS C++ bindings used (accessed via nanobind/pybind11):
  - pyopenms.AASequence          : amino-acid sequence + modifications
  - pyopenms.ProteaseDigestion   : enzymatic in-silico digestion
  - pyopenms.EmpiricalFormula    : molecular formula arithmetic
  - pyopenms.Residue             : individual amino-acid residue info
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import pyopenms as oms


@dataclass
class PeptideInfo:
    """Structured result for a single digested peptide."""
    sequence: str
    start: int
    end: int
    charge: int
    mono_mass: float
    avg_mass: float
    formula: str
    missed_cleavages: int
    modifications: List[str] = field(default_factory=list)

    @property
    def mz(self) -> float:
        """[M+H]+ m/z (singly protonated)."""
        PROTON = 1.007276
        return (self.mono_mass + self.charge * PROTON) / self.charge

    def __str__(self) -> str:
        mods = f"  mods={self.modifications}" if self.modifications else ""
        return (
            f"{self.sequence} | {self.start}-{self.end} | "
            f"z={self.charge} | mono={self.mono_mass:.4f} Da | "
            f"m/z={self.mz:.4f} | MC={self.missed_cleavages}{mods}"
        )


def digest_protein(
    protein_sequence: str,
    enzyme: str = "Trypsin",
    missed_cleavages: int = 1,
    min_length: int = 6,
    max_length: int = 40,
    charges: List[int] = None,
) -> List[PeptideInfo]:
    """
    Perform in-silico enzymatic digestion of a protein sequence.

    Uses pyOpenMS ProteaseDigestion (C++ binding) to enumerate peptides,
    then wraps each into a PeptideInfo with masses from AASequence.

    Parameters
    ----------
    protein_sequence : str
        One-letter amino-acid string (uppercase).
    enzyme : str
        Enzyme name recognized by pyOpenMS (e.g. "Trypsin", "LysC", "AspN").
    missed_cleavages : int
        Maximum number of missed cleavage sites allowed.
    min_length : int
        Minimum peptide length (residues).
    max_length : int
        Maximum peptide length (residues).
    charges : list[int]
        Charge states to annotate. Defaults to [1, 2, 3].

    Returns
    -------
    list[PeptideInfo]
        Deduplicated list of peptide entries sorted by start position.
    """
    if charges is None:
        charges = [1, 2, 3]

    # Sanitize – remove whitespace and unknown characters
    protein_sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", protein_sequence.upper())
    if not protein_sequence:
        raise ValueError("Protein sequence contains no valid amino-acid characters.")

    # Bind the C++ ProteaseDigestion object through pyOpenMS nanobind layer
    digester = oms.ProteaseDigestion()
    digester.setEnzyme(enzyme)
    digester.setMissedCleavages(missed_cleavages)

    protein_aa = oms.AASequence.fromString(protein_sequence)
    peptide_seqs: List[oms.AASequence] = []
    digester.digest(protein_aa, peptide_seqs, min_length, max_length)

    results: List[PeptideInfo] = []

    for pep_aa in peptide_seqs:
        seq_str = pep_aa.toUnmodifiedString()

        # Locate all occurrences of this peptide in the parent protein
        start = 0
        while True:
            pos = protein_sequence.find(seq_str, start)
            if pos == -1:
                break
            end = pos + len(seq_str) - 1

            # Count missed cleavages (K/R not at C-terminus for Trypsin-like)
            mc = _count_missed_cleavages(seq_str, enzyme)

            # Molecular masses via C++ AASequence binding
            mono = pep_aa.getMonoWeight(oms.Residue.ResidueType.Full, 0)
            avg = pep_aa.getAverageWeight(oms.Residue.ResidueType.Full, 0)

            # Molecular formula via C++ EmpiricalFormula binding
            formula_obj: oms.EmpiricalFormula = pep_aa.getFormula(
                oms.Residue.ResidueType.Full, 0
            )
            formula_str = formula_obj.toString()

            # Modifications (if any fixed/variable mods were set)
            mods = []
            for i in range(pep_aa.size()):
                res = pep_aa[i]
                if res.isModified():
                    mods.append(f"{res.getOneLetterCode()}@{i+1}:{res.getModificationName()}")

            for z in charges:
                results.append(
                    PeptideInfo(
                        sequence=seq_str,
                        start=pos,
                        end=end,
                        charge=z,
                        mono_mass=mono,
                        avg_mass=avg,
                        formula=formula_str,
                        missed_cleavages=mc,
                        modifications=mods,
                    )
                )
            start = pos + 1

    # Sort by start position, then by length
    results.sort(key=lambda p: (p.start, len(p.sequence)))
    return results


def _count_missed_cleavages(seq: str, enzyme: str) -> int:
    """Heuristic missed-cleavage counter for common enzymes."""
    if enzyme in ("Trypsin", "Trypsin/P"):
        # K or R not at the last position
        return sum(1 for aa in seq[:-1] if aa in ("K", "R"))
    if enzyme in ("LysC", "Lys-C"):
        return sum(1 for aa in seq[:-1] if aa == "K")
    if enzyme == "AspN":
        return sum(1 for aa in seq[1:] if aa == "D")
    if enzyme == "GluC":
        return sum(1 for aa in seq[:-1] if aa in ("E", "D"))
    return 0


def add_fixed_modification(sequence: str, modification: str, residue: str) -> str:
    """
    Return a modified sequence string with a fixed modification applied.

    Example
    -------
    >>> add_fixed_modification("ACDEFGHIK", "Carbamidomethyl", "C")
    'AC(Carbamidomethyl)DEFGHIK'
    """
    result = []
    for aa in sequence:
        if aa == residue:
            result.append(f"{aa}({modification})")
        else:
            result.append(aa)
    return "".join(result)


def sequence_mass(sequence: str) -> dict:
    """
    Return mono and average masses for a bare peptide string.

    Uses pyOpenMS AASequence C++ binding directly.
    """
    aa = oms.AASequence.fromString(sequence)
    return {
        "sequence": sequence,
        "monoisotopic_mass": aa.getMonoWeight(oms.Residue.ResidueType.Full, 0),
        "average_mass": aa.getAverageWeight(oms.Residue.ResidueType.Full, 0),
        "formula": aa.getFormula(oms.Residue.ResidueType.Full, 0).toString(),
        "length": aa.size(),
    }
