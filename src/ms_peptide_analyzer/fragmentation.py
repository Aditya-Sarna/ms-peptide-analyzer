"""
fragmentation.py
----------------
Theoretical b/y/a/c/z fragment ion generation via pyOpenMS bindings.

Key pyOpenMS C++ bindings used (via nanobind/pybind11):
  - pyopenms.TheoreticalSpectrumGenerator  : generates theoretical MS2 spectra
  - pyopenms.MSSpectrum                    : stores peaks (m/z, intensity)
  - pyopenms.AASequence                    : amino-acid sequence object
  - pyopenms.Param                         : parameter container for config
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import pyopenms as oms


ION_TYPES = Literal["b", "y", "a", "c", "z"]


@dataclass
class FragmentIon:
    """A single theoretical fragment ion."""
    ion_type: str       # b, y, a, c, z
    number: int         # position index (1-based)
    charge: int
    mz: float
    sequence: str       # sub-sequence this ion covers

    def __str__(self) -> str:
        z_str = f"({self.charge}+)" if self.charge > 1 else ""
        return f"{self.ion_type}{self.number}{z_str}  m/z={self.mz:.4f}  [{self.sequence}]"


class FragmentIonGenerator:
    """
    Wraps pyOpenMS TheoreticalSpectrumGenerator to produce fragment ions.

    The C++ TheoreticalSpectrumGenerator is accessed through the
    pyOpenMS pybind11/nanobind binding layer.

    Parameters
    ----------
    ion_types : list[str]
        Ion series to compute. Subset of ["b", "y", "a", "c", "z"].
    max_charge : int
        Maximum fragment charge state (1–4).
    add_losses : bool
        Include NH3 and H2O neutral losses.
    add_metainfo : bool
        Attach ion-type/number metadata to each peak (used for annotation).
    """

    def __init__(
        self,
        ion_types: List[str] = None,
        max_charge: int = 1,
        add_losses: bool = True,
        add_metainfo: bool = True,
    ) -> None:
        if ion_types is None:
            ion_types = ["b", "y"]

        self.ion_types = ion_types
        self.max_charge = max_charge
        self.add_losses = add_losses
        self.add_metainfo = add_metainfo

        # Build the C++ TheoreticalSpectrumGenerator and configure via Param
        self._tsg = oms.TheoreticalSpectrumGenerator()
        params = self._tsg.getParameters()

        # Ion series flags
        params.setValue("add_b_ions", "true" if "b" in ion_types else "false")
        params.setValue("add_y_ions", "true" if "y" in ion_types else "false")
        params.setValue("add_a_ions", "true" if "a" in ion_types else "false")
        params.setValue("add_c_ions", "true" if "c" in ion_types else "false")
        params.setValue("add_z_ions", "true" if "z" in ion_types else "false")

        # Neutral losses
        params.setValue("add_losses", "true" if add_losses else "false")

        # Metainfo (ion name annotations on peaks)
        params.setValue("add_metainfo", "true" if add_metainfo else "false")

        # Precursor peak not needed in theoretical spectra
        params.setValue("add_precursor_peaks", "false")

        self._tsg.setParameters(params)

    def generate(
        self,
        sequence: str,
        charge: int = 1,
    ) -> List[FragmentIon]:
        """
        Generate theoretical fragment ions for a peptide sequence.

        Parameters
        ----------
        sequence : str
            One-letter peptide sequence (may include modifications in
            OpenMS bracket notation, e.g. "PEPTM(Oxidation)IDE").
        charge : int
            Precursor charge. Fragment charges will range from 1..min(charge,max_charge).

        Returns
        -------
        list[FragmentIon]
            Sorted list of fragment ions by m/z.
        """
        aa_seq = oms.AASequence.fromString(sequence)
        spectrum = oms.MSSpectrum()

        # The C++ call via nanobind binding
        self._tsg.getSpectrum(spectrum, aa_seq, 1, min(charge, self.max_charge))

        n = len(sequence.replace("(", "").replace(")", ""))  # approx length
        ions: List[FragmentIon] = []

        for peak in spectrum:
            mz = peak.getMZ()
            annotation = ""

            # Extract annotation from metainfo if available
            if self.add_metainfo:
                try:
                    fl = oms.StringList()
                    peak.getMetaValue("IonNames", fl) if hasattr(peak, "getMetaValue") else None
                except Exception:
                    pass

            ions.append(
                FragmentIon(
                    ion_type="?",
                    number=0,
                    charge=1,
                    mz=mz,
                    sequence="",
                )
            )

        # Re-generate with manual annotation for rich output
        return self._annotate(aa_seq, min(charge, self.max_charge))

    def _annotate(self, aa_seq: oms.AASequence, max_z: int) -> List[FragmentIon]:
        """
        Manually compute and annotate each fragment ion using AASequence slicing.

        This calls the C++ AASequence.getPrefix / getSuffix methods via the
        nanobind binding to get sub-sequence masses directly.
        """
        PROTON = 1.007276
        ions: List[FragmentIon] = []
        n = aa_seq.size()
        seq_str = aa_seq.toUnmodifiedString()

        for z in range(1, max_z + 1):
            for i in range(1, n):
                # N-terminal series (b, a, c)
                prefix = aa_seq.getPrefix(i)
                prefix_sub = seq_str[:i]

                if "b" in self.ion_types:
                    mass = prefix.getMonoWeight(oms.Residue.ResidueType.BIon, z)
                    ions.append(FragmentIon("b", i, z, mass / z + PROTON * (z - 1) / z, prefix_sub))

                if "a" in self.ion_types:
                    mass = prefix.getMonoWeight(oms.Residue.ResidueType.AIon, z)
                    ions.append(FragmentIon("a", i, z, mass / z + PROTON * (z - 1) / z, prefix_sub))

                if "c" in self.ion_types:
                    mass = prefix.getMonoWeight(oms.Residue.ResidueType.CIon, z)
                    ions.append(FragmentIon("c", i, z, mass / z + PROTON * (z - 1) / z, prefix_sub))

                # C-terminal series (y, z)
                suffix = aa_seq.getSuffix(n - i)
                suffix_sub = seq_str[i:]

                if "y" in self.ion_types:
                    mass = suffix.getMonoWeight(oms.Residue.ResidueType.YIon, z)
                    ions.append(FragmentIon("y", n - i, z, mass / z + PROTON * (z - 1) / z, suffix_sub))

                if "z" in self.ion_types:
                    mass = suffix.getMonoWeight(oms.Residue.ResidueType.ZIon, z)
                    ions.append(FragmentIon("z", n - i, z, mass / z + PROTON * (z - 1) / z, suffix_sub))

        ions.sort(key=lambda x: x.mz)
        return ions

    def as_dict(self, sequence: str, charge: int = 1) -> Dict[str, List[dict]]:
        """
        Return fragment ions grouped by ion type as plain dicts.

        Useful for JSON serialization / downstream scoring.
        """
        ions = self.generate(sequence, charge)
        result: Dict[str, List[dict]] = {}
        for ion in ions:
            result.setdefault(ion.ion_type, []).append(
                {"number": ion.number, "charge": ion.charge, "mz": ion.mz, "sub_seq": ion.sequence}
            )
        return result

    def ladder(self, sequence: str) -> Dict[str, List[float]]:
        """
        Return simple m/z ladders for each ion series (z=1 only).

        Returns
        -------
        dict mapping ion_type -> sorted list of m/z values
        """
        ions = self.generate(sequence, charge=1)
        ladders: Dict[str, List[float]] = {}
        for ion in ions:
            if ion.charge == 1:
                ladders.setdefault(ion.ion_type, []).append(ion.mz)
        for v in ladders.values():
            v.sort()
        return ladders
