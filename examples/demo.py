#!/usr/bin/env python3
"""
demo.py
-------
End-to-end demonstration of ms-peptide-analyzer using pyOpenMS
nanobind/pybind11 C++ bindings.

Run with:
    python examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.rule import Rule

from ms_peptide_analyzer.peptide import digest_protein, sequence_mass
from ms_peptide_analyzer.fragmentation import FragmentIonGenerator
from ms_peptide_analyzer.isotopes import IsotopeAnalyzer
from ms_peptide_analyzer.scorer import PeptideScorer

console = Console()

# ─── Demo protein: Human Ubiquitin (76 AA) ────────────────────────────────────
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

def main() -> None:
    console.print(Rule("[bold cyan]ms-peptide-analyzer · pyOpenMS nanobind demo[/bold cyan]"))
    console.print(f"Protein: Human Ubiquitin ({len(UBIQUITIN)} AA)\n")

    # ── 1. In-silico tryptic digestion ────────────────────────────────────────
    console.print(Rule("[yellow]1. Tryptic Digestion[/yellow]"))
    peptides = digest_protein(UBIQUITIN, enzyme="Trypsin", missed_cleavages=1, charges=[1, 2])
    # Show only unique sequences (charge=2)
    shown = {p.sequence: p for p in peptides if p.charge == 2}
    for seq, p in list(shown.items())[:8]:
        console.print(f"  [cyan]{p.sequence:<25}[/cyan] "
                      f"mono={p.mono_mass:.3f} Da  "
                      f"[M+2H]²⁺={p.mz:.4f}  MC={p.missed_cleavages}")
    console.print(f"  … {len(shown)} unique tryptic peptides total\n")

    # ── 2. Fragment ions ──────────────────────────────────────────────────────
    target = "LIFAGK"
    console.print(Rule(f"[yellow]2. Fragment Ions · {target}[/yellow]"))
    gen = FragmentIonGenerator(ion_types=["b", "y", "a"], max_charge=1)
    ions = gen.generate(target, charge=1)
    for ion in ions:
        bar = "█" * int((ion.mz / max(i.mz for i in ions)) * 20)
        console.print(f"  [yellow]{ion.ion_type}{ion.number}[/yellow]  "
                      f"m/z=[green]{ion.mz:.4f}[/green]  [{ion.sequence}]  {bar}")
    console.print()

    # ── 3. Isotope envelope ───────────────────────────────────────────────────
    target2 = "TLSDYNIQK"
    console.print(Rule(f"[yellow]3. Isotope Envelope · {target2}  z=2[/yellow]"))
    iso = IsotopeAnalyzer(max_isotopes=5)
    summary = iso.envelope_summary(target2, charge=2)
    console.print(f"  Formula: [bold]{summary['formula']}[/bold]")
    for pk in summary["peaks"]:
        bar = "█" * int(pk["rel_intensity"] * 30)
        console.print(f"  M+{pk['index']}  m/z=[green]{pk['mz']:.5f}[/green]  "
                      f"[cyan]{bar}[/cyan]  {pk['rel_intensity']:.3f}")
    console.print()

    # ── 4. Physicochemical scoring ────────────────────────────────────────────
    console.print(Rule("[yellow]4. Physicochemical Scoring[/yellow]"))
    scorer = PeptideScorer()
    candidates = [p.sequence for p in list(shown.values())[:10]]
    groups = scorer.rank_by_detectability(candidates)
    for tier, color in [("HIGH", "green"), ("MEDIUM", "yellow"), ("LOW", "red")]:
        for s in groups[tier]:
            console.print(
                f"  [{color}]{tier:6}[/{color}]  [cyan]{s.sequence:<22}[/cyan]  "
                f"pI={s.isoelectric_point:.1f}  GRAVY={s.gravy:.3f}  "
                f"Instab={s.instability_index:.1f}"
            )
    console.print()
    console.print(Rule("[bold green]Done[/bold green]"))


if __name__ == "__main__":
    main()
