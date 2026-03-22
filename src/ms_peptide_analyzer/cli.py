"""
cli.py
------
Command-line interface for ms-peptide-analyzer.

Sub-commands
------------
digest      - In-silico protein digestion
fragment    - Theoretical fragment ions for a peptide
isotopes    - Isotope envelope for a peptide
score       - Physicochemical scoring for one or more peptides
"""

from __future__ import annotations

import json
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich import box

from ms_peptide_analyzer.peptide import digest_protein, sequence_mass
from ms_peptide_analyzer.fragmentation import FragmentIonGenerator
from ms_peptide_analyzer.isotopes import IsotopeAnalyzer
from ms_peptide_analyzer.scorer import PeptideScorer

console = Console()


@click.group()
def main() -> None:
    """ms-peptide-analyzer — in-silico peptide analysis via pyOpenMS."""
    pass


# --------------------------------------------------------------------------
# digest
# --------------------------------------------------------------------------
@main.command()
@click.argument("protein_sequence")
@click.option("--enzyme", "-e", default="Trypsin", show_default=True,
              help="Enzyme name (Trypsin, LysC, AspN, GluC, ...)")
@click.option("--missed-cleavages", "-m", default=1, show_default=True,
              help="Maximum missed cleavages.")
@click.option("--min-len", default=6, show_default=True,
              help="Minimum peptide length.")
@click.option("--max-len", default=40, show_default=True,
              help="Maximum peptide length.")
@click.option("--charges", default="1,2,3", show_default=True,
              help="Comma-separated charge states.")
@click.option("--json-out", is_flag=True, help="Output as JSON.")
def digest(
    protein_sequence: str,
    enzyme: str,
    missed_cleavages: int,
    min_len: int,
    max_len: int,
    charges: str,
    json_out: bool,
) -> None:
    """Perform in-silico enzymatic digestion of PROTEIN_SEQUENCE."""
    charge_list = [int(c.strip()) for c in charges.split(",")]

    try:
        peptides = digest_protein(
            protein_sequence,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_len,
            max_length=max_len,
            charges=charge_list,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if json_out:
        data = [
            {
                "sequence": p.sequence,
                "start": p.start,
                "end": p.end,
                "charge": p.charge,
                "mono_mass": p.mono_mass,
                "mz": p.mz,
                "missed_cleavages": p.missed_cleavages,
                "formula": p.formula,
            }
            for p in peptides
        ]
        click.echo(json.dumps(data, indent=2))
        return

    table = Table(
        title=f"Digest: {enzyme}  MC≤{missed_cleavages}  len={min_len}-{max_len}",
        box=box.SIMPLE,
        show_header=True,
    )
    table.add_column("Peptide", style="cyan", no_wrap=True)
    table.add_column("Pos", justify="right")
    table.add_column("z", justify="center")
    table.add_column("Mono Mass (Da)", justify="right")
    table.add_column("[M+zH]/z m/z", justify="right")
    table.add_column("MC", justify="center")
    table.add_column("Formula")

    seen = set()
    for p in peptides:
        key = (p.sequence, p.charge)
        if key in seen:
            continue
        seen.add(key)
        table.add_row(
            p.sequence,
            f"{p.start}-{p.end}",
            str(p.charge),
            f"{p.mono_mass:.4f}",
            f"{p.mz:.4f}",
            str(p.missed_cleavages),
            p.formula,
        )

    console.print(table)
    console.print(f"[bold green]{len(seen)}[/bold green] unique peptide/charge combinations")


# --------------------------------------------------------------------------
# fragment
# --------------------------------------------------------------------------
@main.command()
@click.argument("peptide_sequence")
@click.option("--ions", "-i", default="b,y", show_default=True,
              help="Ion series (comma-separated, e.g. b,y,a).")
@click.option("--charge", "-z", default=2, show_default=True,
              help="Precursor charge.")
@click.option("--no-losses", is_flag=True, help="Disable neutral loss ions.")
@click.option("--json-out", is_flag=True, help="Output as JSON.")
def fragment(
    peptide_sequence: str,
    ions: str,
    charge: int,
    no_losses: bool,
    json_out: bool,
) -> None:
    """Generate theoretical fragment ions for PEPTIDE_SEQUENCE."""
    ion_list = [t.strip() for t in ions.split(",")]
    gen = FragmentIonGenerator(
        ion_types=ion_list,
        max_charge=charge,
        add_losses=not no_losses,
    )
    frag_ions = gen.generate(peptide_sequence, charge=charge)

    if json_out:
        click.echo(json.dumps(gen.as_dict(peptide_sequence, charge), indent=2))
        return

    table = Table(
        title=f"Fragment Ions: {peptide_sequence}  z≤{charge}",
        box=box.SIMPLE,
    )
    table.add_column("Ion", style="yellow")
    table.add_column("#", justify="right")
    table.add_column("z", justify="center")
    table.add_column("m/z", justify="right", style="green")
    table.add_column("Sub-sequence")

    for ion in frag_ions:
        table.add_row(ion.ion_type, str(ion.number), str(ion.charge),
                      f"{ion.mz:.4f}", ion.sequence)

    console.print(table)
    console.print(f"[bold green]{len(frag_ions)}[/bold green] ions generated")


# --------------------------------------------------------------------------
# isotopes
# --------------------------------------------------------------------------
@main.command()
@click.argument("peptide_sequence")
@click.option("--charge", "-z", default=2, show_default=True,
              help="Charge state for m/z calculation.")
@click.option("--max-peaks", default=5, show_default=True,
              help="Number of isotope peaks (0=monoisotopic).")
@click.option("--fine", is_flag=True,
              help="Use high-resolution FineIsotopePatternGenerator.")
@click.option("--json-out", is_flag=True, help="Output as JSON.")
def isotopes(
    peptide_sequence: str,
    charge: int,
    max_peaks: int,
    fine: bool,
    json_out: bool,
) -> None:
    """Compute the isotope envelope for PEPTIDE_SEQUENCE."""
    analyzer = IsotopeAnalyzer(max_isotopes=max_peaks, use_fine=fine)
    summary = analyzer.envelope_summary(peptide_sequence, charge)

    if json_out:
        click.echo(json.dumps(summary, indent=2))
        return

    table = Table(
        title=f"Isotope Envelope: {peptide_sequence}  z={charge}",
        box=box.SIMPLE,
    )
    table.add_column("Isotope", style="yellow")
    table.add_column("m/z", justify="right", style="green")
    table.add_column("Relative Intensity", justify="right")
    table.add_column("Bar")

    for pk in summary["peaks"]:
        bar_len = int(pk["rel_intensity"] * 30)
        bar = "█" * bar_len
        table.add_row(
            f"M+{pk['index']}",
            f"{pk['mz']:.5f}",
            f"{pk['rel_intensity']:.4f}",
            f"[cyan]{bar}[/cyan]",
        )

    console.print(table)
    console.print(f"Formula: [bold]{summary['formula']}[/bold]")
    console.print(
        f"Monoisotopic m/z (z={charge}): [green]{summary['monoisotopic_mz']:.5f}[/green]  "
        f"Most abundant: [green]{summary['most_abundant_mz']:.5f}[/green] "
        f"(M+{summary['most_abundant_isotope']})"
    )


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
@main.command()
@click.argument("sequences", nargs=-1, required=True)
@click.option("--json-out", is_flag=True, help="Output as JSON.")
def score(sequences: tuple, json_out: bool) -> None:
    """Score physicochemical properties for one or more SEQUENCES."""
    scorer = PeptideScorer()
    results = scorer.score_many(list(sequences))

    if json_out:
        data = [
            {
                "sequence": s.sequence,
                "length": s.length,
                "mono_mass": s.mono_mass,
                "pI": s.isoelectric_point,
                "charge_at_pH7": s.charge_at_ph7,
                "gravy": s.gravy,
                "instability_index": s.instability_index,
                "aromaticity": s.aromaticity,
                "aliphatic_index": s.aliphatic_index,
                "detectability": s.detection_likelihood(),
            }
            for s in results
        ]
        click.echo(json.dumps(data, indent=2))
        return

    table = Table(title="Peptide Physicochemical Scores", box=box.SIMPLE)
    table.add_column("Sequence", style="cyan", no_wrap=True)
    table.add_column("Len", justify="right")
    table.add_column("Mass (Da)", justify="right")
    table.add_column("pI", justify="right")
    table.add_column("z@7", justify="right")
    table.add_column("GRAVY", justify="right")
    table.add_column("Instability", justify="right")
    table.add_column("Detectability", justify="center")

    detect_colors = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}
    for s in results:
        d = s.detection_likelihood()
        color = detect_colors[d]
        table.add_row(
            s.sequence,
            str(s.length),
            f"{s.mono_mass:.3f}",
            f"{s.isoelectric_point:.2f}",
            f"{s.charge_at_ph7:.2f}",
            f"{s.gravy:.3f}",
            f"{s.instability_index:.1f}",
            f"[{color}]{d}[/{color}]",
        )

    console.print(table)


if __name__ == "__main__":
    main()
