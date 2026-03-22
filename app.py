"""
app.py
------
Streamlit UI for ms-peptide-analyzer.
Plain black-and-white, no emojis.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ms_peptide_analyzer.peptide import digest_protein, sequence_mass
from ms_peptide_analyzer.fragmentation import FragmentIonGenerator
from ms_peptide_analyzer.isotopes import IsotopeAnalyzer
from ms_peptide_analyzer.scorer import PeptideScorer

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peptide Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS — strict black/white, no rounded widgets, monospace data ──────
st.markdown(
    """
    <style>
    /* ---- base ---- */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="block-container"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
        border-right: 1px solid #000 !important;
    }
    [data-testid="stSidebar"] * { color: #000 !important; }

    /* ---- headings ---- */
    h1, h2, h3, h4 { color: #000 !important; font-weight: 700; }
    h1 { font-size: 1.6rem; border-bottom: 2px solid #000; padding-bottom: 6px; margin-bottom: 16px; }
    h2 { font-size: 1.2rem; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-bottom: 12px; }

    /* ---- tabs ---- */
    [data-testid="stTabs"] button {
        color: #000 !important;
        border-radius: 0 !important;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-bottom: 2px solid transparent;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        border-bottom: 2px solid #000 !important;
        background: #fff !important;
    }
    [data-testid="stTabs"] button:hover { background: #f0f0f0 !important; }

    /* ---- inputs ---- */
    input, textarea, select,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input {
        background-color: #fff !important;
        color: #000 !important;
        border: 1px solid #000 !important;
        border-radius: 0 !important;
        font-family: "Courier New", monospace !important;
    }
    input:focus, textarea:focus { outline: 2px solid #000 !important; }

    /* ---- slider ---- */
    [data-testid="stSlider"] > div > div > div {
        background-color: #000 !important;
    }

    /* ---- selectbox / multiselect ---- */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background: #fff !important;
        border: 1px solid #000 !important;
        border-radius: 0 !important;
        color: #000 !important;
    }

    /* ---- buttons ---- */
    button[kind="primary"], button[kind="secondary"],
    [data-testid="baseButton-primary"], [data-testid="baseButton-secondary"] {
        background-color: #000 !important;
        color: #fff !important;
        border: 1px solid #000 !important;
        border-radius: 0 !important;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    button[kind="secondary"],
    [data-testid="baseButton-secondary"] {
        background-color: #fff !important;
        color: #000 !important;
    }
    button:hover { opacity: 0.8; }

    /* ---- dataframes / tables ---- */
    [data-testid="stDataFrame"] table,
    [data-testid="stTable"] table { width: 100%; border-collapse: collapse; }
    [data-testid="stDataFrame"] th,
    [data-testid="stTable"] th {
        background: #000 !important;
        color: #fff !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 6px 10px;
        border: 1px solid #000;
    }
    [data-testid="stDataFrame"] td,
    [data-testid="stTable"] td {
        font-family: "Courier New", monospace;
        font-size: 0.82rem;
        padding: 5px 10px;
        border: 1px solid #ccc;
    }
    [data-testid="stDataFrame"] tr:nth-child(even) td { background: #f8f8f8; }

    /* ---- metric cards ---- */
    [data-testid="stMetric"] {
        border: 1px solid #000;
        padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.7rem; text-transform: uppercase;
                                    letter-spacing: 0.08em; color: #555 !important; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #000 !important; }
    [data-testid="stMetricDelta"] { color: #000 !important; }

    /* ---- info / warning / error boxes ---- */
    [data-testid="stAlert"] {
        border-radius: 0 !important;
        border-left: 4px solid #000 !important;
        background: #f5f5f5 !important;
        color: #000 !important;
    }

    /* ---- code blocks ---- */
    code, pre { background: #f0f0f0 !important; color: #000 !important;
                border: 1px solid #ccc; border-radius: 0 !important; }

    /* ---- dividers ---- */
    hr { border: 0; border-top: 1px solid #000; margin: 18px 0; }

    /* ---- matplotlib plots — injected via fig ---- */
    canvas { background: #fff; }

    /* ---- progress bar ---- */
    [data-testid="stProgressBar"] > div { background-color: #000 !important; }

    /* ---- hide Streamlit chrome ---- */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

    /* ---- pandas to_html tables ---- */
    table.dataframe, table {
        width: 100%;
        border-collapse: collapse;
        font-family: "Courier New", monospace;
        font-size: 0.82rem;
        margin-top: 6px;
    }
    table.dataframe thead tr th, table thead tr th {
        background: #000 !important;
        color: #fff !important;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 6px 10px;
        border: 1px solid #000;
        text-align: left;
    }
    table.dataframe tbody tr td, table tbody tr td {
        padding: 5px 10px;
        border: 1px solid #ccc;
        color: #000;
    }
    table.dataframe tbody tr:nth-child(even) td,
    table tbody tr:nth-child(even) td { background: #f8f8f8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Matplotlib global style — black/white ────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "grid.color":        "#cccccc",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
})


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Peptide Analyzer")
    st.markdown("---")
    st.markdown(
        "<small>In-silico peptide analysis<br>powered by pyOpenMS C++ bindings</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Modules**")
    st.markdown(
        "<ul style='font-size:0.8rem;padding-left:16px;line-height:2'>"
        "<li>ProteaseDigestion</li>"
        "<li>TheoreticalSpectrumGenerator</li>"
        "<li>CoarseIsotopePatternGenerator</li>"
        "<li>AASequence</li>"
        "<li>EmpiricalFormula</li>"
        "</ul>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("<small style='color:#888'>pyOpenMS nanobind/pybind11</small>", unsafe_allow_html=True)


# ─── Title ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='margin-bottom:0px;'>"
    "<h1 style='margin-bottom:16px;'>Peptide Analyzer</h1>"
    "<p style='font-size:0.88rem;color:#555;margin-top:0;margin-bottom:24px;"
    "letter-spacing:0.12em;word-spacing:0.35em;line-height:1.8;'>"
    "In-silico digestion,&nbsp;&nbsp;fragmentation,&nbsp;&nbsp;"
    "isotope distribution,&nbsp;&nbsp;and physicochemical scoring</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_digest, tab_frag, tab_iso, tab_score = st.tabs([
    "Protein Digestion",
    "Fragment Ions",
    "Isotope Envelope",
    "Peptide Scoring",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Protein Digestion
# ══════════════════════════════════════════════════════════════════════════════
with tab_digest:
    st.markdown("## Enzymatic Digestion")

    col_in, col_opts = st.columns([3, 2], gap="large")

    with col_in:
        protein_seq = st.text_area(
            "Protein sequence (single-letter codes)",
            value="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            height=110,
            help="Paste any protein sequence. Non-standard characters are stripped.",
        )

    with col_opts:
        enzyme = st.selectbox(
            "Enzyme",
            ["Trypsin", "Lys-C", "Asp-N", "Glu-C", "no cleavage"],
            index=0,
        )
        mc = st.slider("Max missed cleavages", 0, 3, 1)
        min_len = st.slider("Min peptide length", 2, 10, 6)
        max_len = st.slider("Max peptide length", 10, 60, 40)
        charges = st.multiselect("Charge states", [1, 2, 3, 4], default=[1, 2])

    run_digest = st.button("Run Digestion", key="btn_digest")

    if run_digest:
        if not protein_seq.strip():
            st.warning("Please enter a protein sequence.")
        elif not charges:
            st.warning("Select at least one charge state.")
        else:
            with st.spinner("Digesting..."):
                try:
                    peptides = digest_protein(
                        protein_seq,
                        enzyme=enzyme,
                        missed_cleavages=mc,
                        min_length=min_len,
                        max_length=max_len,
                        charges=charges,
                    )
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

            # Deduplicate for display (one row per sequence × charge)
            seen = {}
            for p in peptides:
                k = (p.sequence, p.charge)
                if k not in seen:
                    seen[k] = p
            unique = list(seen.values())

            # Summary metrics
            unique_seqs = len({p.sequence for p in unique})
            m1 = st.columns(4)
            m1[0].metric("Unique peptides", unique_seqs)
            m1[1].metric("Peptide/charge pairs", len(unique))
            m1[2].metric(
                "Mass range (Da)",
                f"{min(p.mono_mass for p in unique):.0f} - {max(p.mono_mass for p in unique):.0f}",
            )
            m1[3].metric(
                "Length range",
                f"{min(len(p.sequence) for p in unique)} - {max(len(p.sequence) for p in unique)}",
            )

            st.markdown("---")

            # Table
            import pandas as pd
            rows = []
            for p in unique:
                rows.append({
                    "Peptide": p.sequence,
                    "Start": p.start,
                    "End": p.end,
                    "Length": len(p.sequence),
                    "z": p.charge,
                    "Mono Mass (Da)": round(p.mono_mass, 4),
                    "m/z": round(p.mz, 4),
                    "MC": p.missed_cleavages,
                    "Formula": p.formula,
                })
            df = pd.DataFrame(rows)
            st.markdown(df.to_html(index=False), unsafe_allow_html=True)

            # Mass distribution plot
            st.markdown("---")
            st.markdown("**Monoisotopic mass distribution**")
            masses_z1 = [p.mono_mass for p in unique if p.charge == charges[0]]
            if masses_z1:
                fig, ax = plt.subplots(figsize=(8, 2.8))
                ax.hist(masses_z1, bins=min(30, max(5, len(masses_z1) // 2)),
                        color="black", edgecolor="white", linewidth=0.4)
                ax.set_xlabel("Monoisotopic mass (Da)")
                ax.set_ylabel("Count")
                ax.set_title(f"Mass distribution  (z={charges[0]})")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fragment Ions
# ══════════════════════════════════════════════════════════════════════════════
with tab_frag:
    st.markdown("## Theoretical Fragment Ions")

    col_f1, col_f2 = st.columns([3, 2], gap="large")

    with col_f1:
        frag_seq = st.text_input(
            "Peptide sequence",
            value="LGGNEQVTR",
            help="One-letter codes. Use OpenMS bracket notation for modifications, e.g. M(Oxidation).",
        )

    with col_f2:
        ion_series = st.multiselect(
            "Ion series",
            ["b", "y", "a", "c", "z"],
            default=["b", "y"],
        )
        frag_charge = st.slider("Max precursor charge", 1, 4, 2, key="frag_z")
        add_losses = st.checkbox("Include neutral losses (NH3, H2O)", value=True)

    run_frag = st.button("Generate Ions", key="btn_frag")

    if run_frag:
        if not frag_seq.strip():
            st.warning("Please enter a peptide sequence.")
        elif not ion_series:
            st.warning("Select at least one ion series.")
        else:
            with st.spinner("Generating..."):
                gen = FragmentIonGenerator(
                    ion_types=ion_series,
                    max_charge=frag_charge,
                    add_losses=add_losses,
                )
                ions = gen.generate(frag_seq, charge=frag_charge)

            # Metrics
            m2 = st.columns(3)
            m2[0].metric("Total ions", len(ions))
            m2[1].metric("Ion series", ", ".join(ion_series))
            m2[2].metric("Max charge", frag_charge)

            st.markdown("---")

            # Stick plot
            st.markdown("**Fragment ion spectrum (stick plot)**")
            ion_colors = {"b": 0.0, "y": 0.55, "a": 0.25, "c": 0.75, "z": 0.9}
            cmap = plt.cm.Greys

            fig2, ax2 = plt.subplots(figsize=(10, 3.2))
            for ion in ions:
                x = ion.mz
                c = ion_colors.get(ion.ion_type, 0.5)
                gray = str(c)
                ax2.vlines(x, 0, 1.0, colors=gray, linewidth=1.2)
                if ion.number <= 3 or ion.number >= len(frag_seq) - 2:
                    ax2.text(x, 1.02,
                             f"{ion.ion_type}{ion.number}{'+'*ion.charge if ion.charge>1 else ''}",
                             ha="center", va="bottom", fontsize=7, color="black",
                             rotation=90)

            ax2.set_xlabel("m/z")
            ax2.set_ylabel("Relative intensity")
            ax2.set_ylim(0, 1.5)
            ax2.set_title(f"Theoretical spectrum: {frag_seq}")
            # Legend
            for label, val in ion_colors.items():
                if label in ion_series:
                    ax2.vlines([], 0, 0, colors=str(val), linewidth=2, label=label)
            ax2.legend(frameon=True, fontsize=8)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.markdown("---")
            st.markdown("**Ion table**")

            import pandas as pd
            ion_rows = [
                {
                    "Ion": f"{i.ion_type}{i.number}",
                    "Type": i.ion_type,
                    "Number": i.number,
                    "Charge": i.charge,
                    "m/z": round(i.mz, 4),
                    "Sub-sequence": i.sequence,
                }
                for i in ions
            ]
            df2 = pd.DataFrame(ion_rows)
            st.markdown(df2.to_html(index=False), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Isotope Envelope
# ══════════════════════════════════════════════════════════════════════════════
with tab_iso:
    st.markdown("## Isotope Distribution")

    col_i1, col_i2 = st.columns([3, 2], gap="large")

    with col_i1:
        iso_seq = st.text_input(
            "Peptide sequence",
            value="TLSDYNIQK",
            key="iso_seq",
        )

    with col_i2:
        iso_z = st.slider("Charge state", 1, 6, 2, key="iso_z")
        iso_peaks = st.slider("Max isotope peaks", 2, 10, 6, key="iso_peaks")
        use_fine = st.checkbox("High-resolution (FineIsotopePatternGenerator)", value=False)

    run_iso = st.button("Compute Envelope", key="btn_iso")

    if run_iso:
        if not iso_seq.strip():
            st.warning("Please enter a peptide sequence.")
        else:
            with st.spinner("Computing..."):
                analyzer = IsotopeAnalyzer(max_isotopes=iso_peaks, use_fine=use_fine)
                try:
                    summary = analyzer.envelope_summary(iso_seq, iso_z)
                    peaks = summary["peaks"]
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            # Metrics
            m3 = st.columns(4)
            m3[0].metric("Formula", summary["formula"])
            m3[1].metric("Mono m/z", f"{summary['monoisotopic_mz']:.5f}")
            m3[2].metric("Most abundant m/z", f"{summary['most_abundant_mz']:.5f}")
            m3[3].metric("Most abundant isotope", f"M+{summary['most_abundant_isotope']}")

            st.markdown("---")

            # Stick plot
            st.markdown("**Isotope envelope**")
            mzs = [p["mz"] for p in peaks]
            ints = [p["rel_intensity"] for p in peaks]

            fig3, ax3 = plt.subplots(figsize=(8, 3.2))
            ax3.vlines(mzs, 0, ints, colors="black", linewidth=2.5)
            ax3.scatter(mzs, ints, color="black", s=18, zorder=3)
            for i, (x, y) in enumerate(zip(mzs, ints)):
                ax3.text(x, y + 0.02, f"M+{i}", ha="center", va="bottom",
                         fontsize=8, color="black")
            ax3.set_xlabel("m/z")
            ax3.set_ylabel("Relative intensity")
            ax3.set_ylim(0, 1.25)
            ax3.set_title(f"Isotope envelope: {iso_seq}  z={iso_z}")
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            st.markdown("---")
            st.markdown("**Peak table**")
            import pandas as pd
            iso_df = pd.DataFrame([
                {"Isotope": f"M+{p['index']}", "m/z": p["mz"],
                 "Relative Intensity": p["rel_intensity"]}
                for p in peaks
            ])
            st.markdown(iso_df.to_html(index=False), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Peptide Scoring
# ══════════════════════════════════════════════════════════════════════════════
with tab_score:
    st.markdown("## Physicochemical Scoring")

    sequences_input = st.text_area(
        "Peptide sequences (one per line)",
        value="LGGNEQVTR\nPEPTIDE\nIFAGK\nTLSDYNIQK\nEGIPPDQQR\nKKKKKKKK\nLLLLLLLL\nMQIFVK",
        height=160,
        help="Enter one peptide per line.",
    )

    run_score = st.button("Score Peptides", key="btn_score")

    if run_score:
        seqs = [s.strip().upper() for s in sequences_input.strip().splitlines() if s.strip()]
        if not seqs:
            st.warning("Please enter at least one sequence.")
        else:
            with st.spinner("Scoring..."):
                scorer = PeptideScorer()
                scores = scorer.score_many(seqs)
                groups = scorer.rank_by_detectability(seqs)

            # Summary metrics
            m4 = st.columns(3)
            m4[0].metric("HIGH detectability", len(groups["HIGH"]))
            m4[1].metric("MEDIUM detectability", len(groups["MEDIUM"]))
            m4[2].metric("LOW detectability", len(groups["LOW"]))

            st.markdown("---")

            # Score table
            import pandas as pd
            score_rows = []
            detect_map = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW"}
            for s in scores:
                score_rows.append({
                    "Sequence": s.sequence,
                    "Length": s.length,
                    "Mono Mass (Da)": round(s.mono_mass, 3),
                    "pI": round(s.isoelectric_point, 2),
                    "Charge @ pH 7": round(s.charge_at_ph7, 2),
                    "GRAVY": round(s.gravy, 3),
                    "Instability": round(s.instability_index, 1),
                    "Aromaticity": round(s.aromaticity, 3),
                    "Aliphatic Index": round(s.aliphatic_index, 1),
                    "Detectability": s.detection_likelihood(),
                })
            df4 = pd.DataFrame(score_rows)
            st.markdown(df4.to_html(index=False), unsafe_allow_html=True)

            st.markdown("---")

            # Two side-by-side comparison plots
            col_p1, col_p2 = st.columns(2, gap="large")

            with col_p1:
                st.markdown("**GRAVY vs pI**")
                fig4a, ax4a = plt.subplots(figsize=(5, 3.5))
                det_styles = {"HIGH": ("black", "o", 60), "MEDIUM": ("#555", "s", 40), "LOW": ("#aaa", "^", 40)}
                for tier, (col, marker, ms) in det_styles.items():
                    grp = groups[tier]
                    if grp:
                        ax4a.scatter(
                            [g.gravy for g in grp],
                            [g.isoelectric_point for g in grp],
                            c=col, marker=marker, s=ms, label=tier, zorder=3,
                        )
                        for g in grp:
                            ax4a.annotate(
                                g.sequence[:6],
                                (g.gravy, g.isoelectric_point),
                                fontsize=6.5, ha="left", va="bottom",
                                xytext=(3, 3), textcoords="offset points",
                            )
                ax4a.axvline(0, color="black", linewidth=0.7, linestyle="--")
                ax4a.set_xlabel("GRAVY (hydrophobicity)")
                ax4a.set_ylabel("Isoelectric point (pI)")
                ax4a.set_title("GRAVY vs pI")
                ax4a.legend(frameon=True, fontsize=8)
                st.pyplot(fig4a, use_container_width=True)
                plt.close(fig4a)

            with col_p2:
                st.markdown("**Instability index**")
                fig4b, ax4b = plt.subplots(figsize=(5, 3.5))
                seq_labels = [s["Sequence"][:8] for s in score_rows]
                instab_vals = [s["Instability"] for s in score_rows]
                bars = ax4b.barh(seq_labels, instab_vals, color="black", edgecolor="black")
                ax4b.axvline(40, color="#888", linestyle="--", linewidth=1,
                             label="Instability threshold (40)")
                ax4b.set_xlabel("Instability index")
                ax4b.set_title("Instability index by peptide")
                ax4b.legend(frameon=True, fontsize=8)
                ax4b.invert_yaxis()
                st.pyplot(fig4b, use_container_width=True)
                plt.close(fig4b)

            st.markdown("---")

            # Per-sequence detail expander
            st.markdown("**Per-peptide detail**")
            for s in scores:
                with st.expander(f"{s.sequence}  |  {s.detection_likelihood()}"):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Length", s.length)
                    c2.metric("Mono Mass", f"{s.mono_mass:.3f} Da")
                    c3.metric("pI", f"{s.isoelectric_point:.2f}")
                    c4.metric("Charge @ pH 7", f"{s.charge_at_ph7:.2f}")
                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("GRAVY", f"{s.gravy:.3f}")
                    c6.metric("Instability", f"{s.instability_index:.1f}")
                    c7.metric("Aromaticity", f"{s.aromaticity:.3f}")
                    c8.metric("Aliphatic Index", f"{s.aliphatic_index:.1f}")
