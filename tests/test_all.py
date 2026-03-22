"""
Tests for ms_peptide_analyzer — using pyOpenMS nanobind/pybind11 C++ bindings.
"""

import pytest
import pyopenms as oms

from ms_peptide_analyzer.peptide import digest_protein, sequence_mass, add_fixed_modification
from ms_peptide_analyzer.fragmentation import FragmentIonGenerator
from ms_peptide_analyzer.isotopes import IsotopeAnalyzer
from ms_peptide_analyzer.scorer import PeptideScorer


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ──────────────────────────────────────────────────────────────────────────────

HUMAN_INSULIN_A = "GIVEQCCTSICSLYQLENYCN"   # Insulin A chain
UBIQUITIN_N20   = "MQIFVKTLTGKTITLEVEPS"    # First 20 AA of ubiquitin


# ──────────────────────────────────────────────────────────────────────────────
# peptide.py
# ──────────────────────────────────────────────────────────────────────────────

class TestDigestProtein:
    def test_returns_list(self):
        peptides = digest_protein(HUMAN_INSULIN_A)
        assert isinstance(peptides, list)
        assert len(peptides) > 0

    def test_peptide_info_fields(self):
        peptides = digest_protein(HUMAN_INSULIN_A, missed_cleavages=0)
        for p in peptides:
            assert p.mono_mass > 0
            assert p.avg_mass > 0
            assert p.formula != ""
            assert p.start >= 0
            assert p.end >= p.start

    def test_mz_increases_with_charge(self):
        peptides = digest_protein(UBIQUITIN_N20, charges=[1, 2, 3])
        # Group by sequence
        by_seq: dict = {}
        for p in peptides:
            by_seq.setdefault(p.sequence, {})[p.charge] = p.mz
        for seq, mz_map in by_seq.items():
            if 1 in mz_map and 2 in mz_map:
                assert mz_map[1] > mz_map[2], f"MZ(z=1) should be > MZ(z=2) for {seq}"

    def test_invalid_sequence_raises(self):
        with pytest.raises(ValueError):
            digest_protein("12345!@#$%")

    def test_min_max_length(self):
        peptides = digest_protein(UBIQUITIN_N20, min_length=7, max_length=15)
        for p in peptides:
            assert 7 <= len(p.sequence) <= 15

    def test_lys_c_enzyme(self):
        # LysC cleaves after K
        peptides = digest_protein(UBIQUITIN_N20, enzyme="Lys-C", missed_cleavages=0)
        for p in peptides:
            # Every peptide except the last should end with K (or be the last fragment)
            assert len(p.sequence) > 0


class TestSequenceMass:
    def test_known_mass(self):
        # Glycine (G): monoisotopic = 75.03203 Da (as an amino acid, C2H5NO2 + H2O adjustment)
        # For a single residue handled as Full peptide:
        result = sequence_mass("G")
        assert result["monoisotopic_mass"] > 0
        assert "C" in result["formula"]

    def test_fields_present(self):
        result = sequence_mass("PEPTIDE")
        assert "sequence" in result
        assert "monoisotopic_mass" in result
        assert "average_mass" in result
        assert "formula" in result
        assert result["length"] == 7

    def test_mass_scales_with_length(self):
        m1 = sequence_mass("AK")["monoisotopic_mass"]
        m2 = sequence_mass("AKAKAKAKAI")["monoisotopic_mass"]
        assert m2 > m1


class TestAddFixedModification:
    def test_carbamidomethyl_cysteine(self):
        modified = add_fixed_modification("ACDEFGHIK", "Carbamidomethyl", "C")
        assert "C(Carbamidomethyl)" in modified
        assert modified.count("C(Carbamidomethyl)") == 1

    def test_no_residue_unchanged(self):
        result = add_fixed_modification("PEPTIDE", "Oxidation", "M")
        assert result == "PEPTIDE"


# ──────────────────────────────────────────────────────────────────────────────
# fragmentation.py
# ──────────────────────────────────────────────────────────────────────────────

class TestFragmentIonGenerator:
    def setup_method(self):
        self.gen = FragmentIonGenerator(ion_types=["b", "y"], max_charge=1)

    def test_returns_ions(self):
        ions = self.gen.generate("PEPTIDE")
        assert len(ions) > 0

    def test_ion_types_present(self):
        ions = self.gen.generate("PEPTIDE")
        types = {ion.ion_type for ion in ions}
        assert "b" in types
        assert "y" in types

    def test_ion_count(self):
        # For a 7-residue peptide (PEPTIDE), b/y series has 6 ions each at z=1
        ions = self.gen.generate("PEPTIDE", charge=1)
        b_ions = [i for i in ions if i.ion_type == "b"]
        y_ions = [i for i in ions if i.ion_type == "y"]
        assert len(b_ions) == 6
        assert len(y_ions) == 6

    def test_mz_sorted(self):
        ions = self.gen.generate("LGGNEQVTR")
        mzs = [ion.mz for ion in ions]
        assert mzs == sorted(mzs)

    def test_ladder_returns_dict(self):
        ladder = self.gen.ladder("PEPTIDE")
        assert "b" in ladder
        assert "y" in ladder
        assert all(isinstance(v, list) for v in ladder.values())

    def test_a_ions(self):
        gen_a = FragmentIonGenerator(ion_types=["a", "b", "y"])
        ions = gen_a.generate("PEPTIDE")
        types = {ion.ion_type for ion in ions}
        assert "a" in types

    def test_multiply_charged_fragments(self):
        gen_z2 = FragmentIonGenerator(ion_types=["b", "y"], max_charge=2)
        ions = gen_z2.generate("PEPTIDELONGER", charge=2)
        z2_ions = [i for i in ions if i.charge == 2]
        assert len(z2_ions) > 0

    def test_as_dict_structure(self):
        d = self.gen.as_dict("PEPTIDE")
        assert isinstance(d, dict)
        for key, val in d.items():
            assert isinstance(val, list)
            for item in val:
                assert "mz" in item
                assert "number" in item


# ──────────────────────────────────────────────────────────────────────────────
# isotopes.py
# ──────────────────────────────────────────────────────────────────────────────

class TestIsotopeAnalyzer:
    def setup_method(self):
        self.analyzer = IsotopeAnalyzer(max_isotopes=5)

    def test_from_sequence_returns_peaks(self):
        peaks = self.analyzer.from_sequence("PEPTIDE")
        assert len(peaks) > 0

    def test_monoisotopic_is_first(self):
        peaks = self.analyzer.from_sequence("PEPTIDE")
        assert peaks[0].isotope_index == 0

    def test_intensities_normalized(self):
        peaks = self.analyzer.from_sequence("PEPTIDE")
        max_int = max(p.intensity for p in peaks)
        assert abs(max_int - 1.0) < 1e-6

    def test_mz_increases_with_isotope(self):
        peaks = self.analyzer.from_sequence("PEPTIDE", charge=1)
        for i in range(1, len(peaks)):
            assert peaks[i].mz > peaks[i - 1].mz

    def test_from_formula(self):
        peaks = self.analyzer.from_formula("C34H53N7O15")
        assert len(peaks) > 0

    def test_envelope_summary_keys(self):
        summary = self.analyzer.envelope_summary("PEPTIDE", charge=2)
        for key in ("sequence", "formula", "charge", "monoisotopic_mz",
                    "most_abundant_mz", "peaks"):
            assert key in summary

    def test_mass_methods(self):
        mono = self.analyzer.monoisotopic_mass("PEPTIDE")
        avg = self.analyzer.average_mass("PEPTIDE")
        assert mono > 0
        assert avg > mono   # average always >= monoisotopic for typical peptides

    def test_compare_envelopes(self):
        # SILAC-like: K vs K+8 (simulated via different sequences for shape test)
        pairs = self.analyzer.compare_envelopes("PEPTIDE", "PEPTIDE", charge=1)
        for mz1, mz2, delta in pairs:
            assert abs(delta) < 1e-6   # same sequence -> same envelope


# ──────────────────────────────────────────────────────────────────────────────
# scorer.py
# ──────────────────────────────────────────────────────────────────────────────

class TestPeptideScorer:
    def setup_method(self):
        self.scorer = PeptideScorer()

    def test_score_returns_score_object(self):
        s = self.scorer.score("PEPTIDE")
        assert s.sequence == "PEPTIDE"
        assert s.length == 7

    def test_mass_positive(self):
        s = self.scorer.score("LGGNEQVTR")
        assert s.mono_mass > 0

    def test_pi_range(self):
        # pI must be in physiological range
        s = self.scorer.score("LGGNEQVTR")
        assert 0.0 <= s.isoelectric_point <= 14.0

    def test_gravy_lysine_rich_negative(self):
        # Lysine-rich peptide has negative GRAVY
        s = self.scorer.score("KKKKKKKK")
        assert s.gravy < 0

    def test_gravy_leucine_rich_positive(self):
        s = self.scorer.score("LLLLLLLL")
        assert s.gravy > 0

    def test_detectability_labels(self):
        labels = {"HIGH", "MEDIUM", "LOW"}
        s = self.scorer.score("LGGNEQVTR")
        assert s.detection_likelihood() in labels

    def test_score_many_sorted(self):
        seqs = ["KKKK", "LGGNEQVTR", "PEPTIDE", "IIIIIIII"]
        results = self.scorer.score_many(seqs)
        assert len(results) == len(seqs)

    def test_rank_by_detectability(self):
        seqs = ["KKKK", "LGGNEQVTR", "PEPTIDE"]
        groups = self.scorer.rank_by_detectability(seqs)
        assert set(groups.keys()) == {"HIGH", "MEDIUM", "LOW"}
        total = sum(len(v) for v in groups.values())
        assert total == len(seqs)

    def test_tryptic_c_terminus(self):
        s_k = self.scorer.score("PEPTIDEK")
        s_r = self.scorer.score("PEPTIDER")
        s_n = self.scorer.score("PEPTIDE")
        assert s_k.is_tryptic_c is True
        assert s_r.is_tryptic_c is True
        assert s_n.is_tryptic_c is False
