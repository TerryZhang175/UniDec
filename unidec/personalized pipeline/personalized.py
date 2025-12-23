from pathlib import Path
import csv
import re
import sys
from typing import Optional

path_root = Path(__file__).parents[2]
sys.path.insert(0, str(path_root))

from unidec.UniDecImporter import ImporterFactory
import pyteomics.mass as ms
from unidec.modules.isotopetools import isojim
try:
    from unidec.modules.unidecstructure import IsoDecConfig
    from unidec.IsoDec.match import (
        calculate_cosinesimilarity as isodec_calculate_cosinesimilarity,
        find_matches as isodec_find_matches,
        find_matched_intensities as isodec_find_matched_intensities,
        make_shifted_peak as isodec_make_shifted_peak,
    )
except Exception:
    IsoDecConfig = None
    isodec_calculate_cosinesimilarity = None
    isodec_find_matches = None
    isodec_find_matched_intensities = None
    isodec_make_shifted_peak = None
import matplotlib.pyplot as plt
import numpy as np

filepath = '/Users/terry/Downloads/Mannually checked ECD_29_July_ZD/S20G_04_10-11ECD_Rep2/RE3.txt'

# What to generate/plot:
# - "precursor": intact species isotope envelope (previous behavior)
# - "fragments": peptide backbone fragments (b/y/c/z for ECD-style MS/MS)
PLOT_MODE = "diagnose"
SCAN = 1

# Optional: focus on an m/z region of interest.
# Set to None to use the full scan.
# Examples:
#   MZ_MIN, MZ_MAX = 900, 1100
#   MZ_MIN, MZ_MAX = 300, 2000
MZ_MIN = None
MZ_MAX = None

# Peptide sequence. Supports chemical-formula mods in brackets, e.g.:
# - Oxidation: "M[O]"
# - Phospho: "S[HPO3]" (equivalent to PO3H)
# - Carbamidomethyl (IAA): "C[C2H3NO]"
# Bracket contents are interpreted as an elemental formula (not a mass delta).
PEPTIDE = "KCNTATCATQRLANFLVHSGNNFGAILSSTNVGSNTY"
CHARGE = 6  # integer charge state, e.g. 10
COPIES = 2  # 1=monomer, 2=dimer (two copies of the same peptide)
AMIDATED = True  # C-terminal amidation (adds HN, removes O; delta = H1N1O-1) per copy
DISULFIDE_BONDS = 2  # total disulfide bonds in the complex (each removes H2, ~-2.01565 Da per bond)
# Example (amidated disulfide-linked dimer): COPIES=2, AMIDATED=True, DISULFIDE_BONDS=2

ION_TYPES = ("b", "y", "c", "z")  # For ECD you may want ("b","y","c","z-dot") depending on your annotation
FRAG_MIN_CHARGE = 1
FRAG_MAX_CHARGE = 3
MATCH_TOL_PPM = 20
MIN_OBS_REL_INT = 0.02
MAX_PLOTTED_FRAGMENTS = 40
LABEL_MATCHES = False

# Hydrogen-transfer handling (ExD/ECD-style). Uses H+ mass (not H atom mass).
# Only enabled for c/z by default; accept transfer only if cosine similarity
# improves by >= 20% vs the neutral (no-transfer) model.
ENABLE_H_TRANSFER = True
H_TRANSFER_MASS = 1.007276467
H_TRANSFER_ION_TYPES_1H = ("c", "z")
H_TRANSFER_ION_TYPES_2H = ("c", "z")
H_TRANSFER_MIN_REL_IMPROVEMENT = 0.20  # 0.20 = +20% vs neutral cosine score

# Neutral losses (configurable). Single-loss candidates only (no mixed-loss combinations).
ENABLE_NEUTRAL_LOSSES = True
NEUTRAL_LOSS_ION_SERIES = ("b", "y", "c", "z")
NEUTRAL_LOSS_MAX_H2O = 2  # 0, 1, 2 allowed
NEUTRAL_LOSS_MAX_NH3 = 2  # 0, 1, 2 allowed
NEUTRAL_LOSS_MAX_CO = 1   # 0, 1 allowed
NEUTRAL_LOSS_MAX_CO2 = 1  # 0, 1 allowed

# Fragment-driven intensity cap stripping:
# 1) Generate theoretical fragment ions (incl. neutral losses + H-transfer shifts).
# 2) For each candidate, take the highest observed peak within MATCH_TOL_PPM of its anchor m/z.
# 3) Let cap = max of those observed intensities.
# 4) Remove all peaks with intensity > cap (reduces dynamic-range domination by precursor/charge-reduced peaks).
ENABLE_FRAGMENT_INTENSITY_CAP = True
FRAGMENT_INTENSITY_CAP_MZ_MIN = 300.0
FRAGMENT_INTENSITY_CAP_MZ_MAX = 1000.0
FRAGMENT_INTENSITY_CAP_TOL_PPM = None  # if None, uses MATCH_TOL_PPM
FRAGMENT_INTENSITY_CAP_MIN_HITS = 25  # require at least this many non-zero windows to activate
FRAGMENT_INTENSITY_CAP_VERBOSE = False

# Diagnostics: set these to inspect why an expected ion was not selected.
# Examples:
#   DIAGNOSE_ION_SPEC = "c7^2+"
#   DIAGNOSE_ION_SPEC = "z12-2H2O^3+"
#   DIAGNOSE_ION_SPEC = "z-dot12-CO"  # will scan charge range if no ^z+ suffix is present
DIAGNOSE_ION_SPEC = "b19-2H2O"
# Hydrogen transfer degree (H+). Use an integer in {-2,-1,0,1,2}.
DIAGNOSE_H_TRANSFER = 0
# If ion spec has no charge, scan FRAG_MIN_CHARGE..FRAG_MAX_CHARGE and report all.
DIAGNOSE_SCAN_CHARGES = True
DIAGNOSE_SHOW_PLOT = True
DIAGNOSE_MAX_TABLE_ROWS = 12
DIAGNOSE_EXPORT_CSV = True
# If None, writes to `unidec/personalized pipeline/diagnose_outputs/` with an auto filename.
DIAGNOSE_CSV_SUMMARY_PATH = None
DIAGNOSE_CSV_PEAKS_PATH = None

# CSV export for normal matching modes (e.g., PLOT_MODE="fragments").
EXPORT_FRAGMENTS_CSV = True
# If None, writes to `unidec/personalized pipeline/match_outputs/` with an auto filename.
FRAGMENTS_CSV_SUMMARY_PATH = None
FRAGMENTS_CSV_PEAKS_PATH = None

# IsoDec-style false-positive suppression rules (preferred over ad-hoc gates).
ENABLE_ISODEC_RULES = True
ISODEC_MINPEAKS = 3
ISODEC_CSS_THRESH = 0.70
ISODEC_MIN_AREA_COVERED = 0.20
ISODEC_MZ_WINDOW_LB = -1.05
ISODEC_MZ_WINDOW_UB = 4.05
ISODEC_MINUSONE_AS_ZERO = True
ISODEC_VERBOSE = False

ISOLEN = 128
ADDUCT_MASS = 1.007276467  # proton mass for positive-mode m/z conversion
MASS_DIFF_C = 1.0033  # ~C13-C12 mass difference (Da)
AMIDATION_FORMULA = "H1N1O-1"

ALIGN_TO_DATA = True
ALIGN_WINDOW_MZ = 1.0
REL_INTENSITY_CUTOFF = 0.01

if ENABLE_ISODEC_RULES and IsoDecConfig is None:
    raise ImportError(
        "ENABLE_ISODEC_RULES=True but IsoDec modules could not be imported. "
        "Install UniDec's Python deps (e.g., pandas/numba) or set ENABLE_ISODEC_RULES=False."
    )

ISODEC_CONFIG = IsoDecConfig() if IsoDecConfig is not None else None
if ISODEC_CONFIG is not None:
    ISODEC_CONFIG.verbose = 1 if ISODEC_VERBOSE else 0
    ISODEC_CONFIG.matchtol = float(MATCH_TOL_PPM)
    ISODEC_CONFIG.minpeaks = int(ISODEC_MINPEAKS)
    ISODEC_CONFIG.css_thresh = float(ISODEC_CSS_THRESH)
    ISODEC_CONFIG.minareacovered = float(ISODEC_MIN_AREA_COVERED)
    ISODEC_CONFIG.mzwindowlb = float(ISODEC_MZ_WINDOW_LB)
    ISODEC_CONFIG.mzwindowub = float(ISODEC_MZ_WINDOW_UB)
    ISODEC_CONFIG.minusoneaszero = 1 if ISODEC_MINUSONE_AS_ZERO else 0
    ISODEC_CONFIG.isotopethreshold = float(REL_INTENSITY_CUTOFF)

importer = ImporterFactory.create_importer(filepath)
if hasattr(importer, "grab_centroid_data"):
    spectrum = importer.grab_centroid_data(SCAN)
else:
    spectrum = importer.get_single_scan(SCAN)

if not PEPTIDE:
    raise ValueError('Set PEPTIDE (e.g. "ACDEFGHIK" or "CKLH[PO4]CKLAH")')
if CHARGE is None or int(CHARGE) == 0:
    raise ValueError("Set CHARGE to a non-zero integer (e.g. 10)")
CHARGE = int(CHARGE)

def _parse_custom_sequence(peptide: str) -> list[tuple[str, list[str]]]:
    """
    Parse a peptide string like 'AC[O]DEK' into per-residue (aa, [formula_mods]).

    Notes:
    - Bracket contents are treated as elemental formulas understood by pyteomics (e.g. "O", "HPO3", "C2H3NO").
    - This parser intentionally does NOT support raw mass deltas (e.g. [+15.99]) because isotope modeling needs formulas.
    """
    peptide = peptide.strip()
    residues: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(peptide):
        aa = peptide[i]
        if aa.isspace():
            i += 1
            continue
        if not ("A" <= aa <= "Z"):
            raise ValueError(
                f"Unsupported character '{aa}' in PEPTIDE. "
                "Use an uppercase amino-acid sequence with optional '[FORMULA]' mods."
            )
        i += 1
        mods: list[str] = []
        while i < len(peptide) and peptide[i] == "[":
            j = peptide.find("]", i + 1)
            if j == -1:
                raise ValueError('Unclosed "[" in PEPTIDE.')
            mod = peptide[i + 1:j].strip()
            if not mod:
                raise ValueError("Empty [] modification in PEPTIDE.")
            mods.append(mod)
            i = j + 1
        residues.append((aa, mods))
    return residues


def _residue_range_composition(residues: list[tuple[str, list[str]]], start: int, end: int) -> ms.Composition:
    comp = ms.Composition()
    for aa, mods in residues[start:end]:
        try:
            comp += ms.std_aa_comp[aa]
        except KeyError as e:
            raise ValueError(f"Unsupported residue '{aa}' in PEPTIDE.") from e
        for mod in mods:
            comp += ms.Composition(mod)
    return comp


def _ion_composition_from_sequence(
    residues: list[tuple[str, list[str]]],
    ion_type: str,
    frag_len: int,
    amidated: bool,
) -> tuple[str, ms.Composition]:
    n = len(residues)
    if frag_len <= 0 or frag_len >= n:
        raise ValueError(f"Invalid fragment length {frag_len} for peptide length {n}.")
    if ion_type not in ms.std_ion_comp:
        raise ValueError(f"Unsupported ion_type '{ion_type}'. Try one of: {sorted(ms.std_ion_comp.keys())}")

    # pyteomics fragment ion comps are defined relative to the corresponding neutral peptide composition (residues + H2O).
    if ion_type.startswith(("a", "b", "c")):
        frag_res = _residue_range_composition(residues, 0, frag_len)
        name = f"{ion_type}{frag_len}"
        has_c_term = False
    else:
        frag_res = _residue_range_composition(residues, n - frag_len, n)
        name = f"{ion_type}{frag_len}"
        has_c_term = True

    pep_comp = frag_res + ms.Composition("H2O")
    if amidated and has_c_term:
        pep_comp += ms.Composition(AMIDATION_FORMULA)

    ion_comp = pep_comp + ms.std_ion_comp[ion_type]
    return name, ion_comp


def _theoretical_isodist_from_comp(
    comp: ms.Composition,
    charge: int,
    proton_count: Optional[int] = None,
) -> np.ndarray:
    mono_mass = float(comp.mass())
    isolist = np.array(
        [
            comp.get("C", 0),
            comp.get("H", 0),
            comp.get("N", 0),
            comp.get("O", 0),
            comp.get("S", 0),
            comp.get("Fe", 0),
            comp.get("K", 0),
            comp.get("Ca", 0),
            comp.get("Ni", 0),
            comp.get("Zn", 0),
            comp.get("Mg", 0),
        ],
        dtype=int,
    )
    if np.any(isolist < 0):
        raise ValueError(f"Computed a negative elemental composition: {isolist}")

    intensities = np.asarray(isojim(isolist, length=ISOLEN), dtype=float)
    isotope_index = np.arange(len(intensities), dtype=float)
    masses = mono_mass + isotope_index * MASS_DIFF_C
    if int(charge) == 0:
        raise ValueError("charge must be non-zero")
    pcount = int(charge) if proton_count is None else int(proton_count)
    mz = (masses + (pcount * ADDUCT_MASS)) / abs(int(charge))
    dist = np.column_stack([mz, intensities])

    max_int = float(np.max(dist[:, 1]))
    dist = dist[dist[:, 1] >= max_int * REL_INTENSITY_CUTOFF].copy()
    return dist


residues = _parse_custom_sequence(PEPTIDE)

spectrum = np.asarray(spectrum, dtype=float)
if spectrum.ndim != 2 or spectrum.shape[1] != 2:
    raise ValueError(f"Expected spectrum shape (N, 2), got {spectrum.shape}")
spectrum = spectrum[np.isfinite(spectrum[:, 0]) & np.isfinite(spectrum[:, 1])]
spectrum = spectrum[spectrum[:, 1] > 0]
spectrum = spectrum[np.argsort(spectrum[:, 0])]

if MZ_MIN is not None or MZ_MAX is not None:
    mz_min = -np.inf if MZ_MIN is None else float(MZ_MIN)
    mz_max = np.inf if MZ_MAX is None else float(MZ_MAX)
    if mz_min >= mz_max:
        raise ValueError(f"Invalid m/z window: MZ_MIN={MZ_MIN}, MZ_MAX={MZ_MAX}")
    spectrum = spectrum[(spectrum[:, 0] >= mz_min) & (spectrum[:, 0] <= mz_max)]
    if len(spectrum) == 0:
        raise ValueError(f"No peaks remain after applying m/z window: [{mz_min}, {mz_max}]")

try:
    # Match the look/interaction of IsoDec's `plot_pks` (vlines + scroll zoom + theory below axis).
    from unidec.IsoDec.plots import cplot, on_scroll
except Exception:
    cplot = None
    on_scroll = None

def _nearest_peak_index(sorted_mzs: np.ndarray, target_mz: float) -> int:
    idx = int(np.searchsorted(sorted_mzs, target_mz))
    if idx <= 0:
        return 0
    if idx >= len(sorted_mzs):
        return len(sorted_mzs) - 1
    left = idx - 1
    right = idx
    if abs(sorted_mzs[left] - target_mz) <= abs(sorted_mzs[right] - target_mz):
        return left
    return right


def _within_ppm(mz_obs: float, mz_theory: float, tol_ppm: float) -> bool:
    if mz_theory == 0:
        return False
    return abs(mz_obs - mz_theory) / abs(mz_theory) * 1e6 <= tol_ppm


def _ion_series(ion_type: str) -> str:
    """
    Return the base ion series letter for a pyteomics ion type string.
    Examples: 'c' -> 'c', 'c-dot' -> 'c', 'z-H2O' -> 'z'.
    """
    if not ion_type:
        return ""
    return ion_type.split("-", 1)[0][:1]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the probability simplex: w>=0 and sum(w)=1.
    Reference: Duchi et al. (2008).
    """
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1))[0]
    if rho.size == 0:
        return np.full_like(v, 1.0 / v.size)
    rho = int(rho[-1])
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = v - theta
    w[w < 0] = 0.0
    s = float(np.sum(w))
    if s == 0.0:
        return np.full_like(v, 1.0 / v.size)
    return w / s


def _build_sample_axis(
    dists: list[np.ndarray],
    decimals: int = 6,
    mz_min=None,
    mz_max=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    keys_all = []
    scale = float(10 ** int(decimals))
    for dist in dists:
        if dist is None or len(dist) == 0:
            continue
        mz = dist[:, 0]
        if mz_min is not None:
            mz = mz[mz >= float(mz_min)]
        if mz_max is not None:
            mz = mz[mz <= float(mz_max)]
        if mz.size == 0:
            continue
        keys = np.rint(mz * scale).astype(np.int64)
        keys_all.append(keys)

    if len(keys_all) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), scale

    keys_union = np.unique(np.concatenate(keys_all))
    mzs_union = keys_union.astype(float) / scale
    return keys_union, mzs_union, scale


def _neutral_loss_label(n: int, formula: str) -> str:
    if n <= 0:
        return ""
    if n == 1:
        return f"-{formula}"
    return f"-{n}{formula}"


def _neutral_loss_variants(comp: ms.Composition, ion_series: str) -> list[tuple[str, ms.Composition]]:
    """
    Generate neutral-loss variants of a fragment composition.
    Returns list of (suffix, composition), including the neutral (no-loss) variant.
    """
    variants: list[tuple[str, ms.Composition]] = [("", comp)]
    if not ENABLE_NEUTRAL_LOSSES:
        return variants
    if ion_series not in set(NEUTRAL_LOSS_ION_SERIES):
        return variants

    max_h2o = int(max(0, NEUTRAL_LOSS_MAX_H2O))
    max_nh3 = int(max(0, NEUTRAL_LOSS_MAX_NH3))
    max_co = int(max(0, NEUTRAL_LOSS_MAX_CO))
    max_co2 = int(max(0, NEUTRAL_LOSS_MAX_CO2))
    if max_h2o == 0 and max_nh3 == 0 and max_co == 0 and max_co2 == 0:
        return variants

    h2o = ms.Composition("H2O")
    nh3 = ms.Composition("NH3")
    co = ms.Composition("CO")
    co2 = ms.Composition("CO2")

    for n_h2o in range(1, max_h2o + 1):
        suffix = _neutral_loss_label(n_h2o, "H2O")
        try:
            new_comp = comp - (h2o * n_h2o)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    for n_nh3 in range(1, max_nh3 + 1):
        suffix = _neutral_loss_label(n_nh3, "NH3")
        try:
            new_comp = comp - (nh3 * n_nh3)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    for n_co in range(1, max_co + 1):
        suffix = _neutral_loss_label(n_co, "CO")
        try:
            new_comp = comp - (co * n_co)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    for n_co2 in range(1, max_co2 + 1):
        suffix = _neutral_loss_label(n_co2, "CO2")
        try:
            new_comp = comp - (co2 * n_co2)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    return variants


def _neutral_loss_columns(loss_suffix: str) -> dict[str, int]:
    cols = {"H2O": 0, "NH3": 0, "CO": 0, "CO2": 0, "2H2O": 0, "2NH3": 0}
    s = str(loss_suffix or "").strip()
    if not s:
        return cols
    if s == "-H2O":
        cols["H2O"] = 1
    elif s == "-NH3":
        cols["NH3"] = 1
    elif s == "-CO":
        cols["CO"] = 1
    elif s == "-CO2":
        cols["CO2"] = 1
    elif s == "-2H2O":
        cols["2H2O"] = 1
    elif s == "-2NH3":
        cols["2NH3"] = 1
    return cols


def _sanitize_filename(text: str) -> str:
    s = str(text).strip()
    if not s:
        return "output"
    # Keep a conservative set of filename-safe characters.
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    # Avoid pathological runs of underscores.
    return re.sub(r"_+", "_", "".join(out)).strip("_") or "output"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _parse_fragment_spec(spec: str) -> tuple[str, int, str, int, Optional[int]]:
    """
    Parse an ion spec like:
      - "c7"
      - "z12-2H2O^3+"
      - "z-dot12-CO"

    Returns: (ion_type, frag_len, loss_formula, loss_count, charge)
    loss_formula is one of {"", "H2O", "NH3", "CO", "CO2"}.
    """
    s = str(spec).strip().replace(" ", "")
    if not s:
        raise ValueError("Empty DIAGNOSE_ION_SPEC.")

    charge = None
    m = re.search(r"(?:\^)?(\d+)\+$", s)
    if m:
        charge = int(m.group(1))
        s = s[: m.start()]

    m = re.match(r"(.+?)(\d+)(.*)$", s)
    if not m:
        raise ValueError(
            f"Could not parse ion spec '{spec}'. Expected something like 'c7', 'z12-2H2O^3+'."
        )
    ion_type = m.group(1)
    frag_len = int(m.group(2))
    tail = m.group(3) or ""

    loss_formula = ""
    loss_count = 0
    if tail:
        # Accept common notations like "-2(H2O)" or "-2×H2O" by normalizing to "-2H2O".
        tail = tail.replace("(", "").replace(")", "").replace("×", "").replace("x", "")
        m2 = re.fullmatch(r"-(\d+)?(H2O|NH3|CO|CO2)", tail)
        if not m2:
            raise ValueError(
                f"Unsupported loss suffix '{tail}' in '{spec}'. "
                "Use '', '-H2O', '-2H2O', '-NH3', '-2NH3', '-CO', or '-CO2'."
            )
        loss_count = int(m2.group(1) or 1)
        loss_formula = str(m2.group(2))

    return ion_type, frag_len, loss_formula, loss_count, charge


def _apply_neutral_loss(comp: ms.Composition, formula: str, count: int) -> ms.Composition:
    if not formula or int(count) <= 0:
        return comp
    if formula not in {"H2O", "NH3", "CO", "CO2"}:
        raise ValueError(f"Unsupported neutral loss formula '{formula}'.")
    loss_comp = ms.Composition(formula)
    return comp - (loss_comp * int(count))

def _max_intensity_in_ppm_window(
    spectrum_mz: np.ndarray, spectrum_int: np.ndarray, target_mz: float, tol_ppm: float
) -> float:
    if target_mz <= 0 or tol_ppm <= 0 or spectrum_mz.size == 0:
        return 0.0
    tol = float(tol_ppm) * 1e-6
    lo = float(target_mz) * (1.0 - tol)
    hi = float(target_mz) * (1.0 + tol)
    i0 = int(np.searchsorted(spectrum_mz, lo, side="left"))
    i1 = int(np.searchsorted(spectrum_mz, hi, side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.max(spectrum_int[i0:i1]))


def _compute_fragment_intensity_cap(
    residues: list[tuple[str, list[str]]],
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    tol_ppm: float,
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
) -> tuple[float, int]:
    cap = 0.0
    hits = 0
    n = len(residues)
    mz_min_f = None if mz_min is None else float(mz_min)
    mz_max_f = None if mz_max is None else float(mz_max)

    for ion_type in ION_TYPES:
        series = _ion_series(ion_type)
        allow_1h = bool(ENABLE_H_TRANSFER) and (series in set(H_TRANSFER_ION_TYPES_1H))
        allow_2h = bool(ENABLE_H_TRANSFER) and (series in set(H_TRANSFER_ION_TYPES_2H))

        for frag_len in range(1, n):
            try:
                _, frag_comp = _ion_composition_from_sequence(residues, ion_type, frag_len, amidated=AMIDATED)
            except Exception:
                continue

            for _, loss_comp in _neutral_loss_variants(frag_comp, ion_series=series):
                for z in range(int(FRAG_MIN_CHARGE), int(FRAG_MAX_CHARGE) + 1):
                    try:
                        dist0 = _theoretical_isodist_from_comp(loss_comp, z)
                    except Exception:
                        continue
                    if dist0.size == 0:
                        continue

                    anchor = float(dist0[np.argmax(dist0[:, 1]), 0])
                    if allow_1h or allow_2h:
                        shift_1 = float(H_TRANSFER_MASS) / float(z) if (allow_1h or allow_2h) else 0.0
                        shift_2 = 2.0 * float(H_TRANSFER_MASS) / float(z) if allow_2h else 0.0
                        shifts = [0.0]
                        if allow_1h:
                            shifts.extend([shift_1, -shift_1])
                        if allow_2h:
                            shifts.extend([shift_2, -shift_2])
                    else:
                        shifts = [0.0]

                    for s in shifts:
                        mz0 = anchor + float(s)
                        if mz_min_f is not None and mz0 < mz_min_f:
                            continue
                        if mz_max_f is not None and mz0 > mz_max_f:
                            continue
                        m = _max_intensity_in_ppm_window(spectrum_mz, spectrum_int, mz0, tol_ppm=float(tol_ppm))
                        if m > 0.0:
                            hits += 1
                            if m > cap:
                                cap = m

    return float(cap), int(hits)


def _strip_peaks_above_intensity_cap(spectrum: np.ndarray, cap: float) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or len(spectrum) == 0:
        return spectrum
    cap_f = float(cap)
    if not np.isfinite(cap_f) or cap_f <= 0:
        return spectrum
    out = spectrum[spectrum[:, 1] <= cap_f].copy()
    if len(out) > 1:
        out = out[np.argsort(out[:, 0])]
    return out


def _match_theory_peaks(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    theory_mz: np.ndarray,
    tol_ppm: float,
    theory_int: Optional[np.ndarray] = None,
) -> list[dict]:
    out: list[dict] = []
    theory_mz = np.asarray(theory_mz, dtype=float)
    theory_int = np.asarray(theory_int, dtype=float) if theory_int is not None else None
    for i, mz in enumerate(theory_mz):
        idx = _nearest_peak_index(spectrum_mz, float(mz))
        mz_obs = float(spectrum_mz[idx])
        ppm = (mz_obs - float(mz)) / float(mz) * 1e6 if mz != 0 else float("inf")
        within = _within_ppm(mz_obs, float(mz), tol_ppm)
        row = {
            "theory_mz": float(mz),
            "theory_int": float(theory_int[i]) if theory_int is not None and i < len(theory_int) else "",
            "obs_mz": mz_obs,
            "ppm": float(ppm),
            "obs_int": float(spectrum_int[idx]),
            "within": bool(within),
            "obs_idx": int(idx),
        }
        out.append(row)
    return out


def _diagnose_candidate(
    residues: list[tuple[str, list[str]]],
    ion_type: str,
    frag_len: int,
    z: int,
    loss_formula: str,
    loss_count: int,
    h_transfer: int,
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
) -> dict:
    result = {
        "ion_type": ion_type,
        "frag_len": int(frag_len),
        "z": int(z),
        "loss_formula": loss_formula,
        "loss_count": int(loss_count),
        "h_transfer": int(h_transfer),
        "ok": False,
        "reason": "",
    }

    frag_name, frag_comp = _ion_composition_from_sequence(residues, ion_type, frag_len, amidated=AMIDATED)
    result["frag_name"] = frag_name

    try:
        loss_comp = _apply_neutral_loss(frag_comp, loss_formula, loss_count)
    except Exception as e:
        result["reason"] = f"neutral_loss_failed: {e}"
        return result

    try:
        dist0 = _theoretical_isodist_from_comp(loss_comp, z)
    except Exception as e:
        result["reason"] = f"theory_failed: {e}"
        return result
    if dist0.size == 0:
        result["reason"] = "theory_empty"
        return result

    hz = float(H_TRANSFER_MASS) / float(z)
    dist = dist0.copy()
    dist[:, 0] += float(h_transfer) * hz

    # Raw (pre-anchor) cosine similarity on the theoretical m/z grid.
    y_obs = _sample_observed_intensities(spectrum_mz, spectrum_int, dist[:, 0], tol_ppm=float(MATCH_TOL_PPM))
    raw_cos = _cosine_similarity(y_obs, dist[:, 1])
    result["raw_cosine_preanchor"] = float(raw_cos)

    anchor_theory_mz = float(dist[np.argmax(dist[:, 1]), 0])
    obs_idx = _nearest_peak_index(spectrum_mz, anchor_theory_mz)
    obs_mz = float(spectrum_mz[obs_idx])
    anchor_ok = _within_ppm(obs_mz, anchor_theory_mz, float(MATCH_TOL_PPM))
    result["anchor_theory_mz"] = float(anchor_theory_mz)
    result["anchor_obs_mz"] = float(obs_mz)
    result["anchor_within_ppm"] = bool(anchor_ok)
    if not anchor_ok:
        result["reason"] = "anchor_outside_ppm"
        result["obs_idx"] = int(obs_idx)
        return result

    obs_int = float(spectrum_int[obs_idx])
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
    result["obs_idx"] = int(obs_idx)
    result["obs_int"] = float(obs_int)
    result["obs_rel_int"] = float(obs_int / obs_max) if obs_max > 0 else 0.0
    if float(MIN_OBS_REL_INT) > 0 and obs_int < obs_max * float(MIN_OBS_REL_INT):
        result["reason"] = "anchor_below_min_rel_int"
        return result

    ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
    result["anchor_ppm"] = float(ppm)

    dist_plot = dist.copy()
    dist_plot[:, 0] += obs_mz - anchor_theory_mz
    dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))

    if MZ_MIN is not None or MZ_MAX is not None:
        mz_min = -np.inf if MZ_MIN is None else float(MZ_MIN)
        mz_max = np.inf if MZ_MAX is None else float(MZ_MAX)
        dist_plot = dist_plot[(dist_plot[:, 0] >= mz_min) & (dist_plot[:, 0] <= mz_max)]
        if dist_plot.size == 0:
            result["reason"] = "outside_mz_window"
            return result

    max_plot = float(np.max(dist_plot[:, 1]))
    keep = dist_plot[:, 1] >= max_plot * float(REL_INTENSITY_CUTOFF)
    dist_plot = dist_plot[keep]
    if dist_plot.size == 0:
        result["reason"] = "below_rel_intensity_cutoff"
        return result

    result["dist_plot"] = dist_plot
    result["theory_matches"] = _match_theory_peaks(
        spectrum_mz, spectrum_int, dist_plot[:, 0], theory_int=dist_plot[:, 1], tol_ppm=float(MATCH_TOL_PPM)
    )

    # IsoDec acceptance + CSS (if enabled/available).
    if ENABLE_ISODEC_RULES and ISODEC_CONFIG is not None:
        local_centroids = _get_local_centroids_window(
            spectrum_mz, spectrum_int, obs_mz, ISODEC_CONFIG.mzwindowlb, ISODEC_CONFIG.mzwindowub
        )
        accepted, css = _isodec_css_and_accept(local_centroids, dist_plot, z=z, peakmz=obs_mz, config=ISODEC_CONFIG)
        result["isodec_css"] = float(css)
        result["isodec_accepted"] = bool(accepted)

        # Extra detail to explain rejection (using the same primitives as IsoDec).
        if isodec_find_matches is not None and local_centroids.size and dist_plot.size:
            matchedindexes, isomatches = isodec_find_matches(local_centroids, dist_plot, float(ISODEC_CONFIG.matchtol))
            mi = np.array(matchedindexes, dtype=int) if len(matchedindexes) else np.array([], dtype=int)
            ii = np.array(isomatches, dtype=int) if len(isomatches) else np.array([], dtype=int)
            matchedcentroids = local_centroids[mi] if mi.size else np.empty((0, 2), dtype=float)
            matchediso = dist_plot[ii] if ii.size else np.empty((0, 2), dtype=float)

            minpeaks_eff = int(ISODEC_CONFIG.minpeaks)
            if int(z) == 1 and len(matchedindexes) == 2 and len(isomatches) == 2:
                if isomatches[0] == 0 and isomatches[1] == 1:
                    int1 = float(local_centroids[matchedindexes[0], 1])
                    int2 = float(local_centroids[matchedindexes[1], 1])
                    ratio = (int2 / int1) if int1 != 0 else 0.0
                    if float(ISODEC_CONFIG.plusoneintwindowlb) < ratio < float(ISODEC_CONFIG.plusoneintwindowub):
                        minpeaks_eff = 2

            areacovered = (
                float(np.sum(matchediso[:, 1])) / float(np.sum(local_centroids[:, 1]))
                if local_centroids.size and np.sum(local_centroids[:, 1]) > 0
                else 0.0
            )
            topn = minpeaks_eff
            topthree = False
            if local_centroids.size and matchedcentroids.size and topn > 0:
                top_iso = np.sort(matchedcentroids[:, 1])[::-1][:topn]
                top_cent = np.sort(local_centroids[:, 1])[::-1][:topn]
                topthree = bool(np.array_equal(top_iso, top_cent))

            result["isodec_detail"] = {
                "local_centroids_n": int(local_centroids.shape[0]),
                "matched_peaks_n": int(len(matchedindexes)),
                "minpeaks_effective": int(minpeaks_eff),
                "css_thresh": float(ISODEC_CONFIG.css_thresh),
                "minareacovered": float(ISODEC_CONFIG.minareacovered),
                "areacovered": float(areacovered),
                "topthree": bool(topthree),
            }

        if not accepted:
            result["reason"] = "failed_isodec_rules"
            return result

    result["ok"] = True
    result["reason"] = "accepted"
    return result


def _vectorize_dist(
    dist: np.ndarray,
    sample_keys: np.ndarray,
    scale: float,
    mz_min=None,
    mz_max=None,
) -> np.ndarray:
    y = np.zeros(len(sample_keys), dtype=float)
    if dist is None or len(dist) == 0:
        return y
    mz = dist[:, 0]
    inten = dist[:, 1]
    if mz_min is not None:
        keep = mz >= float(mz_min)
        mz = mz[keep]
        inten = inten[keep]
    if mz_max is not None:
        keep = mz <= float(mz_max)
        mz = mz[keep]
        inten = inten[keep]
    if mz.size == 0:
        return y

    keys = np.rint(mz * scale).astype(np.int64)
    idx = np.searchsorted(sample_keys, keys)
    good = (idx >= 0) & (idx < len(sample_keys)) & (sample_keys[idx] == keys)
    if np.any(good):
        np.add.at(y, idx[good], inten[good])
    return y


def _sample_observed_intensities(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    target_mzs: np.ndarray,
    tol_ppm: float,
) -> np.ndarray:
    out = np.zeros(len(target_mzs), dtype=float)
    if len(spectrum_mz) == 0 or len(target_mzs) == 0:
        return out
    for i, mz in enumerate(target_mzs):
        idx = _nearest_peak_index(spectrum_mz, float(mz))
        mz_obs = float(spectrum_mz[idx])
        if _within_ppm(mz_obs, float(mz), tol_ppm):
            out[i] = float(spectrum_int[idx])
    return out


def _get_local_centroids_window(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    center_mz: float,
    lb: float,
    ub: float,
) -> np.ndarray:
    start = int(np.searchsorted(spectrum_mz, center_mz + float(lb), side="left"))
    end = int(np.searchsorted(spectrum_mz, center_mz + float(ub), side="right"))
    if end <= start:
        return np.empty((0, 2), dtype=float)
    out = np.empty((end - start, 2), dtype=float)
    out[:, 0] = spectrum_mz[start:end]
    out[:, 1] = spectrum_int[start:end]
    return out


def _isodec_css_and_accept(
    centroids: np.ndarray,
    isodist: np.ndarray,
    z: int,
    peakmz: float,
    config: IsoDecConfig,
) -> tuple[bool, float]:
    if centroids is None or centroids.size == 0 or isodist is None or isodist.size == 0:
        return False, 0.0
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        return False, 0.0
    if isodist.ndim != 2 or isodist.shape[1] != 2:
        return False, 0.0
    if (
        isodec_find_matched_intensities is None
        or isodec_calculate_cosinesimilarity is None
        or isodec_make_shifted_peak is None
    ):
        return False, 0.0

    max_shift = 1 if bool(config.minusoneaszero) else 0
    cent_intensities = isodec_find_matched_intensities(
        centroids[:, 0],
        centroids[:, 1],
        isodist[:, 0],
        int(max_shift),
        tolerance=float(config.matchtol),
        z=int(z),
        peakmz=float(peakmz),
    )

    max_theory = float(np.max(isodist[:, 1]))
    if max_theory <= 0.0:
        return False, 0.0
    max_obs = float(np.max(cent_intensities)) if len(cent_intensities) else 0.0
    norm_factor = max_obs / max_theory if max_obs > 0 else 1.0
    if not np.isfinite(norm_factor) or norm_factor <= 0:
        norm_factor = 1.0

    isodist_scaled = isodist.copy()
    isodist_scaled[:, 1] *= norm_factor
    css = float(
        isodec_calculate_cosinesimilarity(
            cent_intensities,
            isodist_scaled[:, 1],
            0,
            int(max_shift),
            minusoneaszero=bool(config.minusoneaszero),
        )
    )

    massdist = np.column_stack(
        [
            (isodist_scaled[:, 0] * float(z)) - (float(config.adductmass) * float(z)),
            isodist_scaled[:, 1],
        ]
    )
    monoiso = float(massdist[0, 0]) if massdist.size else 0.0
    peakmz_new, *_ = isodec_make_shifted_peak(
        0,
        float(css),
        float(monoiso),
        massdist,
        isodist_scaled,
        float(peakmz),
        int(z),
        centroids,
        float(config.matchtol),
        int(config.minpeaks),
        float(config.plusoneintwindowlb),
        float(config.plusoneintwindowub),
        float(config.css_thresh),
        float(config.minareacovered),
        bool(config.verbose),
    )
    return peakmz_new is not None, css


def _fit_simplex_mixture(y_obs: np.ndarray, components: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    if len(components) == 0:
        return np.empty(0, dtype=float), np.zeros_like(y_obs), 0.0
    A = np.column_stack(components)
    w_ls, *_ = np.linalg.lstsq(A, y_obs, rcond=None)
    w = _project_to_simplex(w_ls)
    y_pred = A @ w
    score = _cosine_similarity(y_obs, y_pred)
    return w, y_pred, score


def _plot_overlay(
    experimental: np.ndarray,
    overlays: list[tuple[np.ndarray, str, str]],
    mz_min=None,
    mz_max=None,
) -> None:
    """
    overlays: list of (isodist, color, label_text)
    """
    xmin = float(mz_min) if mz_min is not None else float(np.min(experimental[:, 0]))
    xmax = float(mz_max) if mz_max is not None else float(np.max(experimental[:, 0]))

    if cplot is not None:
        cplot(experimental, color="k", factor=1)
        for dist, color, _ in overlays:
            cplot(dist, color=color, factor=-1)
        plt.hlines(0, xmin, xmax, color="k", linewidth=0.8)
        if LABEL_MATCHES:
            for dist, color, label in overlays:
                peak_idx = int(np.argmax(dist[:, 1]))
                plt.text(float(dist[peak_idx, 0]), float(dist[peak_idx, 1]), label, fontsize=8, color=color)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        max_pos = float(np.max(experimental[:, 1]))
        max_neg = float(np.max([np.max(d[:, 1]) for d, _, _ in overlays])) if overlays else 0.0
        max_y = max(max_pos, max_neg)
        plt.ylim(-1.1 * max_y, 1.1 * max_y)
        if on_scroll is not None:
            plt.connect("scroll_event", on_scroll)
    else:
        plt.plot(experimental[:, 0], experimental[:, 1], color="k", linewidth=0.8, label="Experimental")
        for dist, color, label in overlays:
            plt.vlines(dist[:, 0], 0, dist[:, 1], color=color, linewidth=1.0, label=label)

    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    plt.show()


if PLOT_MODE.lower() == "precursor":
    base_comp = _residue_range_composition(residues, 0, len(residues)) + ms.Composition("H2O")
    if AMIDATED:
        base_comp += ms.Composition(AMIDATION_FORMULA)

    comp = base_comp * int(COPIES)
    comp += ms.Composition("H-2") * int(DISULFIDE_BONDS)

    theory_plot = _theoretical_isodist_from_comp(comp, CHARGE)
    if theory_plot.size == 0:
        raise ValueError("Theoretical distribution is empty after filtering; lower REL_INTENSITY_CUTOFF.")

    # Optionally align the theoretical most-abundant isotope to the nearest strong experimental peak.
    peak_theory_mz = float(theory_plot[np.argmax(theory_plot[:, 1]), 0])
    peak_obs_mz = None
    if ALIGN_TO_DATA:
        in_win = (spectrum[:, 0] >= peak_theory_mz - ALIGN_WINDOW_MZ) & (spectrum[:, 0] <= peak_theory_mz + ALIGN_WINDOW_MZ)
        if np.any(in_win):
            window = spectrum[in_win]
            peak_obs_mz = float(window[np.argmax(window[:, 1]), 0])
            theory_plot[:, 0] += peak_obs_mz - peak_theory_mz

    # Scale theoretical intensities to the experimental peak (or to max intensity if no local match).
    if peak_obs_mz is not None:
        idx = int(np.argmin(np.abs(spectrum[:, 0] - peak_obs_mz)))
        scale = float(spectrum[idx, 1]) / float(np.max(theory_plot[:, 1]))
    else:
        scale = float(np.max(spectrum[:, 1])) / float(np.max(theory_plot[:, 1]))
    theory_plot[:, 1] *= scale

    if MZ_MIN is not None or MZ_MAX is not None:
        mz_min = -np.inf if MZ_MIN is None else float(MZ_MIN)
        mz_max = np.inf if MZ_MAX is None else float(MZ_MAX)
        theory_plot = theory_plot[(theory_plot[:, 0] >= mz_min) & (theory_plot[:, 0] <= mz_max)]
        if theory_plot.size == 0:
            raise ValueError("Theoretical distribution is empty inside the selected m/z window.")

    _plot_overlay(
        spectrum,
        [(theory_plot, "tab:red", f"Precursor (z={CHARGE})")],
        mz_min=None if MZ_MIN is None else float(MZ_MIN),
        mz_max=None if MZ_MAX is None else float(MZ_MAX),
    )

elif PLOT_MODE.lower() == "fragments":
    if FRAG_MIN_CHARGE <= 0 or FRAG_MAX_CHARGE <= 0 or FRAG_MIN_CHARGE > FRAG_MAX_CHARGE:
        raise ValueError("Set FRAG_MIN_CHARGE/FRAG_MAX_CHARGE to a valid positive range.")

    if bool(ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = (
            float(MATCH_TOL_PPM)
            if FRAGMENT_INTENSITY_CAP_TOL_PPM is None
            else float(FRAGMENT_INTENSITY_CAP_TOL_PPM)
        )
        mz_min_cap = None if FRAGMENT_INTENSITY_CAP_MZ_MIN is None else float(FRAGMENT_INTENSITY_CAP_MZ_MIN)
        mz_max_cap = None if FRAGMENT_INTENSITY_CAP_MZ_MAX is None else float(FRAGMENT_INTENSITY_CAP_MZ_MAX)
        min_hits = int(max(0, FRAGMENT_INTENSITY_CAP_MIN_HITS))
        cap, hits = _compute_fragment_intensity_cap(
            residues,
            spectrum[:, 0],
            spectrum[:, 1],
            tol_ppm=float(tol_ppm),
            mz_min=mz_min_cap,
            mz_max=mz_max_cap,
        )
        if hits >= min_hits and cap > 0:
            n_before = int(len(spectrum))
            spectrum = _strip_peaks_above_intensity_cap(spectrum, cap=float(cap))
            removed = max(0, n_before - int(len(spectrum)))
            if removed > 0 or bool(FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: cap={cap:.3g} ({hits} windows), removed {removed} peaks"
                )
        else:
            if bool(FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: skipped (cap={cap:.3g}, hits={hits} < {min_hits})"
                )

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0

    ion_colors = {
        "b": "tab:blue",
        "y": "tab:orange",
        "c": "tab:green",
        "z": "tab:red",
        "z-dot": "tab:red",
        "c-dot": "tab:green",
    }

    matches: list[dict] = []
    n = len(residues)
    for ion_type in ION_TYPES:
        series = _ion_series(ion_type)
        allow_1h = bool(ENABLE_H_TRANSFER) and (series in set(H_TRANSFER_ION_TYPES_1H))
        allow_2h = bool(ENABLE_H_TRANSFER) and (series in set(H_TRANSFER_ION_TYPES_2H))
        for frag_len in range(1, n):
            frag_name, frag_comp = _ion_composition_from_sequence(residues, ion_type, frag_len, amidated=AMIDATED)
            for loss_suffix, loss_comp in _neutral_loss_variants(frag_comp, ion_series=series):
                for z in range(int(FRAG_MIN_CHARGE), int(FRAG_MAX_CHARGE) + 1):
                    try:
                        dist0 = _theoretical_isodist_from_comp(loss_comp, z)
                    except ValueError:
                        continue
                    if dist0.size == 0:
                        continue

                    # Hydrogen-transfer model: consider +/-H and +/-2H shifts in m/z by (H+/z).
                    shift_1 = float(H_TRANSFER_MASS) / float(z) if (allow_1h or allow_2h) else 0.0
                    shift_2 = 2.0 * float(H_TRANSFER_MASS) / float(z) if allow_2h else 0.0

                    dist_p1 = dist0.copy()
                    dist_p1[:, 0] += shift_1
                    dist_m1 = dist0.copy()
                    dist_m1[:, 0] -= shift_1

                    dist_p2 = None
                    dist_m2 = None
                    if allow_2h:
                        dist_p2 = dist0.copy()
                        dist_p2[:, 0] += shift_2
                        dist_m2 = dist0.copy()
                        dist_m2[:, 0] -= shift_2

                    dists_for_union = [dist0]
                    if allow_1h:
                        dists_for_union.extend([dist_p1, dist_m1])
                    if allow_2h:
                        dists_for_union.extend([dist_p2, dist_m2])

                    sample_keys, sample_mzs, scale = _build_sample_axis(
                        dists_for_union,
                        decimals=6,
                        mz_min=MZ_MIN,
                        mz_max=MZ_MAX,
                    )
                    if len(sample_mzs) == 0:
                        continue

                    y_obs = _sample_observed_intensities(
                        spectrum_mz, spectrum_int, sample_mzs, tol_ppm=float(MATCH_TOL_PPM)
                    )
                    y0 = _vectorize_dist(dist0, sample_keys, scale, mz_min=MZ_MIN, mz_max=MZ_MAX)

                    # Neutral (no H transfer) baseline score.
                    neutral_score = _cosine_similarity(y_obs, y0)
                    best_model = "neutral"
                    best_score = neutral_score
                    best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
                    best_pred = y0

                    # Fit +H/+2H and -H/-2H mixtures and compare by cosine similarity.
                    if allow_1h or allow_2h:
                        yp1 = (
                            _vectorize_dist(dist_p1, sample_keys, scale, mz_min=MZ_MIN, mz_max=MZ_MAX)
                            if allow_1h
                            else None
                        )
                        ym1 = (
                            _vectorize_dist(dist_m1, sample_keys, scale, mz_min=MZ_MIN, mz_max=MZ_MAX)
                            if allow_1h
                            else None
                        )
                        yp2 = (
                            _vectorize_dist(dist_p2, sample_keys, scale, mz_min=MZ_MIN, mz_max=MZ_MAX)
                            if allow_2h
                            else None
                        )
                        ym2 = (
                            _vectorize_dist(dist_m2, sample_keys, scale, mz_min=MZ_MIN, mz_max=MZ_MAX)
                            if allow_2h
                            else None
                        )

                        comps_plus = [("0", y0)]
                        comps_minus = [("0", y0)]
                        if allow_1h:
                            comps_plus.append(("+H", yp1))
                            comps_minus.append(("-H", ym1))
                        if allow_2h:
                            comps_plus.append(("+2H", yp2))
                            comps_minus.append(("-2H", ym2))

                        names_plus, vecs_plus = zip(*comps_plus)
                        w_plus, y_plus, score_plus = _fit_simplex_mixture(y_obs, list(vecs_plus))
                        weights_plus = dict(zip(names_plus, w_plus))

                        names_minus, vecs_minus = zip(*comps_minus)
                        w_minus, y_minus, score_minus = _fit_simplex_mixture(y_obs, list(vecs_minus))
                        weights_minus = dict(zip(names_minus, w_minus))

                        if score_plus > best_score:
                            best_model = "+mix"
                            best_score = score_plus
                            best_pred = y_plus
                            best_weights = {
                                "0": weights_plus.get("0", 0.0),
                                "+H": weights_plus.get("+H", 0.0),
                                "+2H": weights_plus.get("+2H", 0.0),
                                "-H": 0.0,
                                "-2H": 0.0,
                            }

                        if score_minus > best_score:
                            best_model = "-mix"
                            best_score = score_minus
                            best_pred = y_minus
                            best_weights = {
                                "0": weights_minus.get("0", 0.0),
                                "+H": 0.0,
                                "+2H": 0.0,
                                "-H": weights_minus.get("-H", 0.0),
                                "-2H": weights_minus.get("-2H", 0.0),
                            }

                        rel_improve = (best_score - neutral_score) / max(neutral_score, 1e-12)
                        if best_model != "neutral" and rel_improve < float(H_TRANSFER_MIN_REL_IMPROVEMENT):
                            best_model = "neutral"
                            best_score = neutral_score
                            best_pred = y0
                            best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}

                    if float(np.max(best_pred)) <= 0.0:
                        continue

                    # Anchor the predicted distribution to the nearest observed peak at its most intense isotope.
                    anchor_theory_mz = float(sample_mzs[int(np.argmax(best_pred))])
                    obs_idx = _nearest_peak_index(spectrum_mz, anchor_theory_mz)
                    obs_mz = float(spectrum_mz[obs_idx])
                    if not _within_ppm(obs_mz, anchor_theory_mz, float(MATCH_TOL_PPM)):
                        continue
                    obs_int = float(spectrum_int[obs_idx])
                    if float(MIN_OBS_REL_INT) > 0 and obs_int < obs_max * float(MIN_OBS_REL_INT):
                        continue
                    ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6

                    dist_plot = np.column_stack([sample_mzs.copy(), best_pred.copy()])
                    dist_plot[:, 0] += obs_mz - anchor_theory_mz
                    dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))

                    if MZ_MIN is not None or MZ_MAX is not None:
                        mz_min = -np.inf if MZ_MIN is None else float(MZ_MIN)
                        mz_max = np.inf if MZ_MAX is None else float(MZ_MAX)
                        dist_plot = dist_plot[(dist_plot[:, 0] >= mz_min) & (dist_plot[:, 0] <= mz_max)]
                        if dist_plot.size == 0:
                            continue

                    # Remove zero sticks and tiny sticks to keep the plot readable.
                    max_plot = float(np.max(dist_plot[:, 1]))
                    keep = dist_plot[:, 1] >= max_plot * float(REL_INTENSITY_CUTOFF)
                    dist_plot = dist_plot[keep]
                    if dist_plot.size == 0:
                        continue

                    # Apply IsoDec's built-in acceptance rules (min peaks, CSS threshold, area covered/top-three).
                    isodec_css = float(best_score)
                    if ENABLE_ISODEC_RULES and ISODEC_CONFIG is not None:
                        local_centroids = _get_local_centroids_window(
                            spectrum_mz,
                            spectrum_int,
                            obs_mz,
                            ISODEC_CONFIG.mzwindowlb,
                            ISODEC_CONFIG.mzwindowub,
                        )
                        accepted, isodec_css = _isodec_css_and_accept(
                            local_centroids, dist_plot, z=z, peakmz=obs_mz, config=ISODEC_CONFIG
                        )
                        if not accepted:
                            continue

                    frag_id = f"{frag_name}{loss_suffix}"
                    obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0
                    label_parts = [f"{frag_id}^{z}+", f"{ppm:.1f} ppm", f"css={isodec_css:.3f}"]
                    if best_model != "neutral":
                        h_pct = 100.0 * float(best_weights.get("+H", 0.0) + best_weights.get("-H", 0.0))
                        h2_pct = 100.0 * float(best_weights.get("+2H", 0.0) + best_weights.get("-2H", 0.0))
                        label_parts.append(f"%H={h_pct:.0f}")
                        if allow_2h:
                            label_parts.append(f"%2H={h2_pct:.0f}")
                        label_parts.append(best_model)
                    label = " | ".join(label_parts)
                    matches.append(
                        {
                            "frag_id": frag_id,
                            "ion_type": ion_type,
                            "series": series,
                            "frag_len": int(frag_len),
                            "charge": int(z),
                            "loss_suffix": loss_suffix,
                            "best_model": best_model,
                            "obs_idx": int(obs_idx),
                            "obs_mz": float(obs_mz),
                            "obs_int": obs_int,
                            "obs_rel_int": obs_rel_int,
                            "anchor_theory_mz": float(anchor_theory_mz),
                            "ppm": float(ppm),
                            "score": float(isodec_css),
                            "raw_score": float(best_score),
                            "neutral_score": float(neutral_score),
                            "h_weights": best_weights,
                            "dist": dist_plot,
                            "label": label,
                            "color": ion_colors.get(ion_type, "tab:purple"),
                        }
                    )

    # De-duplicate by observed peak index (keep the best scoring match per observed peak).
    best_by_obs: dict[int, dict] = {}
    for m in matches:
        key = m["obs_idx"]
        if key not in best_by_obs:
            best_by_obs[key] = m
            continue
        cur = best_by_obs[key]
        if m["score"] > cur["score"] or (m["score"] == cur["score"] and m["obs_int"] > cur["obs_int"]):
            best_by_obs[key] = m

    best = list(best_by_obs.values())
    best.sort(key=lambda d: (d["score"], d["obs_int"]), reverse=True)
    best = best[: int(MAX_PLOTTED_FRAGMENTS)]

    print(f"Matched fragments: {len(best)} (from {len(matches)} raw matches)")
    for m in best:
        print(
            f"{m['label']}\tI={m['obs_int']:.3g}\tcos0={m['neutral_score']:.3f}\trawcos={m['raw_score']:.3f}"
        )

    if bool(EXPORT_FRAGMENTS_CSV):
        out_dir = Path(__file__).parent / "match_outputs"
        file_tag = _sanitize_filename(Path(str(filepath)).stem)
        mz_tag = f"{'' if MZ_MIN is None else int(MZ_MIN)}-{'' if MZ_MAX is None else int(MZ_MAX)}"
        base = _sanitize_filename(f"fragments_scan{int(SCAN)}_{file_tag}_mz{mz_tag}")

        summary_path = (
            Path(FRAGMENTS_CSV_SUMMARY_PATH)
            if FRAGMENTS_CSV_SUMMARY_PATH
            else (out_dir / f"{base}_summary.csv")
        )
        peaks_path = (
            Path(FRAGMENTS_CSV_PEAKS_PATH)
            if FRAGMENTS_CSV_PEAKS_PATH
            else (out_dir / f"{base}_peaks.csv")
        )

        summary_rows = []
        peaks_rows = []
        for m in best:
            h = m.get("h_weights") or {}
            pct_h = 100.0 * float(h.get("+H", 0.0) + h.get("-H", 0.0))
            pct_2h = 100.0 * float(h.get("+2H", 0.0) + h.get("-2H", 0.0))
            loss_cols = _neutral_loss_columns(m.get("loss_suffix", ""))
            summary_rows.append(
                {
                    "peptide": str(PEPTIDE),
                    "copies": int(COPIES),
                    "amidated": bool(AMIDATED),
                    "disulfide_bonds": int(DISULFIDE_BONDS),
                    "mz_min": "" if MZ_MIN is None else float(MZ_MIN),
                    "mz_max": "" if MZ_MAX is None else float(MZ_MAX),
                    "match_tol_ppm": float(MATCH_TOL_PPM),
                    "min_obs_rel_int": float(MIN_OBS_REL_INT),
                    "rel_intensity_cutoff": float(REL_INTENSITY_CUTOFF),
                    "frag_id": m.get("frag_id", ""),
                    "ion_type": m.get("ion_type", ""),
                    "series": m.get("series", ""),
                    "frag_len": m.get("frag_len", ""),
                    "charge": m.get("charge", ""),
                    "H2O": loss_cols["H2O"],
                    "NH3": loss_cols["NH3"],
                    "CO": loss_cols["CO"],
                    "CO2": loss_cols["CO2"],
                    "2H2O": loss_cols["2H2O"],
                    "2NH3": loss_cols["2NH3"],
                    "%H": pct_h,
                    "%2H": pct_2h,
                    "obs_idx": m.get("obs_idx", ""),
                    "obs_mz": m.get("obs_mz", ""),
                    "obs_int": m.get("obs_int", ""),
                    "obs_rel_int": m.get("obs_rel_int", ""),
                    "anchor_theory_mz": m.get("anchor_theory_mz", ""),
                    "anchor_ppm": m.get("ppm", ""),
                    "css": m.get("score", ""),
                    "rawcos": m.get("raw_score", ""),
                    "cos0": m.get("neutral_score", ""),
                    "w0": h.get("0", ""),
                    "w_plusH": h.get("+H", ""),
                    "w_plus2H": h.get("+2H", ""),
                    "w_minusH": h.get("-H", ""),
                    "w_minus2H": h.get("-2H", ""),
                    "label": m.get("label", ""),
                }
            )

            dist = m.get("dist")
            if isinstance(dist, np.ndarray) and dist.size:
                for p in _match_theory_peaks(
                    spectrum_mz,
                    spectrum_int,
                    dist[:, 0],
                    tol_ppm=float(MATCH_TOL_PPM),
                    theory_int=dist[:, 1],
                ):
                    peaks_rows.append(
                        {
                            "frag_id": m.get("frag_id", ""),
                            "charge": m.get("charge", ""),
                            "H2O": loss_cols["H2O"],
                            "NH3": loss_cols["NH3"],
                            "CO": loss_cols["CO"],
                            "CO2": loss_cols["CO2"],
                            "2H2O": loss_cols["2H2O"],
                            "2NH3": loss_cols["2NH3"],
                            "%H": pct_h,
                            "%2H": pct_2h,
                            "css": m.get("score", ""),
                            "rawcos": m.get("raw_score", ""),
                            "theory_mz": p.get("theory_mz", ""),
                            "theory_int": p.get("theory_int", ""),
                            "obs_mz": p.get("obs_mz", ""),
                            "ppm": p.get("ppm", ""),
                            "obs_int": p.get("obs_int", ""),
                            "within": p.get("within", ""),
                            "obs_idx": p.get("obs_idx", ""),
                        }
                    )

        _write_csv(
            summary_path,
            [
                "peptide",
                "copies",
                "amidated",
                "disulfide_bonds",
                "mz_min",
                "mz_max",
                "match_tol_ppm",
                "min_obs_rel_int",
                "rel_intensity_cutoff",
                "frag_id",
                "ion_type",
                "series",
                "frag_len",
                "charge",
                "H2O",
                "NH3",
                "CO",
                "CO2",
                "2H2O",
                "2NH3",
                "%H",
                "%2H",
                "obs_idx",
                "obs_mz",
                "obs_int",
                "obs_rel_int",
                "anchor_theory_mz",
                "anchor_ppm",
                "css",
                "rawcos",
                "cos0",
                "w0",
                "w_plusH",
                "w_plus2H",
                "w_minusH",
                "w_minus2H",
                "label",
            ],
            summary_rows,
        )
        _write_csv(
            peaks_path,
            [
                "frag_id",
                "charge",
                "H2O",
                "NH3",
                "CO",
                "CO2",
                "2H2O",
                "2NH3",
                "%H",
                "%2H",
                "css",
                "rawcos",
                "theory_mz",
                "theory_int",
                "obs_mz",
                "ppm",
                "obs_int",
                "within",
                "obs_idx",
            ],
            peaks_rows,
        )
        print(f"Wrote CSV: {summary_path}")
        print(f"Wrote CSV: {peaks_path}")

    overlays = [(m["dist"], m["color"], m["label"]) for m in best]
    _plot_overlay(
        spectrum,
        overlays,
        mz_min=None if MZ_MIN is None else float(MZ_MIN),
        mz_max=None if MZ_MAX is None else float(MZ_MAX),
    )

elif PLOT_MODE.lower() == "diagnose":
    if not DIAGNOSE_ION_SPEC:
        raise ValueError('Set DIAGNOSE_ION_SPEC (e.g., "c7^2+" or "z12-2H2O^3+") when using PLOT_MODE="diagnose".')

    if bool(ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = (
            float(MATCH_TOL_PPM)
            if FRAGMENT_INTENSITY_CAP_TOL_PPM is None
            else float(FRAGMENT_INTENSITY_CAP_TOL_PPM)
        )
        mz_min_cap = None if FRAGMENT_INTENSITY_CAP_MZ_MIN is None else float(FRAGMENT_INTENSITY_CAP_MZ_MIN)
        mz_max_cap = None if FRAGMENT_INTENSITY_CAP_MZ_MAX is None else float(FRAGMENT_INTENSITY_CAP_MZ_MAX)
        min_hits = int(max(0, FRAGMENT_INTENSITY_CAP_MIN_HITS))
        cap, hits = _compute_fragment_intensity_cap(
            residues,
            spectrum[:, 0],
            spectrum[:, 1],
            tol_ppm=float(tol_ppm),
            mz_min=mz_min_cap,
            mz_max=mz_max_cap,
        )
        if hits >= min_hits and cap > 0:
            n_before = int(len(spectrum))
            spectrum = _strip_peaks_above_intensity_cap(spectrum, cap=float(cap))
            removed = max(0, n_before - int(len(spectrum)))
            if removed > 0 or bool(FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: cap={cap:.3g} ({hits} windows), removed {removed} peaks"
                )
        else:
            if bool(FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: skipped (cap={cap:.3g}, hits={hits} < {min_hits})"
                )

    ion_type, frag_len, loss_formula, loss_count, charge = _parse_fragment_spec(DIAGNOSE_ION_SPEC)
    charges = [int(charge)] if charge is not None else list(range(int(FRAG_MIN_CHARGE), int(FRAG_MAX_CHARGE) + 1))
    if charge is None and not bool(DIAGNOSE_SCAN_CHARGES):
        raise ValueError("Ion spec has no charge; set DIAGNOSE_SCAN_CHARGES=True or include ^z+ (e.g., c7^2+).")

    try:
        h_transfer = int(DIAGNOSE_H_TRANSFER)
    except Exception as e:
        raise ValueError("DIAGNOSE_H_TRANSFER must be an integer in {-2,-1,0,1,2}.") from e
    if h_transfer not in (-2, -1, 0, 1, 2):
        raise ValueError("DIAGNOSE_H_TRANSFER must be an integer in {-2,-1,0,1,2}.")

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]

    print("=== Ion diagnostic ===")
    print(f"Spec: {DIAGNOSE_ION_SPEC}")
    print(f"Parsed: ion_type={ion_type}, frag_len={frag_len}, loss={loss_formula or 'none'} x{loss_count or 0}")
    print(f"H transfer: {h_transfer:+d} H+")
    print(f"Charge(s): {charges}")

    diagnose_dir = Path(__file__).parent / "diagnose_outputs"
    spec_safe = _sanitize_filename(DIAGNOSE_ION_SPEC)
    base = f"diagnose_scan{int(SCAN)}_{spec_safe}_h{h_transfer:+d}"
    base = _sanitize_filename(base)
    summary_path = (
        Path(DIAGNOSE_CSV_SUMMARY_PATH)
        if DIAGNOSE_CSV_SUMMARY_PATH
        else (diagnose_dir / f"{base}_summary.csv")
    )
    peaks_path = (
        Path(DIAGNOSE_CSV_PEAKS_PATH)
        if DIAGNOSE_CSV_PEAKS_PATH
        else (diagnose_dir / f"{base}_peaks.csv")
    )

    results = []
    for z in charges:
        r = _diagnose_candidate(
            residues=residues,
            ion_type=ion_type,
            frag_len=frag_len,
            z=int(z),
            loss_formula=loss_formula,
            loss_count=int(loss_count),
            h_transfer=h_transfer,
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
        )
        results.append(r)

    # Prefer accepted candidates; otherwise show the best IsoDec CSS (if present) or raw cosine.
    def _rank_key(d: dict):
        css = d.get("isodec_css", float("nan"))
        raw = d.get("raw_cosine_preanchor", 0.0)
        ok = 1 if d.get("ok") else 0
        css_val = css if np.isfinite(css) else -1.0
        return (ok, css_val, raw)

    results.sort(key=_rank_key, reverse=True)

    # CSV export (summary + per-isotope peak table).
    if bool(DIAGNOSE_EXPORT_CSV):
        summary_rows = []
        peaks_rows = []
        for r in results:
            z = r["z"]
            frag_name = r.get("frag_name", f"{ion_type}{frag_len}")
            loss = ""
            if loss_formula and loss_count:
                loss = _neutral_loss_label(int(loss_count), loss_formula)
            label = f"{frag_name}{loss}^{z}+"

            detail = r.get("isodec_detail") if isinstance(r.get("isodec_detail"), dict) else {}
            summary_rows.append(
                {
                    "spec": str(DIAGNOSE_ION_SPEC),
                    "label": label,
                    "ion_type": r.get("ion_type", ""),
                    "frag_name": frag_name,
                    "frag_len": r.get("frag_len", ""),
                    "z": int(z),
                    "loss_formula": r.get("loss_formula", ""),
                    "loss_count": r.get("loss_count", ""),
                    "h_transfer": r.get("h_transfer", ""),
                    "ok": bool(r.get("ok", False)),
                    "reason": r.get("reason", ""),
                    "raw_cosine_preanchor": r.get("raw_cosine_preanchor", ""),
                    "anchor_theory_mz": r.get("anchor_theory_mz", ""),
                    "anchor_obs_mz": r.get("anchor_obs_mz", ""),
                    "anchor_ppm": r.get("anchor_ppm", ""),
                    "anchor_within_ppm": bool(r.get("anchor_within_ppm", False)),
                    "obs_idx": r.get("obs_idx", ""),
                    "obs_int": r.get("obs_int", ""),
                    "obs_rel_int": r.get("obs_rel_int", ""),
                    "isodec_css": r.get("isodec_css", ""),
                    "isodec_accepted": bool(r.get("isodec_accepted", False)),
                    "isodec_local_centroids_n": detail.get("local_centroids_n", ""),
                    "isodec_matched_peaks_n": detail.get("matched_peaks_n", ""),
                    "isodec_minpeaks_effective": detail.get("minpeaks_effective", ""),
                    "isodec_areacovered": detail.get("areacovered", ""),
                    "isodec_topthree": detail.get("topthree", ""),
                    "match_tol_ppm": float(MATCH_TOL_PPM),
                    "min_obs_rel_int": float(MIN_OBS_REL_INT),
                    "rel_intensity_cutoff": float(REL_INTENSITY_CUTOFF),
                    "mz_min": "" if MZ_MIN is None else float(MZ_MIN),
                    "mz_max": "" if MZ_MAX is None else float(MZ_MAX),
                }
            )

            for p in r.get("theory_matches", []) if isinstance(r.get("theory_matches"), list) else []:
                peaks_rows.append(
                    {
                        "spec": str(DIAGNOSE_ION_SPEC),
                        "label": label,
                        "z": int(z),
                        "h_transfer": r.get("h_transfer", ""),
                        "theory_mz": p.get("theory_mz", ""),
                        "theory_int": p.get("theory_int", ""),
                        "obs_mz": p.get("obs_mz", ""),
                        "ppm": p.get("ppm", ""),
                        "obs_int": p.get("obs_int", ""),
                        "within": p.get("within", ""),
                        "obs_idx": p.get("obs_idx", ""),
                    }
                )

        _write_csv(
            summary_path,
            [
                "spec",
                "label",
                "ion_type",
                "frag_name",
                "frag_len",
                "z",
                "loss_formula",
                "loss_count",
                "h_transfer",
                "ok",
                "reason",
                "raw_cosine_preanchor",
                "anchor_theory_mz",
                "anchor_obs_mz",
                "anchor_ppm",
                "anchor_within_ppm",
                "obs_idx",
                "obs_int",
                "obs_rel_int",
                "isodec_css",
                "isodec_accepted",
                "isodec_local_centroids_n",
                "isodec_matched_peaks_n",
                "isodec_minpeaks_effective",
                "isodec_areacovered",
                "isodec_topthree",
                "match_tol_ppm",
                "min_obs_rel_int",
                "rel_intensity_cutoff",
                "mz_min",
                "mz_max",
            ],
            summary_rows,
        )
        _write_csv(
            peaks_path,
            [
                "spec",
                "label",
                "z",
                "h_transfer",
                "theory_mz",
                "theory_int",
                "obs_mz",
                "ppm",
                "obs_int",
                "within",
                "obs_idx",
            ],
            peaks_rows,
        )
        print(f"Wrote CSV: {summary_path}")
        print(f"Wrote CSV: {peaks_path}")

    for r in results:
        z = r["z"]
        frag_name = r.get("frag_name", f"{ion_type}{frag_len}")
        loss = ""
        if loss_formula and loss_count:
            loss = _neutral_loss_label(int(loss_count), loss_formula)
        label = f"{frag_name}{loss}^{z}+"
        raw = r.get("raw_cosine_preanchor", 0.0)
        css = r.get("isodec_css", None)
        css_txt = f"{css:.3f}" if isinstance(css, (int, float)) and np.isfinite(css) else "n/a"
        print(f"- {label}\tok={r['ok']}\treason={r['reason']}\trawcos={raw:.3f}\tcss={css_txt}")
        if r.get("anchor_within_ppm"):
            print(
                f"  anchor: theory={r['anchor_theory_mz']:.6f}\tobs={r['anchor_obs_mz']:.6f}\tppm={r.get('anchor_ppm', 0.0):.1f}\t"
                f"I={r.get('obs_int', 0.0):.3g}\trelI={100.0*r.get('obs_rel_int', 0.0):.2f}%"
            )
        detail = r.get("isodec_detail")
        if isinstance(detail, dict):
            print(
                f"  isodec: matched={detail.get('matched_peaks_n')}/{detail.get('local_centroids_n')}\t"
                f"minpeaks={detail.get('minpeaks_effective')}\t"
                f"areacovered={detail.get('areacovered'):.3f}\tminarea={detail.get('minareacovered'):.3f}\t"
                f"top={detail.get('topthree')}\tcss_thresh={detail.get('css_thresh'):.2f}"
            )

        if DIAGNOSE_MAX_TABLE_ROWS and isinstance(r.get("theory_matches"), list):
            matches = r["theory_matches"]
            # Prioritize within-ppm matches, then by observed intensity.
            matches_sorted = sorted(matches, key=lambda x: (x["within"], x["obs_int"]), reverse=True)
            print("  peaks (theory_mz -> obs_mz, ppm, I):")
            for row in matches_sorted[: int(DIAGNOSE_MAX_TABLE_ROWS)]:
                flag = "*" if row["within"] else " "
                print(
                    f"   {flag} {row['theory_mz']:.6f} -> {row['obs_mz']:.6f}\tppm={row['ppm']:+.1f}\tI={row['obs_int']:.3g}"
                )

    best = results[0] if results else None
    if bool(DIAGNOSE_SHOW_PLOT) and isinstance(best, dict) and isinstance(best.get("dist_plot"), np.ndarray):
        z = best["z"]
        frag_name = best.get("frag_name", f"{ion_type}{frag_len}")
        loss = ""
        if loss_formula and loss_count:
            loss = _neutral_loss_label(int(loss_count), loss_formula)
        label = f"{frag_name}{loss}^{z}+"
        _plot_overlay(
            spectrum,
            [(best["dist_plot"], "tab:purple", f"diagnose {label}")],
            mz_min=None if MZ_MIN is None else float(MZ_MIN),
            mz_max=None if MZ_MAX is None else float(MZ_MAX),
        )

else:
    raise ValueError('PLOT_MODE must be "precursor", "fragments", or "diagnose".')
