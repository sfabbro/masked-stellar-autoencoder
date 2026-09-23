"""Column registry. A new survey is a new entry, not a new loss."""

from __future__ import annotations

from dataclasses import dataclass

# Photometry (magnitudes, plus reddening). Error names are the catalogue names,
# not a copy of the value column.
_PHOTOMETRY: tuple[tuple[str, str | None, str], ...] = (
    ("W1", None, "mag"),
    ("W2", None, "mag"),
    ("G", None, "mag"),
    ("BP", None, "mag"),
    ("RP", None, "mag"),
    ("U_SMSS", "E_U_SMSS", "mag"),
    ("V_SMSS", "E_V_SMSS", "mag"),
    ("G_SMSS", "E_G_SMSS", "mag"),
    ("R_SMSS", "E_R_SMSS", "mag"),
    ("I_SMSS", "E_I_SMSS", "mag"),
    ("Z_SMSS", "E_Z_SMSS", "mag"),
    ("U_SDSS", "E_U_SDSS", "mag"),
    ("G_SDSS", "E_G_SDSS", "mag"),
    ("R_SDSS", "E_R_SDSS", "mag"),
    ("I_SDSS", "E_I_SDSS", "mag"),
    ("Z_SDSS", "E_Z_SDSS", "mag"),
    ("G_PS1", "E_G_PS1", "mag"),
    ("R_PS1", "E_R_PS1", "mag"),
    ("I_PS1", "E_I_PS1", "mag"),
    ("Z_PS1", "E_Z_PS1", "mag"),
    ("Y_PS1", "E_Y_PS1", "mag"),
    ("J", "E_J", "mag"),
    ("H", "E_H", "mag"),
    ("KS", "E_KS", "mag"),
    ("EBV", None, "ebv"),
)

_ASTROMETRY: tuple[tuple[str, str], ...] = (
    ("PARALLAX", "e_parallax"),
    ("pmra", "e_pmra"),
    ("pmdec", "e_pmdec"),
)

_LABELS: tuple[tuple[str, str, str, str], ...] = (
    ("teff", "e_teff", "K", "log10"),
    ("logg", "e_logg", "dex", "robust"),
    ("fe_h", "e_fe_h", "dex", "robust"),
    ("alpha", "e_alpha", "dex", "robust"),
    ("age", "e_age", "Gyr", "robust"),
    ("parallax", "e_parallax", "mas", "asinh"),
)


@dataclass(frozen=True)
class Column:
    name: str
    group: str
    unit: str
    error: str | None = None
    transform: str = "robust"


@dataclass(frozen=True)
class ColumnRegistry:
    """Widths come from this object. The loss never assumes 55 XP orders."""

    columns: tuple[Column, ...]
    g_ref: float = 8.5
    g_column: str = "G"

    def names(self, group: str) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns if col.group == group)

    def group(self, name: str) -> tuple[Column, ...]:
        return tuple(col for col in self.columns if col.group == name)

    def index(self, group: str, column: str) -> int:
        names = self.names(group)
        return names.index(column)


def gaia_dr3_registry(
    n_bp: int = 55, n_rp: int = 55, g_ref: float = 8.5
) -> ColumnRegistry:
    """DR3 is 55. A DR4 file passes its own n_bp and n_rp."""
    if n_bp < 1 or n_rp < 1:
        raise ValueError(f"n_bp and n_rp must be positive, got {n_bp}, {n_rp}")
    columns: list[Column] = []
    for i in range(1, n_bp + 1):
        columns.append(Column(f"bp_{i}", "xp_bp", "coeff", f"bpe_{i}", "xp"))
    for i in range(1, n_rp + 1):
        columns.append(Column(f"rp_{i}", "xp_rp", "coeff", f"rpe_{i}", "xp"))
    for name, error, unit in _PHOTOMETRY:
        columns.append(Column(name, "photometry", unit, error, "robust"))
    for name, error in _ASTROMETRY:
        columns.append(Column(name, "astrometry", "mas", error, "snr"))
    for name, error, unit, transform in _LABELS:
        columns.append(Column(name, "labels", unit, error, transform))
    return ColumnRegistry(tuple(columns), g_ref=g_ref)
