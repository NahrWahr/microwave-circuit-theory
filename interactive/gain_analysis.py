"""
RF Amplifier Gain Analysis
Two Smith Charts (ΓS / ΓL planes) with gain circles, stability circles,
and real-time GT / GP / GA / MAG metrics.
"""

import sys
import numpy as np
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QLabel, QDoubleSpinBox, QCheckBox,
    QComboBox, QPushButton, QFormLayout, QFrame, QSizePolicy,
    QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

# ─────────────────────────────────────────────────────────────────────────────
# RF Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def gamma_to_z(gamma: complex, z0: float = 50.0) -> complex:
    d = 1.0 - gamma
    return z0 * (1.0 + gamma) / d if abs(d) > 1e-10 else complex(1e9, 0)


def z_to_gamma(z: complex, z0: float = 50.0) -> complex:
    zn = z / z0
    d = zn + 1.0
    return (zn - 1.0) / d if abs(d) > 1e-10 else complex(-1, 0)


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin_to_db(x: float) -> float:
    return 20.0 * np.log10(max(abs(x), 1e-15))


def pwr_db(x: float) -> float:
    return 10.0 * np.log10(max(abs(x), 1e-15))


def resistance_circle_pts(r: float, n: int = 300):
    cx = r / (r + 1.0)
    rad = 1.0 / (r + 1.0)
    theta = np.linspace(0, 2 * np.pi, n)
    re = cx + rad * np.cos(theta)
    im = rad * np.sin(theta)
    mask = re ** 2 + im ** 2 <= 1.0 + 1e-6
    return re[mask], im[mask]


def reactance_arc_pts(x: float, n: int = 400):
    if abs(x) < 1e-9:
        return np.array([]), np.array([])
    cy = 1.0 / x
    rad = 1.0 / abs(x)
    theta = np.linspace(0, 2 * np.pi, n)
    re = 1.0 + rad * np.cos(theta)
    im = cy + rad * np.sin(theta)
    mask = re ** 2 + im ** 2 <= 1.0 + 1e-6
    re, im = re[mask], im[mask]
    if len(re) < 2:
        return np.array([]), np.array([])
    angles = np.arctan2(im - cy, re - 1.0)
    order = np.argsort(angles)
    return re[order], im[order]


def circle_pts(cx: float, cy: float, r: float, n: int = 300):
    """Points on a circle, clipped to |Γ| ≤ 1."""
    theta = np.linspace(0, 2 * np.pi, n)
    re = cx + r * np.cos(theta)
    im = cy + r * np.sin(theta)
    return re, im


# ─────────────────────────────────────────────────────────────────────────────
# Wave diagram layout constants
# ─────────────────────────────────────────────────────────────────────────────

_SRC_XC   = -2.00    # source bar centre x (display units)
_SRC_HW   =  0.07    # bar half-width
_DUT_L    = -0.22    # DUT box left  edge
_DUT_R    =  0.22    # DUT box right edge
_LOAD_XC  =  2.00    # load bar centre x
_LOAD_HW  =  0.07
_WL_LEFT  =  2.0     # wavelengths on left TL
_WL_RIGHT =  2.0     # wavelengths on right TL
_N_PTS    =  500
_PHASE_STEP = 2 * np.pi / 40   # 1 full cycle in ~1.3 s at 30 fps

_pos_l = np.linspace(0, _WL_LEFT,  _N_PTS)
_pos_r = np.linspace(0, _WL_RIGHT, _N_PTS)
# Map wavelength positions → display x for each TL
_x_l = np.linspace(_SRC_XC + _SRC_HW, _DUT_L,            _N_PTS)
_x_r = np.linspace(_DUT_R,             _LOAD_XC - _LOAD_HW, _N_PTS)


def _waves_left(gamma_in: complex, phase: float):
    """Incident (→), reflected-from-DUT (←), and total waves on left TL."""
    bz    = 2.0 * np.pi * _pos_l
    v_inc = np.cos(phase - bz)
    v_ref = np.real(gamma_in * np.exp(1j * (phase + bz)))
    return v_inc, v_ref, v_inc + v_ref


def _waves_right(b2: complex, gamma_L: complex, phase: float):
    """Forward (→), backward-from-load (←), and total waves on right TL."""
    bz     = 2.0 * np.pi * _pos_r
    bz_end = 2.0 * np.pi * _WL_RIGHT
    v_fwd  = np.real(b2 * np.exp(1j * (phase - bz)))
    v_bwd  = np.real(b2 * gamma_L * np.exp(1j * (phase + bz - 2.0 * bz_end)))
    return v_fwd, v_bwd, v_fwd + v_bwd


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GainState:
    z0:      float   = 50.0
    gamma_S: complex = complex(0, 0)
    s11:     complex = complex(0.447, -0.239)   # LNA-like default
    s12:     complex = complex(0.040,  0.012)
    s21:     complex = complex(2.800,  0.400)
    s22:     complex = complex(0.500, -0.436)
    gamma_L: complex = complex(0, 0)

    show_unilateral: bool = True
    show_bilateral:  bool = True
    show_stability:  bool = True
    show_vswr:       bool = False
    gain_steps:      int  = 6

    # ── derived S-param quantities ──────────────────────────────────────────

    @property
    def delta(self) -> complex:
        return self.s11 * self.s22 - self.s12 * self.s21

    @property
    def k(self) -> float:
        num = 1.0 - abs(self.s11)**2 - abs(self.s22)**2 + abs(self.delta)**2
        den = 2.0 * abs(self.s12 * self.s21)
        return num / den if den > 1e-12 else float("inf")

    @property
    def mu(self) -> float:
        """μ-factor (Edwards-Sinsky) — k>1 AND μ>1 → unconditionally stable."""
        den = abs(self.s22)**2 - abs(self.delta)**2
        if abs(den) < 1e-12:
            return float("inf")
        return (1.0 - abs(self.s11)**2) / (abs(self.s22 - np.conj(self.delta) * self.s11) + abs(self.s12 * self.s21))

    @property
    def unconditionally_stable(self) -> bool:
        return self.k >= 1.0 and abs(self.delta) < 1.0

    @property
    def gamma_in(self) -> complex:
        d = 1.0 - self.s22 * self.gamma_L
        if abs(d) < 1e-10:
            return self.s11
        return self.s11 + self.s12 * self.s21 * self.gamma_L / d

    @property
    def gamma_out(self) -> complex:
        d = 1.0 - self.s11 * self.gamma_S
        if abs(d) < 1e-10:
            return self.s22
        return self.s22 + self.s12 * self.s21 * self.gamma_S / d

    # ── power gains ─────────────────────────────────────────────────────────

    @property
    def GT(self) -> float:
        gin = self.gamma_in
        num = (1.0 - abs(self.gamma_S)**2) * abs(self.s21)**2 * (1.0 - abs(self.gamma_L)**2)
        den = abs(1.0 - gin * self.gamma_S)**2 * abs(1.0 - self.s22 * self.gamma_L)**2
        return num / den if den > 1e-30 else 0.0

    @property
    def GP(self) -> float:
        gin = self.gamma_in
        num = abs(self.s21)**2 * (1.0 - abs(self.gamma_L)**2)
        den = (1.0 - abs(gin)**2) * abs(1.0 - self.s22 * self.gamma_L)**2
        return num / den if den > 1e-30 else 0.0

    @property
    def GA(self) -> float:
        gout = self.gamma_out
        num = abs(self.s21)**2 * (1.0 - abs(self.gamma_S)**2)
        den = abs(1.0 - self.s11 * self.gamma_S)**2 * (1.0 - abs(gout)**2)
        return num / den if den > 1e-30 else 0.0

    @property
    def GT_unilateral(self) -> float:
        gs = (1.0 - abs(self.gamma_S)**2) / abs(1.0 - self.s11 * self.gamma_S)**2
        g0 = abs(self.s21)**2
        gl = (1.0 - abs(self.gamma_L)**2) / abs(1.0 - self.s22 * self.gamma_L)**2
        return gs * g0 * gl

    @property
    def MAG(self) -> float:
        k = self.k
        if k < 1.0 or abs(self.s12) < 1e-12:
            return float("nan")
        return abs(self.s21 / self.s12) * (k - np.sqrt(k**2 - 1.0))

    @property
    def MSG(self) -> float:
        return abs(self.s21 / self.s12) if abs(self.s12) > 1e-12 else float("inf")

    # ── optimal conjugate match points ──────────────────────────────────────

    @property
    def gamma_S_opt(self) -> complex:
        """ΓS for conjugate match to input (max GA) — bilateral."""
        d = abs(self.s11)**2 - abs(self.delta)**2
        if abs(d) < 1e-10:
            return np.conj(self.s11)
        return np.conj((self.s11 - self.delta * np.conj(self.s22)) / d)

    @property
    def gamma_L_opt(self) -> complex:
        """ΓL for conjugate match to output (max GP) — bilateral."""
        d = abs(self.s22)**2 - abs(self.delta)**2
        if abs(d) < 1e-10:
            return np.conj(self.s22)
        return np.conj((self.s22 - self.delta * np.conj(self.s11)) / d)

    @property
    def gamma_S_opt_uni(self) -> complex:
        """ΓS for unilateral max GS."""
        return np.conj(self.s11)

    @property
    def gamma_L_opt_uni(self) -> complex:
        """ΓL for unilateral max GL."""
        return np.conj(self.s22)

    # ── gain-circle geometry ─────────────────────────────────────────────────

    def unilateral_source_circles(self, n_steps: int):
        """
        Returns list of (cx, cy, r, gain_lin) for n_steps GS gain circles
        in the ΓS plane.  GS_max = 1/(1−|S11|²).
        """
        s11 = self.s11
        s11_sq = abs(s11)**2
        gs_max_lin = 1.0 / (1.0 - s11_sq) if s11_sq < 0.9999 else 1e6

        circles = []
        for i in range(n_steps):
            # gs_norm from 0 → 1 (exclusive of 1 to avoid radius=0)
            g_norm = (i + 0.5) / n_steps          # normalised GS·(1−|S11|²)
            denom = 1.0 - (1.0 - g_norm) * s11_sq
            if abs(denom) < 1e-10:
                continue
            cx_c = g_norm * np.conj(s11) / denom
            r_c  = np.sqrt(1.0 - g_norm) * (1.0 - s11_sq) / abs(denom)
            gain_lin = g_norm / (1.0 - s11_sq) if (1.0 - s11_sq) > 1e-10 else 0.0
            circles.append((cx_c.real, cx_c.imag, r_c, gain_lin))
        return circles

    def unilateral_load_circles(self, n_steps: int):
        """Constant-GL circles in the ΓL plane."""
        s22 = self.s22
        s22_sq = abs(s22)**2

        circles = []
        for i in range(n_steps):
            g_norm = (i + 0.5) / n_steps
            denom = 1.0 - (1.0 - g_norm) * s22_sq
            if abs(denom) < 1e-10:
                continue
            cx_c = g_norm * np.conj(s22) / denom
            r_c  = np.sqrt(1.0 - g_norm) * (1.0 - s22_sq) / abs(denom)
            gain_lin = g_norm / (1.0 - s22_sq) if (1.0 - s22_sq) > 1e-10 else 0.0
            circles.append((cx_c.real, cx_c.imag, r_c, gain_lin))
        return circles

    def bilateral_GA_circles(self, n_steps: int):
        """Constant-GA circles in the ΓS plane."""
        C1 = self.s11 - self.delta * np.conj(self.s22)
        s21_sq = abs(self.s21)**2
        k = self.k
        s12s21 = abs(self.s12 * self.s21)
        d_s11 = abs(self.s11)**2 - abs(self.delta)**2

        if abs(d_s11) < 1e-10 or s21_sq < 1e-12:
            return []

        ga_max = self.MAG if self.unconditionally_stable else self.MSG * 2

        circles = []
        for i in range(n_steps):
            frac = (i + 0.5) / n_steps
            ga = frac * ga_max
            ga_n = ga / s21_sq   # normalised
            denom_val = 1.0 + ga_n * d_s11
            if abs(denom_val) < 1e-10:
                continue
            da = ga_n * np.conj(C1) / denom_val
            disc = 1.0 - 2.0 * k * s12s21 * ga_n + (s12s21 * ga_n)**2
            if disc < 0:
                disc = 0.0
            ra = np.sqrt(disc) / abs(denom_val)
            circles.append((da.real, da.imag, ra, ga))
        return circles

    def bilateral_GP_circles(self, n_steps: int):
        """Constant-GP circles in the ΓL plane."""
        C2 = self.s22 - self.delta * np.conj(self.s11)
        s21_sq = abs(self.s21)**2
        k = self.k
        s12s21 = abs(self.s12 * self.s21)
        d_s22 = abs(self.s22)**2 - abs(self.delta)**2

        if abs(d_s22) < 1e-10 or s21_sq < 1e-12:
            return []

        gp_max = self.MAG if self.unconditionally_stable else self.MSG * 2

        circles = []
        for i in range(n_steps):
            frac = (i + 0.5) / n_steps
            gp = frac * gp_max
            gp_n = gp / s21_sq
            denom_val = 1.0 + gp_n * d_s22
            if abs(denom_val) < 1e-10:
                continue
            dp = gp_n * np.conj(C2) / denom_val
            disc = 1.0 - 2.0 * k * s12s21 * gp_n + (s12s21 * gp_n)**2
            if disc < 0:
                disc = 0.0
            rp = np.sqrt(disc) / abs(denom_val)
            circles.append((dp.real, dp.imag, rp, gp))
        return circles

    def source_stability_circle(self):
        """(cx, cy, r) of source stability circle in ΓS plane."""
        d = abs(self.s11)**2 - abs(self.delta)**2
        if abs(d) < 1e-10:
            return None
        cs = np.conj(self.s11 - self.delta * np.conj(self.s22)) / d
        rs = abs(self.s12 * self.s21) / abs(d)
        return cs.real, cs.imag, rs

    def load_stability_circle(self):
        """(cx, cy, r) of load stability circle in ΓL plane."""
        d = abs(self.s22)**2 - abs(self.delta)**2
        if abs(d) < 1e-10:
            return None
        cl = np.conj(self.s22 - self.delta * np.conj(self.s11)) / d
        rl = abs(self.s12 * self.s21) / abs(d)
        return cl.real, cl.imag, rl


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

BG         = "#12121f"
PANEL_BG   = "#1a1a2e"
GRID_MAJ   = "#2a3060"
GRID_MIN   = "#1e2248"
AXIS_COL   = "#444466"
RIM_COL    = "#6677bb"
LBL_COL    = "#8899cc"
TXT_COL    = "#ccccee"
GS_COL     = "#ff4455"   # ΓS point — red
GL_COL     = "#4488ff"   # ΓL point — blue
STAB_COL   = "#ff6644"   # stability circle
OPT_COL    = "#ffee44"   # optimal Γ star
GAIN_LOW   = "#2244aa"   # gain circle — low gain
GAIN_HIGH  = "#ff4455"   # gain circle — high gain

R_VALUES   = [0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
X_VALUES   = [0.2, 0.5, 1.0, 2.0, 5.0]
DRAG_THR   = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Gain Circle Canvas (reusable for ΓS and ΓL planes)
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_color(t: float, c0: str, c1: str) -> str:
    """Linear interpolate between two hex colours, t ∈ [0,1]."""
    def _parse(c):
        c = c.lstrip("#")
        return [int(c[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    r0, g0, b0 = _parse(c0)
    r1, g1, b1 = _parse(c1)
    r = r0 + t * (r1 - r0)
    g = g0 + t * (g1 - g0)
    b = b0 + t * (b1 - b0)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


class GainCircleCanvas(FigureCanvas):
    """
    Interactive Smith Chart with gain circles for either the ΓS or ΓL plane.
    plane = "source"  → draws GS (unilateral) / GA (bilateral) circles, draggable ΓS
    plane = "load"    → draws GL (unilateral) / GP (bilateral) circles, draggable ΓL
    """

    stateChanged = pyqtSignal(object)   # emits GainState

    def __init__(self, state: GainState, plane: str = "source", parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.state = state
        self.plane = plane

        self._dragging = False
        self._grid_artists  = []
        self._circle_artists = []
        self._stab_artist   = None
        self._opt_artist    = None
        self._opt_lbl       = None
        self._pt            = None
        self._pt_lbl        = None

        self._ax = self.fig.add_axes([0.05, 0.05, 0.90, 0.90])
        self._setup_axes()
        self._draw_grid()
        self.redraw_circles()
        self._draw_dynamic()

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    # ── axes ─────────────────────────────────────────────────────────────────

    def _setup_axes(self):
        ax = self._ax
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.tick_params(colors=AXIS_COL, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(AXIS_COL)
        plane_lbl = "ΓS" if self.plane == "source" else "ΓL"
        ax.set_xlabel(f"Re({plane_lbl})", color=LBL_COL, fontsize=8)
        ax.set_ylabel(f"Im({plane_lbl})", color=LBL_COL, fontsize=8)
        title = ("Source Plane  (ΓS)" if self.plane == "source"
                 else "Load Plane  (ΓL)")
        ax.set_title(title, color=TXT_COL, fontsize=9, pad=3)

    # ── static Smith Chart grid ───────────────────────────────────────────────

    def _draw_grid(self):
        ax = self._ax
        for a in self._grid_artists:
            try: a.remove()
            except Exception: pass
        self._grid_artists.clear()

        theta = np.linspace(0, 2 * np.pi, 500)
        (ln,) = ax.plot(np.cos(theta), np.sin(theta),
                        color=RIM_COL, lw=1.8, zorder=3)
        self._grid_artists.append(ln)

        (ln,) = ax.plot([-1, 1], [0, 0], color=AXIS_COL, lw=0.7, zorder=2)
        self._grid_artists.append(ln)

        for r in R_VALUES:
            re, im = resistance_circle_pts(r)
            (ln,) = ax.plot(re, im, color=GRID_MAJ, lw=0.75, alpha=0.8, zorder=2)
            self._grid_artists.append(ln)
            if len(re) > 0:
                idx = np.argmin(re)
                lbl = ax.text(re[idx] - 0.02, 0.03, f"{r}",
                              fontsize=5.5, color=LBL_COL, zorder=5, ha="right")
                self._grid_artists.append(lbl)

        for x in X_VALUES:
            for sign in (+1, -1):
                re, im = reactance_arc_pts(sign * x)
                if len(re) < 2:
                    continue
                (ln,) = ax.plot(re, im, color=GRID_MIN, lw=0.65, alpha=0.7, zorder=2)
                self._grid_artists.append(ln)

        self.draw_idle()

    # ── gain circles + stability circle (static per S-param) ─────────────────

    def redraw_circles(self):
        """Call when S-params or options change."""
        ax = self._ax
        for a in self._circle_artists:
            try: a.remove()
            except Exception: pass
        self._circle_artists.clear()

        for a in (self._stab_artist, self._opt_artist, self._opt_lbl):
            if a is not None:
                try: a.remove()
                except Exception: pass
        self._stab_artist = self._opt_artist = self._opt_lbl = None

        st = self.state
        n  = st.gain_steps

        # Determine which circles to draw
        if self.plane == "source":
            uni_circles  = st.unilateral_source_circles(n) if st.show_unilateral else []
            bil_circles  = st.bilateral_GA_circles(n)       if st.show_bilateral  else []
            stab_circ    = st.source_stability_circle()      if st.show_stability  else None
            opt_uni      = st.gamma_S_opt_uni
            opt_bil      = st.gamma_S_opt
            pt_color     = GS_COL
        else:
            uni_circles  = st.unilateral_load_circles(n)   if st.show_unilateral else []
            bil_circles  = st.bilateral_GP_circles(n)       if st.show_bilateral  else []
            stab_circ    = st.load_stability_circle()       if st.show_stability  else None
            opt_uni      = st.gamma_L_opt_uni
            opt_bil      = st.gamma_L_opt
            pt_color     = GL_COL

        # Unilateral circles (solid, blue→red gradient)
        all_uni = uni_circles
        for i, (cx, cy, r, g_lin) in enumerate(all_uni):
            t   = i / max(len(all_uni) - 1, 1)
            col = _lerp_color(t, GAIN_LOW, GAIN_HIGH)
            re, im = circle_pts(cx, cy, r)
            (ln,) = ax.plot(re, im, color=col, lw=1.1, alpha=0.75,
                            linestyle="-", zorder=4)
            self._circle_artists.append(ln)
            g_db = pwr_db(g_lin)
            lbl = ax.text(cx + r * 0.7, cy + r * 0.7,
                          f"{g_db:.1f}dB",
                          fontsize=5, color=col, alpha=0.9, zorder=5)
            self._circle_artists.append(lbl)

        # Bilateral circles (dashed, cyan)
        for i, (cx, cy, r, g_lin) in enumerate(bil_circles):
            t = i / max(len(bil_circles) - 1, 1)
            col = _lerp_color(t, "#006688", "#44ffcc")
            re, im = circle_pts(cx, cy, r)
            (ln,) = ax.plot(re, im, color=col, lw=1.0, alpha=0.65,
                            linestyle="--", zorder=4)
            self._circle_artists.append(ln)
            g_db = pwr_db(g_lin)
            lbl = ax.text(cx, cy + r + 0.05, f"{g_db:.1f}",
                          fontsize=4.5, color=col, alpha=0.8, zorder=5)
            self._circle_artists.append(lbl)

        # Stability circle (dashed red)
        if stab_circ is not None:
            cx, cy, r = stab_circ
            re, im = circle_pts(cx, cy, r)
            (ln,) = ax.plot(re, im, color=STAB_COL, lw=1.4,
                            linestyle=":", alpha=0.85, zorder=6)
            self._stab_artist = ln
            lbl = ax.text(cx, cy, "Stab", fontsize=5.5,
                          color=STAB_COL, ha="center", va="center",
                          alpha=0.7, zorder=6)
            self._circle_artists.append(lbl)

        # VSWR reference circles (concentric, centred at origin)
        if st.show_vswr:
            _VSWR_LEVELS = [
                (1.5,  "#555577"),
                (2.0,  "#666688"),
                (3.0,  "#777799"),
                (5.0,  "#8888aa"),
            ]
            theta = np.linspace(0, 2 * np.pi, 300)
            for vswr, col in _VSWR_LEVELS:
                r_vswr = (vswr - 1.0) / (vswr + 1.0)
                (ln,) = ax.plot(r_vswr * np.cos(theta), r_vswr * np.sin(theta),
                                color=col, lw=0.8, linestyle=":", alpha=0.7, zorder=3)
                self._circle_artists.append(ln)
                lbl = ax.text(0, r_vswr + 0.03, f"VSWR {vswr:.0f}",
                              ha="center", va="bottom", fontsize=4.5,
                              color=col, alpha=0.75, zorder=3)
                self._circle_artists.append(lbl)

        # Optimal points: unilateral (star) and bilateral (diamond)
        if abs(opt_uni) <= 1.05:
            (self._opt_artist,) = ax.plot(opt_uni.real, opt_uni.imag, "*",
                                           color=OPT_COL, ms=10,
                                           markeredgecolor="white", mew=0.5,
                                           zorder=9, label="Opt (uni)")
        if abs(opt_bil) <= 1.05:
            (ln,) = ax.plot(opt_bil.real, opt_bil.imag, "D",
                             color="#aaffaa", ms=7,
                             markeredgecolor="white", mew=0.5,
                             zorder=9, label="Opt (bil)")
            self._circle_artists.append(ln)

        self.draw_idle()

    # ── dynamic layer (moved only during drag) ────────────────────────────────

    def _draw_dynamic(self):
        ax = self._ax
        for a in (self._pt, self._pt_lbl):
            if a is not None:
                try: a.remove()
                except Exception: pass
        self._pt = self._pt_lbl = None

        gamma = self.state.gamma_S if self.plane == "source" else self.state.gamma_L
        col   = GS_COL if self.plane == "source" else GL_COL
        lbl_s = "ΓS" if self.plane == "source" else "ΓL"

        (self._pt,) = ax.plot(gamma.real, gamma.imag, "o",
                               color=col, ms=13,
                               markeredgecolor="white", mew=1.5, zorder=10)
        self._pt_lbl = ax.text(gamma.real + 0.07, gamma.imag + 0.07,
                                f"{lbl_s}\n{abs(gamma):.3f}∠{np.degrees(np.angle(gamma)):.0f}°",
                                color=col, fontsize=7, zorder=11, fontweight="bold")
        self.draw_idle()

    def _update_fast(self):
        gamma = self.state.gamma_S if self.plane == "source" else self.state.gamma_L
        lbl_s = "ΓS" if self.plane == "source" else "ΓL"
        self._pt.set_data([gamma.real], [gamma.imag])
        if self._pt_lbl is not None:
            self._pt_lbl.set_position((gamma.real + 0.07, gamma.imag + 0.07))
            self._pt_lbl.set_text(
                f"{lbl_s}\n{abs(gamma):.3f}∠{np.degrees(np.angle(gamma)):.0f}°")
        self.draw_idle()

    # ── drag interaction ─────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes != self._ax or event.button != 1:
            return
        gamma = self.state.gamma_S if self.plane == "source" else self.state.gamma_L
        dist  = np.hypot(event.xdata - gamma.real, event.ydata - gamma.imag)
        if dist < DRAG_THR:
            self._dragging = True

    def _on_motion(self, event):
        if not self._dragging or event.inaxes != self._ax:
            return
        g = complex(event.xdata, event.ydata)
        if abs(g) > 0.999:
            g = g / abs(g) * 0.999
        if self.plane == "source":
            self.state.gamma_S = g
        else:
            self.state.gamma_L = g
        self._update_fast()
        self.stateChanged.emit(self.state)

    def _on_release(self, event):
        self._dragging = False

    def full_redraw(self):
        self._draw_grid()
        self.redraw_circles()
        self._draw_dynamic()


# ─────────────────────────────────────────────────────────────────────────────
# S-Parameter Control Panel
# ─────────────────────────────────────────────────────────────────────────────

class SParamPanel(QWidget):
    """Four S-parameter inputs (magnitude dB + phase °) plus S21 linear readout."""

    sparamChanged = pyqtSignal(object)   # emits GainState

    _PARAMS = [
        ("S₁₁", "s11", GS_COL),
        ("S₁₂", "s12", "#ff8844"),
        ("S₂₁", "s21", "#44ffcc"),
        ("S₂₂", "s22", GL_COL),
    ]

    def __init__(self, state: GainState, parent=None):
        super().__init__(parent)
        self.state = state
        self._block = False

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        grp = QGroupBox("S-Parameters  (magnitude dB + phase °)")
        form = QFormLayout(grp)
        form.setSpacing(3)
        form.setContentsMargins(6, 10, 6, 6)

        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)

        self._spins = {}   # key → (mag_spin, ph_spin)

        for label, key, col in self._PARAMS:
            s = getattr(state, key)
            mag_db  = lin_to_db(abs(s))
            phase_d = np.degrees(np.angle(s))

            mag_spin = QDoubleSpinBox()
            mag_spin.setRange(-60.0, 40.0)
            mag_spin.setDecimals(2)
            mag_spin.setSuffix(" dB")
            mag_spin.setValue(mag_db)
            mag_spin.setFixedWidth(90)

            ph_spin = QDoubleSpinBox()
            ph_spin.setRange(-180.0, 180.0)
            ph_spin.setDecimals(1)
            ph_spin.setSuffix(" °")
            ph_spin.setValue(phase_d)
            ph_spin.setFixedWidth(80)

            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(4)
            row_h.addWidget(mag_spin)
            row_h.addWidget(ph_spin)

            lbl_w = QLabel(label)
            lbl_w.setStyleSheet(f"color:{col};font-weight:bold;font-size:11px;")
            form.addRow(lbl_w, row_w)

            mag_spin.valueChanged.connect(self._on_change)
            ph_spin.valueChanged.connect(self._on_change)
            self._spins[key] = (mag_spin, ph_spin)

        layout.addWidget(grp)

        # S21 linear readout
        self._s21_lbl = QLabel("")
        self._s21_lbl.setFont(mono)
        self._s21_lbl.setStyleSheet(f"color:#44ffcc;font-size:10px;padding-left:6px;")
        layout.addWidget(self._s21_lbl)

        self._refresh_s21()

    def _on_change(self):
        if self._block:
            return
        for key, (mag_spin, ph_spin) in self._spins.items():
            mag = db_to_lin(mag_spin.value())
            ph  = np.radians(ph_spin.value())
            setattr(self.state, key, complex(mag * np.cos(ph), mag * np.sin(ph)))
        self._refresh_s21()
        self.sparamChanged.emit(self.state)

    def _refresh_s21(self):
        s21 = self.state.s21
        self._s21_lbl.setText(
            f"|S₂₁| = {abs(s21):.3f}  ({pwr_db(abs(s21)**2):.2f} dB power)")

    def sync_from_state(self):
        self._block = True
        for key, (mag_spin, ph_spin) in self._spins.items():
            s = getattr(self.state, key)
            mag_spin.setValue(lin_to_db(abs(s)))
            ph_spin.setValue(np.degrees(np.angle(s)))
        self._block = False
        self._refresh_s21()


# ─────────────────────────────────────────────────────────────────────────────
# Source / Load / Z0 Panel
# ─────────────────────────────────────────────────────────────────────────────

class SourceLoadPanel(QWidget):
    """ZS, ZL, and Z0 spinboxes with bidirectional sync to canvas drag."""

    z0Changed  = pyqtSignal(float)
    zsChanged  = pyqtSignal(float, float)   # Re, Im of ZS
    zlChanged  = pyqtSignal(float, float)

    def __init__(self, state: GainState, parent=None):
        super().__init__(parent)
        self.state = state
        self._block = False

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)

        def _spin(lo, hi, val, suf=" Ω", dec=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSuffix(suf)
            s.setValue(val)
            return s

        # ── Z0 ───────────────────────────────────────────────────────────────
        z0_grp = QGroupBox("Reference  Z₀")
        z0_form = QFormLayout(z0_grp)
        z0_form.setSpacing(3)
        self.z0_spin = _spin(1, 10000, state.z0)
        self.z0_spin.valueChanged.connect(self.z0Changed.emit)
        z0_form.addRow("Z₀ =", self.z0_spin)
        layout.addWidget(z0_grp)

        # ── ZS ───────────────────────────────────────────────────────────────
        zs = gamma_to_z(state.gamma_S, state.z0)
        zs_grp = QGroupBox("Source Impedance  ZS")
        zs_grp.setStyleSheet(f"QGroupBox{{border-color:{GS_COL};color:{GS_COL};}}")
        zs_form = QFormLayout(zs_grp)
        zs_form.setSpacing(3)
        self.zs_re = _spin(-9999, 9999, zs.real)
        self.zs_im = _spin(-9999, 9999, zs.imag)
        self.gs_lbl = QLabel("0.000 ∠ 0.0°")
        self.gs_lbl.setFont(mono)
        self.gs_lbl.setStyleSheet(f"color:{GS_COL};font-size:10px;")
        zs_form.addRow("Re(ZS):", self.zs_re)
        zs_form.addRow("Im(ZS):", self.zs_im)
        zs_form.addRow("ΓS:",     self.gs_lbl)
        self.zs_re.valueChanged.connect(self._on_zs)
        self.zs_im.valueChanged.connect(self._on_zs)
        layout.addWidget(zs_grp)

        # ── ZL ───────────────────────────────────────────────────────────────
        zl = gamma_to_z(state.gamma_L, state.z0)
        zl_grp = QGroupBox("Load Impedance  ZL")
        zl_grp.setStyleSheet(f"QGroupBox{{border-color:{GL_COL};color:{GL_COL};}}")
        zl_form = QFormLayout(zl_grp)
        zl_form.setSpacing(3)
        self.zl_re = _spin(-9999, 9999, zl.real)
        self.zl_im = _spin(-9999, 9999, zl.imag)
        self.gl_lbl = QLabel("0.000 ∠ 0.0°")
        self.gl_lbl.setFont(mono)
        self.gl_lbl.setStyleSheet(f"color:{GL_COL};font-size:10px;")
        zl_form.addRow("Re(ZL):", self.zl_re)
        zl_form.addRow("Im(ZL):", self.zl_im)
        zl_form.addRow("ΓL:",     self.gl_lbl)
        self.zl_re.valueChanged.connect(self._on_zl)
        self.zl_im.valueChanged.connect(self._on_zl)
        layout.addWidget(zl_grp)

        self._update_labels()

    def _on_zs(self):
        if not self._block:
            self.zsChanged.emit(self.zs_re.value(), self.zs_im.value())

    def _on_zl(self):
        if not self._block:
            self.zlChanged.emit(self.zl_re.value(), self.zl_im.value())

    def _update_labels(self):
        gS = self.state.gamma_S
        gL = self.state.gamma_L
        self.gs_lbl.setText(f"{abs(gS):.3f} ∠ {np.degrees(np.angle(gS)):.1f}°")
        self.gl_lbl.setText(f"{abs(gL):.3f} ∠ {np.degrees(np.angle(gL)):.1f}°")

    def sync_from_gamma_S(self):
        self._block = True
        zs = gamma_to_z(self.state.gamma_S, self.state.z0)
        self.zs_re.setValue(zs.real)
        self.zs_im.setValue(zs.imag)
        self._block = False
        self._update_labels()

    def sync_from_gamma_L(self):
        self._block = True
        zl = gamma_to_z(self.state.gamma_L, self.state.z0)
        self.zl_re.setValue(zl.real)
        self.zl_im.setValue(zl.imag)
        self._block = False
        self._update_labels()


# ─────────────────────────────────────────────────────────────────────────────
# Display Options + Presets Panel
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "Custom": None,
    "LNA (HP-AT41511)": {
        "s11": (0.6,  -170.0), "s12": (0.05,  16.0),
        "s21": (2.5,    30.0), "s22": (0.5,  -95.0),
    },
    "Power Amp": {
        "s11": (0.7,   -60.0), "s12": (0.10,  45.0),
        "s21": (3.0,   120.0), "s22": (0.4,  -40.0),
    },
    "Wideband Amp": {
        "s11": (0.30,  -90.0), "s12": (0.02,  10.0),
        "s21": (1.80,   90.0), "s22": (0.30, -80.0),
    },
    "Bilateral k<1": {
        "s11": (0.50,  -60.0), "s12": (0.40,  60.0),
        "s21": (1.50,   90.0), "s22": (0.50, -60.0),
    },
    "Unilateral (S12=0)": {
        "s11": (0.40,  -90.0), "s12": (0.00,   0.0),
        "s21": (2.00,   80.0), "s22": (0.30, -70.0),
    },
}


class OptionsPanel(QWidget):
    optionChanged  = pyqtSignal(str, object)
    presetSelected = pyqtSignal(str)

    def __init__(self, state: GainState, parent=None):
        super().__init__(parent)
        self.state = state

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        disp_grp = QGroupBox("Display")
        disp_h = QHBoxLayout(disp_grp)
        disp_h.setSpacing(6)
        self.cb_uni  = QCheckBox("Unilateral")
        self.cb_uni.setChecked(state.show_unilateral)
        self.cb_bil  = QCheckBox("Bilateral")
        self.cb_bil.setChecked(state.show_bilateral)
        self.cb_stab = QCheckBox("Stability")
        self.cb_stab.setChecked(state.show_stability)
        self.cb_vswr = QCheckBox("VSWR")
        self.cb_vswr.setChecked(state.show_vswr)
        for cb, key in ((self.cb_uni,  "show_unilateral"),
                        (self.cb_bil,  "show_bilateral"),
                        (self.cb_stab, "show_stability"),
                        (self.cb_vswr, "show_vswr")):
            cb.toggled.connect(lambda v, k=key: self.optionChanged.emit(k, v))
            disp_h.addWidget(cb)
        layout.addWidget(disp_grp)

        preset_grp = QGroupBox("Presets")
        preset_h = QHBoxLayout(preset_grp)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.presetSelected.emit)
        preset_h.addWidget(self.preset_combo)
        layout.addWidget(preset_grp)


# ─────────────────────────────────────────────────────────────────────────────
# Gain Metrics Panel (bottom bar)
# ─────────────────────────────────────────────────────────────────────────────

class GainMetricsPanel(QWidget):
    def __init__(self, state: GainState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(130)

        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)

        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.setContentsMargins(6, 4, 6, 4)

        # ── Row 1: big gain numbers ───────────────────────────────────────────
        nums_w = QWidget()
        nums_h = QHBoxLayout(nums_w)
        nums_h.setSpacing(2)

        def _big(label, col):
            w = QGroupBox()
            w.setStyleSheet(
                f"QGroupBox{{border:1px solid {col};border-radius:6px;"
                f"background:{PANEL_BG};}}")
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 6, 8, 4)
            lay.setSpacing(1)
            title = QLabel(label)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(f"color:{col};font-weight:bold;font-size:10px;"
                                "border:none;background:transparent;")
            val = QLabel("—")
            val.setAlignment(Qt.AlignCenter)
            val.setFont(mono)
            val.setStyleSheet(f"color:{col};font-size:14px;font-weight:bold;"
                              "border:none;background:transparent;")
            lay.addWidget(title)
            lay.addWidget(val)
            return w, val

        self._gt_box,  self._gt_lbl  = _big("GT  (Transducer)",  "#ff8833")
        self._gp_box,  self._gp_lbl  = _big("GP  (Operating)",   "#44ddff")
        self._ga_box,  self._ga_lbl  = _big("GA  (Available)",   "#44ff88")
        self._mag_box, self._mag_lbl = _big("MAG",               "#ffee44")
        self._msg_box, self._msg_lbl = _big("MSG",               "#aaaacc")
        self._gtu_box, self._gtu_lbl = _big("GT  (Unilateral)",  "#cc88ff")

        for box in (self._gt_box, self._gp_box, self._ga_box,
                    self._mag_box, self._msg_box, self._gtu_box):
            nums_h.addWidget(box, stretch=1)

        root.addWidget(nums_w, stretch=2)

        # ── Row 2: stability + derived Γ values ───────────────────────────────
        det_w = QWidget()
        det_h = QHBoxLayout(det_w)
        det_h.setSpacing(12)
        det_h.setContentsMargins(4, 0, 4, 0)

        def _row_lbl(text):
            l = QLabel(text)
            l.setFont(mono)
            l.setStyleSheet(f"color:{TXT_COL};font-size:10px;")
            return l

        self._k_lbl      = _row_lbl("k = —")
        self._delta_lbl  = _row_lbl("|Δ| = —")
        self._stab_badge = QLabel("—")
        self._stab_badge.setFont(mono)
        self._stab_badge.setStyleSheet("font-size:10px;font-weight:bold;padding:2px 6px;"
                                       "border-radius:4px;")
        self._gin_lbl    = _row_lbl("Γin = —")
        self._gout_lbl   = _row_lbl("Γout = —")
        self._gsopt_lbl  = _row_lbl("ΓS_opt = —")
        self._glopt_lbl  = _row_lbl("ΓL_opt = —")

        for w in (self._k_lbl, self._delta_lbl, self._stab_badge,
                  self._gin_lbl, self._gout_lbl,
                  self._gsopt_lbl, self._glopt_lbl):
            det_h.addWidget(w)
        det_h.addStretch(1)

        root.addWidget(det_w, stretch=1)

        self.update_metrics(state)

    def update_metrics(self, state: GainState):
        self.state = state

        def _db(x):
            if np.isnan(x) or np.isinf(x):
                return "∞ dB" if x > 0 else "—"
            return f"{pwr_db(x):.2f} dB"

        self._gt_lbl.setText(_db(state.GT))
        self._gp_lbl.setText(_db(state.GP))
        self._ga_lbl.setText(_db(state.GA))

        mag = state.MAG
        self._mag_lbl.setText(_db(mag) if not np.isnan(mag) else "k < 1")
        self._msg_lbl.setText(_db(state.MSG))
        self._gtu_lbl.setText(_db(state.GT_unilateral))

        k   = state.k
        dlt = abs(state.delta)
        self._k_lbl.setText(f"k = {k:.3f}")
        self._delta_lbl.setText(f"|Δ| = {dlt:.3f}")

        if state.unconditionally_stable:
            self._stab_badge.setText("Unconditionally Stable")
            self._stab_badge.setStyleSheet(
                "color:#000000;background:#44bb44;font-size:10px;"
                "font-weight:bold;padding:2px 6px;border-radius:4px;")
        else:
            self._stab_badge.setText("Potentially Unstable")
            self._stab_badge.setStyleSheet(
                "color:#ffffff;background:#bb2222;font-size:10px;"
                "font-weight:bold;padding:2px 6px;border-radius:4px;")

        def _gfmt(g: complex) -> str:
            return f"{abs(g):.3f} ∠ {np.degrees(np.angle(g)):.0f}°"

        self._gin_lbl.setText(f"Γin = {_gfmt(state.gamma_in)}")
        self._gout_lbl.setText(f"Γout = {_gfmt(state.gamma_out)}")
        self._gsopt_lbl.setText(f"ΓS_opt = {_gfmt(state.gamma_S_opt)}")
        self._glopt_lbl.setText(f"ΓL_opt = {_gfmt(state.gamma_L_opt)}")


# ─────────────────────────────────────────────────────────────────────────────
# Wave Canvas — animated time-domain travelling-wave diagram
# ─────────────────────────────────────────────────────────────────────────────

class GainWaveCanvas(FigureCanvas):
    """
    Schematic: Source | ←Left TL→ | DUT | ←Right TL→ | Load
    Six animated wave lines: incident, reflected, total on each TL.
    """

    _TIMER_MS = 33

    def __init__(self, state: GainState, parent=None):
        self.fig = Figure(figsize=(10, 2.8), facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.state = state
        self._phase = 0.0

        self._ax = self.fig.add_axes([0.035, 0.14, 0.955, 0.74])
        self._setup_axes()
        self._draw_static()
        self._init_wave_lines()

        self._timer = QTimer(self)
        self._timer.setInterval(self._TIMER_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── axes ─────────────────────────────────────────────────────────────────

    def _setup_axes(self):
        ax = self._ax
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(-2.22, 2.22)
        ax.set_ylim(-2.8, 3.1)
        ax.set_xlabel("Position  (schematic, each TL = 2λ)", color=LBL_COL, fontsize=7)
        ax.tick_params(colors=AXIS_COL, labelsize=6)
        for sp in ax.spines.values():
            sp.set_color(AXIS_COL)
        ax.set_title("Time-Domain Travelling Waves — Source · · · DUT · · · Load",
                     color=TXT_COL, fontsize=8, pad=2)
        ax.axhline(0, color=AXIS_COL, lw=0.5, zorder=1)
        # suppress x-ticks (physical units are schematic only)
        ax.set_xticks([])

    # ── static schematic elements ─────────────────────────────────────────────

    def _draw_static(self):
        ax = self._ax

        # ── Source block ──────────────────────────────────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (_SRC_XC - _SRC_HW, -2.5), 2 * _SRC_HW, 5.0,
            boxstyle="round,pad=0.01",
            facecolor="#1e1e40", edgecolor=GS_COL, lw=1.8, zorder=5))
        ax.text(_SRC_XC, 2.8, "Source", ha="center", va="top",
                color=GS_COL, fontsize=6.5, fontweight="bold", zorder=6)
        ax.text(_SRC_XC, 2.5, "ZS  /  ΓS", ha="center", va="top",
                color=GS_COL, fontsize=5.5, zorder=6)

        # Dynamic annotations (updated each tick)
        self._ann_gs  = ax.text(
            _SRC_XC + _SRC_HW + 0.06, 2.4, self._gs_str(),
            ha="left", va="top", color=GS_COL, fontsize=5.5, zorder=6)
        self._ann_gin = ax.text(
            _SRC_XC + _SRC_HW + 0.06, 1.8, self._gin_str(),
            ha="left", va="top", color="#ff8844", fontsize=5.5, zorder=6)

        # ── DUT box ───────────────────────────────────────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (_DUT_L, -2.1), _DUT_R - _DUT_L, 4.2,
            boxstyle="round,pad=0.02",
            facecolor="#1a2038", edgecolor="#5566aa", lw=1.8, zorder=5))
        ax.text(0, 0.6, "DUT", ha="center", va="center",
                color="#8899cc", fontsize=8, fontweight="bold", zorder=6)
        ax.text(0, 0.0, "2-Port", ha="center", va="center",
                color="#6677aa", fontsize=6.5, zorder=6)
        ax.text(0, -0.7, "S₂₁ →", ha="center", va="center",
                color="#44ffcc", fontsize=6, alpha=0.8, zorder=6)
        ax.text(0, -1.2, "← S₁₂", ha="center", va="center",
                color="#ff8844", fontsize=6, alpha=0.8, zorder=6)
        # Port labels
        ax.text(_DUT_L + 0.01, -2.3, "Port 1", color="#5566aa",
                fontsize=5.5, ha="left", va="top", zorder=6)
        ax.text(_DUT_R - 0.01, -2.3, "Port 2", color="#5566aa",
                fontsize=5.5, ha="right", va="top", zorder=6)

        # ── Load block ────────────────────────────────────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (_LOAD_XC - _LOAD_HW, -2.5), 2 * _LOAD_HW, 5.0,
            boxstyle="round,pad=0.01",
            facecolor="#1e1e40", edgecolor=GL_COL, lw=1.8, zorder=5))
        ax.text(_LOAD_XC, 2.8, "Load", ha="center", va="top",
                color=GL_COL, fontsize=6.5, fontweight="bold", zorder=6)
        ax.text(_LOAD_XC, 2.5, "ZL  /  ΓL", ha="center", va="top",
                color=GL_COL, fontsize=5.5, zorder=6)
        self._ann_gl = ax.text(
            _LOAD_XC - _LOAD_HW - 0.06, 2.4, self._gl_str(),
            ha="right", va="top", color=GL_COL, fontsize=5.5, zorder=6)

        # ── Wire centre-lines (y=0) ───────────────────────────────────────────
        for x0, x1 in ((_SRC_XC + _SRC_HW,  _DUT_L),
                        (_DUT_R, _LOAD_XC - _LOAD_HW)):
            ax.plot([x0, x1], [0, 0], color=AXIS_COL,
                    lw=0.7, ls=":", zorder=2, alpha=0.5)

        # ── Section span labels ───────────────────────────────────────────────
        mid_l = (_SRC_XC + _SRC_HW + _DUT_L) / 2.0
        mid_r = (_DUT_R + _LOAD_XC - _LOAD_HW) / 2.0
        ax.text(mid_l, -2.7, f"Left TL  ({_WL_LEFT}λ)",
                ha="center", va="bottom", color=LBL_COL, fontsize=6)
        ax.text(mid_r, -2.7, f"Right TL  ({_WL_RIGHT}λ)",
                ha="center", va="bottom", color=LBL_COL, fontsize=6)

        # ── Direction arrows ──────────────────────────────────────────────────
        arrow_kw = dict(arrowstyle="-|>", lw=0.9)
        ax.annotate("", xy=(_DUT_L - 0.05, 2.1),
                    xytext=(_SRC_XC + _SRC_HW + 0.5, 2.1),
                    arrowprops=dict(color="#4488ff", **arrow_kw), zorder=4)
        ax.annotate("", xy=(_SRC_XC + _SRC_HW + 0.5, 1.65),
                    xytext=(_DUT_L - 0.05, 1.65),
                    arrowprops=dict(color="#ff4455", **arrow_kw), zorder=4)
        ax.annotate("", xy=(_LOAD_XC - _LOAD_HW - 0.05, 2.1),
                    xytext=(_DUT_R + 0.5, 2.1),
                    arrowprops=dict(color="#44ffcc", **arrow_kw), zorder=4)
        ax.annotate("", xy=(_DUT_R + 0.5, 1.65),
                    xytext=(_LOAD_XC - _LOAD_HW - 0.05, 1.65),
                    arrowprops=dict(color="#ffcc33", **arrow_kw), zorder=4)

        # ── Legend ────────────────────────────────────────────────────────────
        ax.legend(handles=[
            Line2D([0], [0], color="#4488ff", lw=1.2, label="Incident →"),
            Line2D([0], [0], color="#ff4455", lw=1.2, label="← Γin reflected"),
            Line2D([0], [0], color="#44ff88", lw=1.8, label="Total (left)"),
            Line2D([0], [0], color="#44ffcc", lw=1.2, label="S₂₁ forward →"),
            Line2D([0], [0], color="#ffcc33", lw=1.2, label="← ΓL reflected"),
            Line2D([0], [0], color="#ffff66", lw=1.8, label="Total (right)"),
        ], loc="lower right", fontsize=5.5, facecolor=PANEL_BG,
           labelcolor=TXT_COL, edgecolor=AXIS_COL,
           handlelength=1.2, borderpad=0.4, ncol=2, framealpha=0.85)

    # ── dynamic wave lines ────────────────────────────────────────────────────

    def _init_wave_lines(self):
        ax  = self._ax
        gin = self.state.gamma_in
        b2  = self._b2()
        gL  = self.state.gamma_L
        vi0, vr0, vt0 = _waves_left(gin, 0.0)
        vf0, vb0, vtr0 = _waves_right(b2, gL, 0.0)

        (self._li,)  = ax.plot(_x_l, vi0,  color="#4488ff", lw=1.2, alpha=0.85, zorder=7)
        (self._lr,)  = ax.plot(_x_l, vr0,  color="#ff4455", lw=1.2, alpha=0.85, zorder=7)
        (self._lt,)  = ax.plot(_x_l, vt0,  color="#44ff88", lw=1.8, zorder=8)
        (self._lf,)  = ax.plot(_x_r, vf0,  color="#44ffcc", lw=1.2, alpha=0.85, zorder=7)
        (self._lb,)  = ax.plot(_x_r, vb0,  color="#ffcc33", lw=1.2, alpha=0.85, zorder=7)
        (self._ltr,) = ax.plot(_x_r, vtr0, color="#ffff66", lw=1.8, zorder=8)

    def _b2(self) -> complex:
        st = self.state
        d  = 1.0 - st.s22 * st.gamma_L
        return st.s21 / d if abs(d) > 1e-10 else complex(0)

    # ── annotation helpers ────────────────────────────────────────────────────

    def _gs_str(self) -> str:
        g = self.state.gamma_S
        return f"ΓS = {abs(g):.3f}∠{np.degrees(np.angle(g)):.0f}°"

    def _gin_str(self) -> str:
        g = self.state.gamma_in
        return f"Γin = {abs(g):.3f}∠{np.degrees(np.angle(g)):.0f}°"

    def _gl_str(self) -> str:
        g = self.state.gamma_L
        return f"ΓL = {abs(g):.3f}∠{np.degrees(np.angle(g)):.0f}°"

    # ── animation tick ────────────────────────────────────────────────────────

    def _tick(self):
        self._phase = (self._phase + _PHASE_STEP) % (2 * np.pi)
        st  = self.state
        gin = st.gamma_in
        b2  = self._b2()
        gL  = st.gamma_L

        vi, vr, vt   = _waves_left(gin, self._phase)
        vf, vb, vtr  = _waves_right(b2, gL, self._phase)

        self._li.set_data(_x_l, vi)
        self._lr.set_data(_x_l, vr)
        self._lt.set_data(_x_l, vt)
        self._lf.set_data(_x_r, vf)
        self._lb.set_data(_x_r, vb)
        self._ltr.set_data(_x_r, vtr)
        self._ann_gs.set_text(self._gs_str())
        self._ann_gin.set_text(self._gin_str())
        self._ann_gl.set_text(self._gl_str())
        self.draw_idle()

    def update_waves(self, state: GainState):
        self.state = state


# ─────────────────────────────────────────────────────────────────────────────
# Dark stylesheet
# ─────────────────────────────────────────────────────────────────────────────

DARK_QSS = f"""
QWidget {{ background-color: {BG}; color: {TXT_COL}; font-family: "Segoe UI", sans-serif; font-size: 11px; }}
QMainWindow {{ background-color: {BG}; }}
QGroupBox {{
    border: 1px solid #333355;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    color: #9999cc;
    font-weight: bold;
    font-size: 11px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: #aaaadd; }}
QDoubleSpinBox, QComboBox {{
    background-color: #1e1e35;
    border: 1px solid #444466;
    border-radius: 4px;
    padding: 3px 6px;
    color: {TXT_COL};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ background: #2a2a4a; }}
QComboBox QAbstractItemView {{ background: #1e1e35; color: {TXT_COL}; selection-background-color: #2a3580; }}
QPushButton {{ background-color: #2a2a4a; border: 1px solid #555577; border-radius: 4px; padding: 4px 10px; color: #aaaadd; }}
QPushButton:hover  {{ background-color: #3a3a6a; }}
QPushButton:pressed {{ background-color: #222240; }}
QCheckBox {{ color: #aaaacc; spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid #555577; border-radius: 3px; background: #1e1e35; }}
QCheckBox::indicator:checked {{ background: #4466cc; }}
QSplitter::handle {{ background: #2a2a44; width: 3px; height: 3px; }}
QLabel {{ color: {TXT_COL}; }}
QScrollBar {{ background: #1a1a2e; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF Amplifier Gain Analysis")
        self.setMinimumSize(1280, 940)
        self.state = GainState()

        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(DARK_QSS)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Top row: ΓS chart | ΓL chart | right controls ─────────────────────
        top_split = QSplitter(Qt.Horizontal)

        self.canvas_s = GainCircleCanvas(self.state, plane="source")
        self.canvas_s.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top_split.addWidget(self.canvas_s)

        self.canvas_l = GainCircleCanvas(self.state, plane="load")
        self.canvas_l.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top_split.addWidget(self.canvas_l)

        # Right column: scrollable controls
        right_w = QWidget()
        right_w.setMinimumWidth(280)
        right_w.setMaximumWidth(360)
        right_lay = QVBoxLayout(right_w)
        right_lay.setSpacing(4)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.sparam_panel = SParamPanel(self.state)
        self.sl_panel     = SourceLoadPanel(self.state)
        self.opt_panel    = OptionsPanel(self.state)

        right_lay.addWidget(self.sparam_panel)
        right_lay.addWidget(self.sl_panel)
        right_lay.addWidget(self.opt_panel)
        right_lay.addStretch(1)

        top_split.addWidget(right_w)
        top_split.setSizes([520, 520, 300])
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)
        top_split.setStretchFactor(2, 0)

        # ── Middle: metrics ────────────────────────────────────────────────────
        metrics_grp = QGroupBox("Power Gain Metrics")
        metrics_lay = QVBoxLayout(metrics_grp)
        metrics_lay.setContentsMargins(4, 4, 4, 4)
        self.metrics = GainMetricsPanel(self.state)
        metrics_lay.addWidget(self.metrics)

        # ── Bottom: wave analysis ──────────────────────────────────────────────
        wave_grp = QGroupBox("Wave Analysis")
        wave_lay = QVBoxLayout(wave_grp)
        wave_lay.setContentsMargins(4, 4, 4, 4)
        self.wave = GainWaveCanvas(self.state)
        wave_lay.addWidget(self.wave)

        root.addWidget(top_split,   stretch=4)
        root.addWidget(metrics_grp, stretch=1)
        root.addWidget(wave_grp,    stretch=2)

    # ── signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.canvas_s.stateChanged.connect(self._on_gamma_S_changed)
        self.canvas_l.stateChanged.connect(self._on_gamma_L_changed)

        self.sparam_panel.sparamChanged.connect(self._on_sparams_changed)
        self.sl_panel.z0Changed.connect(self._on_z0_changed)
        self.sl_panel.zsChanged.connect(self._on_zs_changed)
        self.sl_panel.zlChanged.connect(self._on_zl_changed)

        self.opt_panel.optionChanged.connect(self._on_option_changed)
        self.opt_panel.presetSelected.connect(self._on_preset)

    # ── handlers ──────────────────────────────────────────────────────────────

    def _refresh_metrics(self):
        self.metrics.update_metrics(self.state)
        self.wave.update_waves(self.state)

    def _on_gamma_S_changed(self, state: GainState):
        """Called during ΓS drag."""
        self.sl_panel.sync_from_gamma_S()
        self._refresh_metrics()

    def _on_gamma_L_changed(self, state: GainState):
        """Called during ΓL drag."""
        self.sl_panel.sync_from_gamma_L()
        self._refresh_metrics()

    def _on_sparams_changed(self, state: GainState):
        """S-params changed → redraw gain circles on both canvases."""
        self.canvas_s.redraw_circles()
        self.canvas_l.redraw_circles()
        self._refresh_metrics()

    def _on_z0_changed(self, z0: float):
        self.state.z0 = z0
        self.canvas_s.full_redraw()
        self.canvas_l.full_redraw()
        self._refresh_metrics()

    def _on_zs_changed(self, re: float, im: float):
        self.state.gamma_S = z_to_gamma(complex(re, im), self.state.z0)
        self.canvas_s._update_fast()
        self.sl_panel._update_labels()
        self._refresh_metrics()

    def _on_zl_changed(self, re: float, im: float):
        self.state.gamma_L = z_to_gamma(complex(re, im), self.state.z0)
        self.canvas_l._update_fast()
        self.sl_panel._update_labels()
        self._refresh_metrics()

    def _on_option_changed(self, key: str, value):
        setattr(self.state, key, value)
        self.canvas_s.redraw_circles()
        self.canvas_l.redraw_circles()

    def _on_preset(self, name: str):
        data = PRESETS.get(name)
        if data is None:
            return
        for key, (mag_lin, phase_deg) in data.items():
            ph = np.radians(phase_deg)
            setattr(self.state, key,
                    complex(mag_lin * np.cos(ph), mag_lin * np.sin(ph)))
        self.sparam_panel.sync_from_state()
        self.canvas_s.redraw_circles()
        self.canvas_l.redraw_circles()
        self._refresh_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setApplicationName("RF Gain Analysis")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
