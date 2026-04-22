"""
Interactive Smith Chart Simulator
Drag S11 (red) and S22 (blue) points to explore impedances.
"""

import sys
import numpy as np
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QLabel, QDoubleSpinBox, QCheckBox,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QFormLayout, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

# ─────────────────────────────────────────────────────────────────────────────
# RF Math
# ─────────────────────────────────────────────────────────────────────────────

def gamma_to_z(gamma: complex, z0: float = 50.0) -> complex:
    denom = 1.0 - gamma
    if abs(denom) < 1e-10:
        return complex(1e9, 0)
    return z0 * (1.0 + gamma) / denom


def z_to_gamma(z: complex, z0: float = 50.0) -> complex:
    zn = z / z0
    denom = zn + 1.0
    if abs(denom) < 1e-10:
        return complex(-1, 0)
    return (zn - 1.0) / denom


def vswr(gamma: complex) -> float:
    mag = min(abs(gamma), 0.9999)
    return (1.0 + mag) / (1.0 - mag)


def return_loss_db(gamma: complex) -> float:
    mag = abs(gamma)
    if mag < 1e-12:
        return float("inf")
    return -20.0 * np.log10(mag)


def resistance_circle_pts(r: float, n: int = 300):
    """(re, im) arrays for constant-r circle, clipped inside |Γ|≤1."""
    cx = r / (r + 1.0)
    rad = 1.0 / (r + 1.0)
    theta = np.linspace(0, 2 * np.pi, n)
    re = cx + rad * np.cos(theta)
    im = rad * np.sin(theta)
    mask = re ** 2 + im ** 2 <= 1.0 + 1e-6
    return re[mask], im[mask]


def reactance_arc_pts(x: float, n: int = 400):
    """(re, im) arrays for constant-x arc, clipped inside |Γ|≤1."""
    if abs(x) < 1e-9:
        return np.array([]), np.array([])
    cy = 1.0 / x
    rad = 1.0 / abs(x)
    theta = np.linspace(0, 2 * np.pi, n)
    re = 1.0 + rad * np.cos(theta)
    im = cy + rad * np.sin(theta)
    mask = re ** 2 + im ** 2 <= 1.0 + 1e-6
    pts_re, pts_im = re[mask], im[mask]
    if len(pts_re) < 2:
        return np.array([]), np.array([])
    # Sort by angle around arc centre so line doesn't cross itself
    angles = np.arctan2(pts_im - cy, pts_re - 1.0)
    order = np.argsort(angles)
    return pts_re[order], pts_im[order]


def voltage_standing_wave(gamma: complex, z0: float = 50.0,
                          phase: float = 0.0, n: int = 600):
    """
    Time-domain travelling waves on a transmission line.
      V_inc(z,t) = cos(ωt − βz)            → travels in +z direction
      V_ref(z,t) = |Γ|·cos(ωt + βz + ∠Γ)  → travels in −z direction
      V_tot      = V_inc + V_ref            → standing-wave envelope
    phase = ωt, advances each animation frame.
    """
    pos   = np.linspace(0, 2.0, n)        # 0 … 2λ  (λ = 1 unit)
    beta_z = 2 * np.pi * pos              # βz
    v_inc = np.cos(phase - beta_z)
    v_ref = np.real(gamma * np.exp(1j * (phase + beta_z)))
    v_tot = v_inc + v_ref
    return pos, v_inc, v_ref, v_tot


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class State:
    z0: float = 50.0
    gamma1: complex = complex(0, 0)      # S11 — DUT input reflection
    gamma2: complex = complex(0.3, 0.2)  # S22 — DUT output reflection
    zL_re: float = 50.0                  # load impedance, real part (Ω)
    zL_im: float = 0.0                   # load impedance, imaginary part (Ω)
    show_vswr: bool = True
    show_admittance: bool = False
    show_labels: bool = True

    @property
    def z1(self) -> complex:
        return gamma_to_z(self.gamma1, self.z0)

    @property
    def z2(self) -> complex:
        return gamma_to_z(self.gamma2, self.z0)

    @property
    def zL(self) -> complex:
        return complex(self.zL_re, self.zL_im)

    @property
    def gamma_L(self) -> complex:
        """Reflection coefficient of the termination load."""
        return z_to_gamma(self.zL, self.z0)

    @property
    def s_matrix(self) -> np.ndarray:
        s11 = self.gamma1
        s22 = self.gamma2
        s21_mag = np.sqrt(max(0.0, 1.0 - abs(s11) ** 2))
        s21_phase = (np.angle(s11) + np.angle(s22)) / 2 - np.pi / 2
        s21 = s21_mag * np.exp(1j * s21_phase)
        return np.array([[s11, s21], [s21, s22]])

    @property
    def b2(self) -> complex:
        """
        Forward wave amplitude at DUT port 2, accounting for load mismatch.
        b2 = S21 / (1 − S22·ΓL)  — includes multiple internal reflections.
        """
        S = self.s_matrix
        d = 1.0 - S[1, 1] * self.gamma_L
        return S[1, 0] / d if abs(d) > 1e-10 else complex(0)

    @property
    def gamma_in(self) -> complex:
        """
        Effective input reflection coefficient with load present.
        Γin = S11 + S12·S21·ΓL / (1 − S22·ΓL)
        """
        S = self.s_matrix
        d = 1.0 - S[1, 1] * self.gamma_L
        if abs(d) < 1e-10:
            return S[0, 0]
        return S[0, 0] + S[0, 1] * S[1, 0] * self.gamma_L / d


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

BG          = "#12121f"
PANEL_BG    = "#1a1a2e"
GRID_MAJOR  = "#2a3060"
GRID_MINOR  = "#1e2248"
AXIS_COLOR  = "#444466"
RIM_COLOR   = "#6677bb"
LABEL_COLOR = "#8899cc"
TEXT_COLOR  = "#ccccee"
PT1_COLOR   = "#ff4455"   # Port 1 — red
PT2_COLOR   = "#4488ff"   # Port 2 — blue
VSWR_COLOR  = "#ff8833"
ADM_COLOR   = "#44ffaa"
INC_COLOR   = "#4488ff"
REF_COLOR   = "#ff4455"
TOT_COLOR   = "#44ff88"
TRANS_COLOR = "#44ffcc"   # transmitted / forward wave (output side)
LOAD_COLOR  = "#ffcc33"   # load ΓL point and backward-from-load wave
LOAD_BWD    = "#ff8833"   # backward wave reflected from load

R_VALUES = [0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
X_VALUES = [0.2, 0.5, 1.0, 2.0, 5.0]

DRAG_THRESH = 0.08   # Γ-plane units

PRESETS = {
    "Custom":        None,
    "Matched (Z=Z₀)":  (complex(0, 0),      complex(0.3, 0.2)),
    "Short Circuit":    (complex(-0.999, 0), complex(-0.8, 0.1)),
    "Open Circuit":     (complex(0.999, 0),  complex(0.9,  0.1)),
    "Pure Inductive":   (complex(0, 0.5),    complex(0,    0.3)),
    "Pure Capacitive":  (complex(0, -0.5),   complex(0,   -0.3)),
}

# ─────────────────────────────────────────────────────────────────────────────
# Smith Chart Canvas
# ─────────────────────────────────────────────────────────────────────────────

class SmithChartCanvas(FigureCanvas):
    """Matplotlib canvas with draggable Γ points."""

    stateChanged = pyqtSignal(object)

    def __init__(self, state: State, parent=None):
        self.fig = Figure(figsize=(6, 6), facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.state = state

        self._dragging = None       # None | 'p1' | 'p2' | 'load'
        self._grid_artists = []     # Line2D objects for static grid
        self._vswr_line = None
        self._adm1 = None
        self._adm2 = None
        self._pt1 = None
        self._pt2 = None
        self._pt_load = None
        self._lbl1 = None
        self._lbl2 = None
        self._lbl_load = None

        self._ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])
        self._setup_axes()
        self._draw_grid()
        self._draw_dynamic()

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    # ── axes setup ────────────────────────────────────────────────────────────

    def _setup_axes(self):
        ax = self._ax
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal")
        ax.tick_params(colors=AXIS_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
        ax.set_xlabel("Re(Γ)", color=LABEL_COLOR, fontsize=8)
        ax.set_ylabel("Im(Γ)", color=LABEL_COLOR, fontsize=8)

    # ── static grid (drawn once) ───────────────────────────────────────────────

    def _draw_grid(self):
        ax = self._ax
        # Remove old grid artists
        for a in self._grid_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._grid_artists.clear()

        # Outer unit circle
        theta = np.linspace(0, 2 * np.pi, 500)
        (line,) = ax.plot(np.cos(theta), np.sin(theta),
                          color=RIM_COLOR, lw=1.8, zorder=3)
        self._grid_artists.append(line)

        # Real axis
        (line,) = ax.plot([-1, 1], [0, 0], color=AXIS_COLOR, lw=0.7, zorder=2)
        self._grid_artists.append(line)

        # Constant-r circles
        for r in R_VALUES:
            re, im = resistance_circle_pts(r)
            (line,) = ax.plot(re, im, color=GRID_MAJOR, lw=0.8, alpha=0.8, zorder=2)
            self._grid_artists.append(line)
            if self.state.show_labels and len(re) > 0:
                # Label at the leftmost point of the circle (where im≈0)
                idx = np.argmin(re)
                lbl = ax.text(re[idx] - 0.02, 0.03, f"{r}",
                              fontsize=6, color=LABEL_COLOR, zorder=5,
                              ha="right")
                self._grid_artists.append(lbl)

        # Constant-x arcs
        for x in X_VALUES:
            for sign in (+1, -1):
                xv = sign * x
                re, im = reactance_arc_pts(xv)
                if len(re) < 2:
                    continue
                (line,) = ax.plot(re, im, color=GRID_MINOR, lw=0.7,
                                  alpha=0.75, zorder=2)
                self._grid_artists.append(line)
                if self.state.show_labels:
                    idx = np.argmax(re ** 2 + im ** 2)
                    prefix = "+j" if xv > 0 else "−j"
                    lbl = ax.text(re[idx], im[idx],
                                  f"{prefix}{abs(xv):.1f}",
                                  fontsize=5.5, color=LABEL_COLOR,
                                  ha="center", va="center", zorder=5)
                    self._grid_artists.append(lbl)

        self.draw_idle()

    # ── dynamic layer ─────────────────────────────────────────────────────────

    def _draw_dynamic(self):
        """(Re-)create VSWR circle, admittance markers, and draggable points."""
        ax = self._ax

        # Remove old dynamic artists
        for attr in ("_vswr_line", "_adm1", "_adm2", "_pt1", "_pt2",
                     "_pt_load", "_lbl1", "_lbl2", "_lbl_load"):
            a = getattr(self, attr, None)
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
            setattr(self, attr, None)

        g1 = self.state.gamma1
        g2 = self.state.gamma2

        # VSWR circle
        if self.state.show_vswr:
            r = abs(g1)
            theta = np.linspace(0, 2 * np.pi, 300)
            (self._vswr_line,) = ax.plot(r * np.cos(theta), r * np.sin(theta),
                                         color=VSWR_COLOR, lw=1.0,
                                         linestyle="--", alpha=0.75, zorder=4)

        # Admittance mirror points
        if self.state.show_admittance:
            ag1 = -g1
            (self._adm1,) = ax.plot(ag1.real, ag1.imag, "s",
                                     color=ADM_COLOR, ms=8,
                                     markeredgecolor="white", mew=1.0, zorder=7)
            ag2 = -g2
            (self._adm2,) = ax.plot(ag2.real, ag2.imag, "s",
                                     color=ADM_COLOR, ms=8,
                                     markeredgecolor="white", mew=1.0, zorder=7)

        # Port markers (S11, S22)
        (self._pt1,) = ax.plot(g1.real, g1.imag, "o",
                                color=PT1_COLOR, ms=13,
                                markeredgecolor="white", mew=1.5, zorder=10)
        (self._pt2,) = ax.plot(g2.real, g2.imag, "o",
                                color=PT2_COLOR, ms=13,
                                markeredgecolor="white", mew=1.5, zorder=10)

        # Load marker (ΓL) — yellow diamond
        gL = self.state.gamma_L
        (self._pt_load,) = ax.plot(gL.real, gL.imag, "D",
                                    color=LOAD_COLOR, ms=11,
                                    markeredgecolor="white", mew=1.5, zorder=10)

        if self.state.show_labels:
            self._lbl1 = ax.text(g1.real + 0.06, g1.imag + 0.06, "S₁₁",
                                  color=PT1_COLOR, fontsize=9, zorder=11,
                                  fontweight="bold")
            self._lbl2 = ax.text(g2.real + 0.06, g2.imag + 0.06, "S₂₂",
                                  color=PT2_COLOR, fontsize=9, zorder=11,
                                  fontweight="bold")
            self._lbl_load = ax.text(gL.real + 0.06, gL.imag - 0.12, "ΓL",
                                      color=LOAD_COLOR, fontsize=9, zorder=11,
                                      fontweight="bold")

        self.draw_idle()

    def full_redraw(self):
        """Call after Z0 or display option changes."""
        self._draw_grid()
        self._draw_dynamic()

    # ── fast update during drag ───────────────────────────────────────────────

    def _update_dynamic_fast(self):
        g1 = self.state.gamma1
        g2 = self.state.gamma2

        # Move point markers
        self._pt1.set_data([g1.real], [g1.imag])
        self._pt2.set_data([g2.real], [g2.imag])

        # Update labels
        if self._lbl1 is not None:
            self._lbl1.set_position((g1.real + 0.06, g1.imag + 0.06))
        if self._lbl2 is not None:
            self._lbl2.set_position((g2.real + 0.06, g2.imag + 0.06))

        # Update VSWR circle
        if self._vswr_line is not None:
            r = abs(g1)
            theta = np.linspace(0, 2 * np.pi, 300)
            self._vswr_line.set_data(r * np.cos(theta), r * np.sin(theta))

        # Update admittance mirrors
        if self._adm1 is not None:
            ag1 = -g1
            self._adm1.set_data([ag1.real], [ag1.imag])
        if self._adm2 is not None:
            ag2 = -g2
            self._adm2.set_data([ag2.real], [ag2.imag])

        # Update load point
        if self._pt_load is not None:
            gL = self.state.gamma_L
            self._pt_load.set_data([gL.real], [gL.imag])
        if self._lbl_load is not None:
            gL = self.state.gamma_L
            self._lbl_load.set_position((gL.real + 0.06, gL.imag - 0.12))

        self.draw_idle()

    # ── mouse events ──────────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is not self._ax or event.button != 1:
            return
        pt = complex(event.xdata, event.ydata)
        d1 = abs(pt - self.state.gamma1)
        d2 = abs(pt - self.state.gamma2)
        dL = abs(pt - self.state.gamma_L)
        # Nearest point wins; ties broken by order
        nearest = min((d1, "p1"), (d2, "p2"), (dL, "load"))
        if nearest[0] < DRAG_THRESH:
            self._dragging = nearest[1]

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        g = complex(event.xdata, event.ydata)
        if abs(g) >= 0.9999:
            g = g / abs(g) * 0.9999
        if self._dragging == "p1":
            self.state.gamma1 = g
        elif self._dragging == "p2":
            self.state.gamma2 = g
        else:  # load
            z_new = gamma_to_z(g, self.state.z0)
            self.state.zL_re = z_new.real
            self.state.zL_im = z_new.imag
        self._update_dynamic_fast()
        self.stateChanged.emit(self.state)

    def _on_release(self, event):
        self._dragging = None


# ─────────────────────────────────────────────────────────────────────────────
# Wave Analysis Canvas
# ─────────────────────────────────────────────────────────────────────────────

# ── Geometry constants (wavelengths) ─────────────────────────────────────────
_DUT_CTR   = 1.00   # DUT centre  →  box: 0.90 … 1.10 λ
_DUT_HALF  = 0.10
_LOAD_CTR  = 1.95   # Load centre →  box: 1.87 … 2.03 λ
_LOAD_HALF = 0.08
_XMAX      = 2.10   # right edge of x-axis


def _wave_left(gamma_in: complex, phase: float, pos: np.ndarray):
    """
    Input-side TL  (0 … DUT).
    Uses Γin (= S11 + S12·S21·ΓL / (1−S22·ΓL)) so the load effect is visible.
    """
    bz = 2 * np.pi * pos
    v_inc = np.cos(phase - bz)
    v_ref = np.real(gamma_in * np.exp(1j * (phase + bz)))
    return v_inc, v_ref, v_inc + v_ref


def _wave_right_full(b2: complex, gamma_L: complex,
                     phase: float, pos: np.ndarray,
                     z_dut: float = _DUT_CTR, z_load: float = _LOAD_CTR):
    """
    Output-side TL  (DUT … Load).
    Forward wave:  b2 · e^(jωt − jβ(z−z_dut))
    Backward wave: ΓL · b2 · e^(−jβ(z_load−z_dut)) · e^(jωt + jβ(z−z_load))
                 = ΓL · b2 · e^(jωt + jβ(z − 2·z_load + z_dut))
    b2 = S21/(1−S22·ΓL) already encodes S22 mismatch.
    ΓL encodes the load reflection.  S12 feeds back into Γin on the left.
    """
    beta = 2 * np.pi
    v_fwd = np.real(b2 * np.exp(1j * (phase - beta * (pos - z_dut))))
    v_bwd = np.real(gamma_L * b2 *
                    np.exp(1j * (phase + beta * (pos - 2 * z_load + z_dut))))
    return v_fwd, v_bwd, v_fwd + v_bwd


class WaveCanvas(FigureCanvas):
    """
    Split-view time-domain wave plot.

    Left  (0 … 0.90 λ):        incident (→) + Γin reflected (←) + total
    Centre (0.90 … 1.10 λ):    2-port DUT schematic box
    Right (1.10 … 1.87 λ):     forward S21 (→) + ΓL reflected (←) + total
    Right-end (1.87 … 2.03 λ): termination load box  ZL
    """
    _PHASE_STEP = 2 * np.pi / 40   # one period in 40 frames ≈ 1.3 s

    def __init__(self, state: State, parent=None):
        self.fig = Figure(figsize=(10, 2.4), facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self.state = state
        self._phase = 0.0

        ax = self.fig.add_axes([0.06, 0.18, 0.93, 0.72])
        self._ax = ax
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors=AXIS_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
        ax.set_xlabel("Position (λ)", color=LABEL_COLOR, fontsize=8)
        ax.set_ylabel("Voltage (V)", color=LABEL_COLOR, fontsize=8)
        ax.set_title("Wave Analysis — Time-Domain Travelling Waves",
                     color=TEXT_COLOR, fontsize=9, pad=4)
        ax.grid(True, color=AXIS_COLOR, lw=0.4, alpha=0.4)
        ax.set_xlim(0, _XMAX)
        ax.set_ylim(-2.2, 2.2)

        # ── position arrays ───────────────────────────────────────────────────
        self._pos_l = np.linspace(0,
                                  _DUT_CTR - _DUT_HALF, 260)
        self._pos_r = np.linspace(_DUT_CTR + _DUT_HALF,
                                  _LOAD_CTR - _LOAD_HALF, 220)

        # ── DUT box ───────────────────────────────────────────────────────────
        dut_x = _DUT_CTR - _DUT_HALF
        dut_w = 2 * _DUT_HALF
        ax.add_patch(mpatches.FancyBboxPatch(
            (dut_x, -1.8), dut_w, 3.6,
            boxstyle="round,pad=0.01",
            facecolor="#1a1a3a", edgecolor="#5566cc", lw=1.6, zorder=5))

        # Wire stubs & inner box
        ax.annotate("", xy=(dut_x + dut_w * 0.38, 0), xytext=(dut_x + 0.005, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaff", lw=1.0), zorder=6)
        ax.annotate("", xy=(dut_x + dut_w - 0.005, 0),
                    xytext=(dut_x + dut_w * 0.62, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaff", lw=1.0), zorder=6)
        ax.add_patch(mpatches.FancyBboxPatch(
            (dut_x + dut_w * 0.3, -0.72), dut_w * 0.4, 1.44,
            boxstyle="round,pad=0.01",
            facecolor="#252550", edgecolor="#7788cc", lw=1.0, zorder=7))
        ax.text(_DUT_CTR, 0.26, "DUT",    ha="center", va="center",
                color="#ddddff", fontsize=7.5, fontweight="bold", zorder=8)
        ax.text(_DUT_CTR, -0.28, "2-Port", ha="center", va="center",
                color="#8888bb", fontsize=5.5, zorder=8)

        # S11 reflection arrow
        ax.annotate("", xy=(dut_x - 0.04, -1.2), xytext=(dut_x + dut_w * 0.1, -1.2),
                    arrowprops=dict(arrowstyle="-|>", color=PT1_COLOR, lw=0.9, alpha=0.55),
                    zorder=6)
        ax.text(dut_x - 0.06, -1.38, "S₁₁", ha="right",
                color=PT1_COLOR, fontsize=5.5, alpha=0.7, zorder=6)

        # S21 transmission arrow
        ax.annotate("", xy=(dut_x + dut_w + 0.04, -1.2),
                    xytext=(dut_x + dut_w * 0.9, -1.2),
                    arrowprops=dict(arrowstyle="-|>", color=TRANS_COLOR, lw=0.9, alpha=0.55),
                    zorder=6)
        ax.text(dut_x + dut_w + 0.05, -1.38, "S₂₁", ha="left",
                color=TRANS_COLOR, fontsize=5.5, alpha=0.7, zorder=6)

        # S12 back-transmission arrow (above centre line)
        ax.annotate("", xy=(dut_x + 0.005, 1.2), xytext=(dut_x + dut_w - 0.005, 1.2),
                    arrowprops=dict(arrowstyle="-|>", color=LOAD_BWD, lw=0.9, alpha=0.55),
                    zorder=6)
        ax.text(dut_x - 0.06, 1.30, "S₁₂", ha="right",
                color=LOAD_BWD, fontsize=5.5, alpha=0.7, zorder=6)

        # ── Load box ─────────────────────────────────────────────────────────
        load_x = _LOAD_CTR - _LOAD_HALF
        load_w = 2 * _LOAD_HALF
        ax.add_patch(mpatches.FancyBboxPatch(
            (load_x, -1.5), load_w, 3.0,
            boxstyle="round,pad=0.01",
            facecolor="#2a1a10", edgecolor=LOAD_COLOR, lw=1.6, zorder=5))

        # Zig-zag resistor symbol (3 teeth)
        zz_x = np.array([load_x + load_w * 0.5,
                          load_x + load_w * 0.55, load_x + load_w * 0.65,
                          load_x + load_w * 0.75, load_x + load_w * 0.85,
                          load_x + load_w * 0.9])
        zz_y = np.array([0, 0.35, -0.35, 0.35, -0.35, 0])
        ax.plot(zz_x, zz_y, color=LOAD_COLOR, lw=1.2, zorder=8, alpha=0.8)
        # Vertical wire to top
        ax.plot([load_x + load_w * 0.5, load_x + load_w * 0.5],
                [0, 0.9], color=LOAD_COLOR, lw=1.0, zorder=8, alpha=0.8)
        # Ground symbol
        ax.plot([load_x + load_w * 0.5, load_x + load_w * 0.5],
                [0, -0.9], color=LOAD_COLOR, lw=1.0, zorder=8, alpha=0.8)
        for width, y_pos in ((0.06, -0.9), (0.04, -1.05), (0.02, -1.2)):
            ax.plot([load_x + load_w * 0.5 - width,
                     load_x + load_w * 0.5 + width],
                    [y_pos, y_pos], color=LOAD_COLOR, lw=1.0, zorder=8, alpha=0.8)

        # Wire from TL to load
        ax.annotate("", xy=(load_x + load_w * 0.5, 0),
                    xytext=(load_x + 0.005, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#aaaaff", lw=1.0), zorder=6)

        # Load S22 reflection arrow (→ back toward DUT)
        ax.annotate("", xy=(load_x - 0.04, -1.2), xytext=(load_x + load_w * 0.3, -1.2),
                    arrowprops=dict(arrowstyle="-|>", color=LOAD_BWD, lw=0.9, alpha=0.55),
                    zorder=6)
        ax.text(load_x - 0.06, -1.38, "ΓL", ha="right",
                color=LOAD_BWD, fontsize=5.5, alpha=0.7, zorder=6)

        # Dynamic ZL label (updated each tick)
        self._zl_text = ax.text(
            _LOAD_CTR, 1.65,
            self._zl_str(state),
            ha="center", va="center", color=LOAD_COLOR,
            fontsize=5.5, fontweight="bold", zorder=9)

        # ── Separators & port labels ──────────────────────────────────────────
        for xv in (_DUT_CTR - _DUT_HALF, _DUT_CTR + _DUT_HALF,
                   _LOAD_CTR - _LOAD_HALF, _LOAD_CTR + _LOAD_HALF):
            ax.axvline(xv, color="#444466", lw=0.6, ls=":", zorder=4)

        ax.text(0.03,               -2.05, "Port 1", color=LABEL_COLOR, fontsize=7)
        ax.text(_LOAD_CTR + _LOAD_HALF + 0.02, -2.05, "ZL",
                color=LOAD_COLOR, fontsize=7, fontweight="bold")

        # ── wave lines ────────────────────────────────────────────────────────
        vf0, vb0, vt0 = _wave_left(state.gamma_in, 0.0, self._pos_l)
        vrf0, vrb0, vrt0 = _wave_right_full(
            state.b2, state.gamma_L, 0.0, self._pos_r)

        (self._li,)   = ax.plot(self._pos_l, vf0,  color=INC_COLOR,   lw=1.2,
                                 label="Incident →",  alpha=0.85)
        (self._lr,)   = ax.plot(self._pos_l, vb0,  color=REF_COLOR,   lw=1.2,
                                 label="← Γin refl.", alpha=0.85)
        (self._lt,)   = ax.plot(self._pos_l, vt0,  color=TOT_COLOR,   lw=1.7,
                                 label="Total (left)")
        (self._lrfwd,)= ax.plot(self._pos_r, vrf0, color=TRANS_COLOR, lw=1.2,
                                 label="S₂₁ fwd →",  alpha=0.85)
        (self._lrbwd,)= ax.plot(self._pos_r, vrb0, color=LOAD_BWD,    lw=1.2,
                                 label="← ΓL refl.", alpha=0.85)
        (self._lrtot,)= ax.plot(self._pos_r, vrt0, color="#ffff66",   lw=1.7,
                                 label="Total (right)")

        ax.legend(loc="upper right", fontsize=6.5, facecolor=PANEL_BG,
                  labelcolor=TEXT_COLOR, edgecolor=AXIS_COLOR,
                  handlelength=1.4, borderpad=0.4, ncol=2)

        self.draw_idle()

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @staticmethod
    def _zl_str(state: State) -> str:
        z = state.zL
        sign = "+" if z.imag >= 0 else "−"
        return f"ZL={z.real:.0f}{sign}j{abs(z.imag):.0f}Ω"

    def _tick(self):
        self._phase = (self._phase + self._PHASE_STEP) % (2 * np.pi)
        st = self.state
        vf, vb, vt       = _wave_left(st.gamma_in, self._phase, self._pos_l)
        vrf, vrb, vrt    = _wave_right_full(st.b2, st.gamma_L,
                                             self._phase, self._pos_r)
        self._li.set_data(self._pos_l, vf)
        self._lr.set_data(self._pos_l, vb)
        self._lt.set_data(self._pos_l, vt)
        self._lrfwd.set_data(self._pos_r, vrf)
        self._lrbwd.set_data(self._pos_r, vrb)
        self._lrtot.set_data(self._pos_r, vrt)
        self._zl_text.set_text(self._zl_str(st))
        self.draw_idle()

    def update_waves(self, state: State):
        self.state = state



# ─────────────────────────────────────────────────────────────────────────────
# Right Control Panel
# ─────────────────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #333355;")
    return f


class RightPanel(QWidget):
    """Compact controls panel — sits above the RF metrics in the right column."""
    z0Changed      = pyqtSignal(float)
    zLChanged      = pyqtSignal(float, float)
    optionChanged  = pyqtSignal(str, bool)
    presetSelected = pyqtSignal(str)
    animToggled    = pyqtSignal(bool)

    def __init__(self, state: State, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumWidth(240)
        from PyQt5.QtWidgets import QSizePolicy as SP
        self.setSizePolicy(SP.Expanding, SP.Preferred)

        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(6, 6, 6, 6)

        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)

        # ── Z0 + Impedances (single compact group) ────────────────────────────
        imp_grp = QGroupBox("Impedances")
        imp_lay = QFormLayout(imp_grp)
        imp_lay.setSpacing(3)

        z0_row = QWidget()
        z0_h = QHBoxLayout(z0_row)
        z0_h.setContentsMargins(0, 0, 0, 0)
        z0_h.addWidget(QLabel("Z₀ ="))
        self.z0_spin = QDoubleSpinBox()
        self.z0_spin.setRange(1.0, 10000.0)
        self.z0_spin.setValue(state.z0)
        self.z0_spin.setDecimals(1)
        self.z0_spin.setSuffix(" Ω")
        self.z0_spin.valueChanged.connect(self.z0Changed.emit)
        z0_h.addWidget(self.z0_spin)
        imp_lay.addRow(z0_row)

        self.z1_lbl = QLabel("50.00 + j0.00 Ω")
        self.z2_lbl = QLabel("—")
        self.z1_lbl.setFont(mono)
        self.z2_lbl.setFont(mono)
        self.z1_lbl.setStyleSheet(f"color:{PT1_COLOR};font-size:11px;")
        self.z2_lbl.setStyleSheet(f"color:{PT2_COLOR};font-size:11px;")
        imp_lay.addRow("Z(P1):", self.z1_lbl)
        imp_lay.addRow("Z(P2):", self.z2_lbl)
        layout.addWidget(imp_grp)

        # ── Termination Load ──────────────────────────────────────────────────
        load_grp = QGroupBox("Termination Load  ZL")
        load_grp.setStyleSheet(
            f"QGroupBox{{border-color:{LOAD_COLOR};color:{LOAD_COLOR};}}")
        load_lay = QFormLayout(load_grp)
        load_lay.setSpacing(3)
        self._block_zl = False

        self.zl_re_spin = QDoubleSpinBox()
        self.zl_re_spin.setRange(-9999.0, 9999.0)
        self.zl_re_spin.setValue(state.zL_re)
        self.zl_re_spin.setDecimals(1)
        self.zl_re_spin.setSuffix(" Ω")
        self.zl_re_spin.valueChanged.connect(self._on_zl_changed)

        self.zl_im_spin = QDoubleSpinBox()
        self.zl_im_spin.setRange(-9999.0, 9999.0)
        self.zl_im_spin.setValue(state.zL_im)
        self.zl_im_spin.setDecimals(1)
        self.zl_im_spin.setSuffix(" Ω")
        self.zl_im_spin.valueChanged.connect(self._on_zl_changed)

        self.gamma_L_lbl  = QLabel("0.000 ∠ 0.0°")
        self.gamma_in_lbl = QLabel("0.000 ∠ 0.0°")
        for lbl in (self.gamma_L_lbl, self.gamma_in_lbl):
            lbl.setFont(mono)
            lbl.setStyleSheet(f"color:{LOAD_COLOR};font-size:10px;")

        load_lay.addRow("Re(ZL):", self.zl_re_spin)
        load_lay.addRow("Im(ZL):", self.zl_im_spin)
        load_lay.addRow("ΓL:",     self.gamma_L_lbl)
        load_lay.addRow("Γin:",    self.gamma_in_lbl)
        layout.addWidget(load_grp)

        # ── Display options + Presets + Animation (one compact row each) ──────
        opt_grp = QGroupBox("Display")
        opt_h = QHBoxLayout(opt_grp)
        opt_h.setSpacing(6)
        self.cb_vswr = QCheckBox("VSWR")
        self.cb_vswr.setChecked(state.show_vswr)
        self.cb_adm  = QCheckBox("Admittance")
        self.cb_adm.setChecked(state.show_admittance)
        self.cb_lbl  = QCheckBox("Labels")
        self.cb_lbl.setChecked(state.show_labels)
        for cb, key in ((self.cb_vswr, "show_vswr"),
                        (self.cb_adm,  "show_admittance"),
                        (self.cb_lbl,  "show_labels")):
            cb.toggled.connect(lambda v, k=key: self.optionChanged.emit(k, v))
            opt_h.addWidget(cb)
        layout.addWidget(opt_grp)

        # Presets + Animation on same row
        bot_row = QWidget()
        bot_h = QHBoxLayout(bot_row)
        bot_h.setContentsMargins(0, 0, 0, 0)
        bot_h.setSpacing(6)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self.presetSelected.emit)
        bot_h.addWidget(self.preset_combo, stretch=1)

        for sym, play in (("▶", True), ("⏸", False)):
            btn = QPushButton(sym)
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda checked, p=play: self.animToggled.emit(p))
            bot_h.addWidget(btn)
        # Keep references for stop
        self.btn_play  = bot_h.itemAt(1).widget()
        self.btn_pause = bot_h.itemAt(2).widget()
        self.btn_stop  = QPushButton("⏹")
        self.btn_stop.setFixedWidth(36)
        self.btn_stop.clicked.connect(lambda: self.animToggled.emit(False))
        bot_h.addWidget(self.btn_stop)
        layout.addWidget(bot_row)

        # ── Match quality badge ───────────────────────────────────────────────
        self.match_lbl = QLabel("")
        self.match_lbl.setWordWrap(True)
        self.match_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.match_lbl)

        self.update_display(state)

    def _on_zl_changed(self):
        if not self._block_zl:
            self.zLChanged.emit(self.zl_re_spin.value(), self.zl_im_spin.value())

    def _sync_zl_spinboxes(self, state: State):
        self._block_zl = True
        self.zl_re_spin.setValue(state.zL_re)
        self.zl_im_spin.setValue(state.zL_im)
        self._block_zl = False

    def _anim_stop(self):
        self.animToggled.emit(False)

    def update_display(self, state: State):
        self.z1_lbl.setText(_fmt_z(state.z1))
        self.z2_lbl.setText(_fmt_z(state.z2))
        self._sync_zl_spinboxes(state)
        gL  = state.gamma_L
        gin = state.gamma_in
        self.gamma_L_lbl.setText(
            f"{abs(gL):.4f}  ∠ {np.degrees(np.angle(gL)):.1f}°")
        self.gamma_in_lbl.setText(
            f"{abs(gin):.4f}  ∠ {np.degrees(np.angle(gin)):.1f}°")
        # Match quality — use Γin so load is factored in
        rl = return_loss_db(gin)
        if rl == float("inf") or rl >= 30:
            txt, css = ("✓ Excellent Match  RL > 30 dB",
                        "background:#0d2a1a;border:1px solid #1a6632;"
                        "border-radius:5px;padding:4px;font-size:10px;color:#55ff99;")
        elif rl >= 20:
            txt, css = ("✓ Good Match  RL > 20 dB",
                        "background:#132813;border:1px solid #225522;"
                        "border-radius:5px;padding:4px;font-size:10px;color:#88dd88;")
        elif rl >= 10:
            txt, css = ("~ Fair Match  RL > 10 dB",
                        "background:#2a2a10;border:1px solid #665520;"
                        "border-radius:5px;padding:4px;font-size:10px;color:#ddcc55;")
        else:
            txt, css = ("✗ Poor Match  RL < 10 dB",
                        "background:#2a1010;border:1px solid #662020;"
                        "border-radius:5px;padding:4px;font-size:10px;color:#ff6666;")
        self.match_lbl.setText(txt)
        self.match_lbl.setStyleSheet(css)


def _fmt_z(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:.2f} {sign} j{abs(z.imag):.2f} Ω"


# ─────────────────────────────────────────────────────────────────────────────
# RF Metrics Panel  (replaces SParamPanel + CircuitCanvas)
# ─────────────────────────────────────────────────────────────────────────────

def _db(linear_mag: float) -> str:
    if linear_mag < 1e-12:
        return "-∞ dB"
    return f"{20 * np.log10(linear_mag):.2f} dB"

def _pct(power_ratio: float) -> str:
    return f"{min(power_ratio, 1.0) * 100:.1f} %"

def _fmt_gamma(g: complex) -> str:
    return f"{abs(g):.4f}  ∠ {np.degrees(np.angle(g)):.1f}°"


class RFMetricsPanel(QWidget):
    """
    Right-column panel: S-parameter table at top, 2×2 metric groups below.
    Designed to sit in the right splitter column alongside the Smith Chart.
    """
    from PyQt5.QtWidgets import QGridLayout as _QGridLayout

    # Colour per metric category
    _C_P1   = PT1_COLOR    # port 1 / red
    _C_P2   = PT2_COLOR    # port 2 / blue
    _C_TX   = TRANS_COLOR  # transmission / cyan
    _C_LOAD = LOAD_COLOR   # load / yellow
    _C_DEF  = TEXT_COLOR   # default

    def __init__(self, state: State, parent=None):
        super().__init__(parent)
        from PyQt5.QtWidgets import QGridLayout
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(5)

        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)

        # ── S-parameter table (full width at top) ─────────────────────────────
        hdr = QLabel("S-Parameters  (2-Port DUT)")
        hdr.setAlignment(Qt.AlignCenter)
        hdr.setStyleSheet(
            f"color:{TEXT_COLOR};font-weight:bold;font-size:11px;"
            "border-bottom:1px solid #333355;padding-bottom:2px;")
        root.addWidget(hdr)

        # 5 columns: Param | |S| dB | ∠ (°) | |S| lin | |S|² %
        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels(
            ["", "|S| dB", "∠ (°)", "|S|", "|S|² %"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setDefaultSectionSize(56)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(118)

        from PyQt5.QtGui import QColor
        _PARAM_COLORS = [PT1_COLOR, TRANS_COLOR, LOAD_BWD, PT2_COLOR]
        for i, (name, col) in enumerate(
                zip(["S₁₁", "S₂₁", "S₁₂", "S₂₂"], _PARAM_COLORS)):
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor(col))
            self.table.setItem(i, 0, item)

        root.addWidget(self.table)

        # ── 2 × 2 metric grid ─────────────────────────────────────────────────
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        def _mrow(layout, label_txt, color=TEXT_COLOR):
            lbl = QLabel()
            lbl.setFont(mono)
            lbl.setStyleSheet(f"color:{color};font-size:10px;")
            layout.addRow(label_txt, lbl)
            return lbl

        # Port 1 / Γin  (top-left) ─────────────────────────────────────────────
        grp1 = QGroupBox("Port 1  (Γin)")
        grp1.setStyleSheet(f"QGroupBox{{color:{self._C_P1};"
                           f"border-color:{self._C_P1};}}")
        lay1 = QFormLayout(grp1)
        lay1.setSpacing(1)
        lay1.setContentsMargins(6, 10, 6, 4)
        self.m_gin       = _mrow(lay1, "Γin:",          self._C_P1)
        self.m_vswr1     = _mrow(lay1, "VSWR:",         self._C_P1)
        self.m_rl1       = _mrow(lay1, "Ret. Loss:",    self._C_P1)
        self.m_refl_pwr1 = _mrow(lay1, "Refl. Pwr:",   self._C_P1)
        self.m_mism1     = _mrow(lay1, "Mism. Loss:",   self._C_P1)
        self.m_avail1    = _mrow(lay1, "Avail. Pwr:",   self._C_P1)
        grid.addWidget(grp1, 0, 0)

        # Transmission S21 / S12  (top-right) ─────────────────────────────────
        grp_tx = QGroupBox("Transmission  S21/S12")
        grp_tx.setStyleSheet(f"QGroupBox{{color:{self._C_TX};"
                             f"border-color:{self._C_TX};}}")
        lay_tx = QFormLayout(grp_tx)
        lay_tx.setSpacing(1)
        lay_tx.setContentsMargins(6, 10, 6, 4)
        self.m_ins_loss  = _mrow(lay_tx, "Ins. Loss:",    self._C_TX)
        self.m_tx_pwr    = _mrow(lay_tx, "Trans. Pwr:",   self._C_TX)
        self.m_tx_phase  = _mrow(lay_tx, "Phase S21:",    self._C_TX)
        grid.addWidget(grp_tx, 0, 1)

        # Port 2 / S22  (bottom-left) ──────────────────────────────────────────
        grp2 = QGroupBox("Port 2  (S22)")
        grp2.setStyleSheet(f"QGroupBox{{color:{self._C_P2};"
                           f"border-color:{self._C_P2};}}")
        lay2 = QFormLayout(grp2)
        lay2.setSpacing(1)
        lay2.setContentsMargins(6, 10, 6, 4)
        self.m_s22       = _mrow(lay2, "S22:",          self._C_P2)
        self.m_vswr2     = _mrow(lay2, "VSWR:",         self._C_P2)
        self.m_rl2       = _mrow(lay2, "Ret. Loss:",    self._C_P2)
        self.m_refl_pwr2 = _mrow(lay2, "Refl. Pwr:",   self._C_P2)
        grid.addWidget(grp2, 1, 0)

        # Load  (bottom-right) ─────────────────────────────────────────────────
        grp_ld = QGroupBox("Load  (ZL / ΓL)")
        grp_ld.setStyleSheet(f"QGroupBox{{color:{self._C_LOAD};"
                             f"border-color:{self._C_LOAD};}}")
        lay_ld = QFormLayout(grp_ld)
        lay_ld.setSpacing(1)
        lay_ld.setContentsMargins(6, 10, 6, 4)
        self.m_zl        = _mrow(lay_ld, "ZL:",             self._C_LOAD)
        self.m_gL        = _mrow(lay_ld, "ΓL:",             self._C_LOAD)
        self.m_vswr_ld   = _mrow(lay_ld, "VSWR:",           self._C_LOAD)
        self.m_rl_ld     = _mrow(lay_ld, "Ret. Loss:",      self._C_LOAD)
        self.m_pwr_ld    = _mrow(lay_ld, "Pwr to Load:",    self._C_LOAD)
        grid.addWidget(grp_ld, 1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        root.addWidget(grid_w, stretch=1)

        self.update_metrics(state)

    def update_metrics(self, state: State):
        S   = state.s_matrix
        s11, s21, s12, s22 = S[0,0], S[1,0], S[0,1], S[1,1]
        gin = state.gamma_in
        gL  = state.gamma_L
        b2  = state.b2

        # ── S-parameter table ─────────────────────────────────────────────────
        for i, s in enumerate([s11, s21, s12, s22]):
            mag  = abs(s)
            cols = [
                f"{20*np.log10(max(mag,1e-12)):.2f}",
                f"{np.degrees(np.angle(s)):.1f}°",
                f"{mag:.4f}",
                f"{mag**2*100:.1f}",
            ]
            for col, txt in enumerate(cols, start=1):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, item)

        # ── Port 1 / Γin ──────────────────────────────────────────────────────
        refl1  = abs(gin) ** 2
        avail1 = 1.0 - refl1
        ml1    = -10*np.log10(max(avail1, 1e-12))   # mismatch loss dB
        self.m_gin.setText(_fmt_gamma(gin))
        self.m_vswr1.setText(f"{vswr(gin):.3f}")
        self.m_rl1.setText(_db(abs(gin)))
        self.m_refl_pwr1.setText(_pct(refl1))
        self.m_mism1.setText(f"{ml1:.3f} dB")
        self.m_avail1.setText(_pct(avail1))

        # ── Transmission ──────────────────────────────────────────────────────
        self.m_ins_loss.setText(_db(abs(s21)))       # insertion loss (–ve = gain)
        self.m_tx_pwr.setText(_pct(abs(s21) ** 2))
        self.m_tx_phase.setText(f"{np.degrees(np.angle(s21)):.1f}°")

        # ── Port 2 / S22 ──────────────────────────────────────────────────────
        self.m_s22.setText(_fmt_gamma(s22))
        self.m_vswr2.setText(f"{vswr(s22):.3f}")
        self.m_rl2.setText(_db(abs(s22)))
        self.m_refl_pwr2.setText(_pct(abs(s22) ** 2))

        # ── Load ──────────────────────────────────────────────────────────────
        zl = state.zL
        sign = "+" if zl.imag >= 0 else "−"
        self.m_zl.setText(f"{zl.real:.2f} {sign} j{abs(zl.imag):.2f} Ω")
        self.m_gL.setText(_fmt_gamma(gL))
        self.m_vswr_ld.setText(f"{vswr(gL):.3f}")
        self.m_rl_ld.setText(_db(abs(gL)))
        # Power to load: |b2|²·(1−|ΓL|²), normalised to incident power at port 1
        pwr_load = abs(b2) ** 2 * (1.0 - abs(gL) ** 2)
        self.m_pwr_ld.setText(_pct(pwr_load))


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

DARK_QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT_COLOR};
    font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid #333355;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 6px;
    color: #9999cc;
    font-weight: bold;
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    color: #aaaadd;
}}
QDoubleSpinBox, QComboBox {{
    background-color: #1e1e35;
    border: 1px solid #444466;
    border-radius: 4px;
    padding: 3px 6px;
    color: {TEXT_COLOR};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: #2a2a4a;
}}
QComboBox QAbstractItemView {{
    background: #1e1e35;
    color: {TEXT_COLOR};
    selection-background-color: #2a3580;
}}
QPushButton {{
    background-color: #2a2a4a;
    border: 1px solid #555577;
    border-radius: 4px;
    padding: 4px 10px;
    color: #aaaadd;
}}
QPushButton:hover  {{ background-color: #3a3a6a; }}
QPushButton:pressed {{ background-color: #222240; }}
QCheckBox {{ color: #aaaacc; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid #555577;
    border-radius: 3px;
    background: #1e1e35;
}}
QCheckBox::indicator:checked {{ background: #4466cc; }}
QSplitter::handle {{ background: #2a2a44; width: 3px; height: 3px; }}
QLabel {{ color: {TEXT_COLOR}; }}
QScrollBar {{ background: #1a1a2e; }}
QTableWidget {{
    background-color: {PANEL_BG};
    gridline-color: #333355;
    color: {TEXT_COLOR};
    border: none;
}}
QTableWidget QHeaderView::section {{
    background-color: #1e1e35;
    color: {LABEL_COLOR};
    border: none;
    padding: 3px;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smith Chart Simulator")
        self.setMinimumSize(1100, 780)
        self.state = State()

        self._build_ui()
        self._connect_signals()
        self.setStyleSheet(DARK_QSS)

        # Animation timer
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(40)  # 25 FPS
        self._anim_angle = np.angle(self.state.gamma1)
        self._anim_radius = max(abs(self.state.gamma1), 0.3)
        self._anim_timer.timeout.connect(self._anim_step)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Top row: Smith Chart (left) | RightPanel + RFMetrics (right) ──────
        main_split = QSplitter(Qt.Horizontal)

        # Left: square Smith Chart
        self.smith = SmithChartCanvas(self.state)
        self.smith.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_split.addWidget(self.smith)

        # Right column: controls on top, metrics on bottom
        right_vsplit = QSplitter(Qt.Vertical)
        self.right   = RightPanel(self.state)
        self.sparams = RFMetricsPanel(self.state)
        right_vsplit.addWidget(self.right)
        right_vsplit.addWidget(self.sparams)
        right_vsplit.setSizes([280, 500])
        right_vsplit.setStretchFactor(0, 0)
        right_vsplit.setStretchFactor(1, 1)

        main_split.addWidget(right_vsplit)
        main_split.setSizes([640, 420])
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 0)

        # ── Bottom: Wave analysis (full width) ────────────────────────────────
        wave_wrap = QGroupBox("Wave Analysis — Voltage vs Position")
        wave_lay = QVBoxLayout(wave_wrap)
        wave_lay.setContentsMargins(4, 4, 4, 4)
        self.wave = WaveCanvas(self.state)
        wave_lay.addWidget(self.wave)

        root.addWidget(main_split, stretch=5)
        root.addWidget(wave_wrap,  stretch=2)

    # ── signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.smith.stateChanged.connect(self._on_state_changed)
        self.right.z0Changed.connect(self._on_z0_changed)
        self.right.zLChanged.connect(self._on_zL_changed)
        self.right.optionChanged.connect(self._on_option_changed)
        self.right.presetSelected.connect(self._on_preset)
        self.right.animToggled.connect(self._on_anim_toggle)

    # ── handlers ──────────────────────────────────────────────────────────────

    def _on_state_changed(self, state: State):
        """Hot path — called during every drag motion event."""
        self.right.update_display(state)
        self.sparams.update_metrics(state)
        self.wave.update_waves(state)

    def _on_z0_changed(self, z0: float):
        self.state.z0 = z0
        self.smith.full_redraw()
        self._on_state_changed(self.state)

    def _on_zL_changed(self, re: float, im: float):
        self.state.zL_re = re
        self.state.zL_im = im
        # Update ΓL point on Smith Chart without full grid redraw
        self.smith._update_dynamic_fast()
        self._on_state_changed(self.state)

    def _on_option_changed(self, key: str, value: bool):
        setattr(self.state, key, value)
        self.smith.full_redraw()

    def _on_preset(self, name: str):
        vals = PRESETS.get(name)
        if vals is None:
            return
        g1, g2 = vals
        self.state.gamma1 = g1
        self.state.gamma2 = g2
        self.smith._draw_dynamic()
        self._on_state_changed(self.state)

    def _on_anim_toggle(self, play: bool):
        if play:
            self._anim_radius = max(abs(self.state.gamma1), 0.3)
            self._anim_angle  = np.angle(self.state.gamma1)
            self._anim_timer.start()
        else:
            self._anim_timer.stop()

    def _anim_step(self):
        self._anim_angle = (self._anim_angle + 0.04) % (2 * np.pi)
        r = self._anim_radius
        self.state.gamma1 = complex(r * np.cos(self._anim_angle),
                                    r * np.sin(self._anim_angle))
        self.smith._update_dynamic_fast()
        self._on_state_changed(self.state)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    app.setApplicationName("Smith Chart Simulator")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
