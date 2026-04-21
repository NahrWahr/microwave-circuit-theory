from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Scene 3 — Two-Port Networks & S-Parameters
# Topics 8–12
# ---------------------------------------------------------------------------

class TwoPortAndSParams(Scene):

    def construct(self):
        self.act1_title()
        self.act2_two_port_basics()
        self.act3_why_sparams()
        self.act4_sparam_definitions()
        self.act5_smatrix_elements()
        self.act6_smatrix_properties()

    # ── helpers ───────────────────────────────────────────────────────────

    def _twoport_base(self, yc=0.0, label=r"\text{Network}") -> VGroup:
        """Central box + double-wire port stubs. Returns VGroup."""
        box = Rectangle(width=3.2, height=2.0, color=WHITE, stroke_width=2)
        box.move_to([0, yc, 0])
        lbl = MathTex(label, font_size=26).move_to(box)

        def stub(x1, x2, dy):
            return Line([x1, yc + dy, 0], [x2, yc + dy, 0],
                        color=WHITE, stroke_width=2)

        stubs = VGroup(
            stub(-2.6, -1.6, +0.38), stub(-2.6, -1.6, -0.38),  # port 1
            stub(+1.6, +2.6, +0.38), stub(+1.6, +2.6, -0.38),  # port 2
        )
        # Port index labels
        p1 = Text("port 1", font_size=18, color=GRAY)\
            .move_to([-2.8, yc - 0.72, 0])
        p2 = Text("port 2", font_size=18, color=GRAY)\
            .move_to([+2.8, yc - 0.72, 0])
        return VGroup(box, lbl, stubs, p1, p2)

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Two-Port Networks & S-Parameters", font_size=40)
        sub = Text("the language of high-frequency circuits",
                   font_size=26, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2: Two-Port Basics ─────────────────────────────────────────────
    def act2_two_port_basics(self):
        title = Text("Two-Port Network Parameters",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        tp = self._twoport_base(yc=0.7)
        self.play(Create(tp))

        # Voltage / current labels
        v1 = MathTex(r"V_1", font_size=28, color=BLUE).move_to([-3.6, 1.35, 0])
        i1 = MathTex(r"I_1 \!\rightarrow", font_size=28, color=BLUE)\
            .move_to([-3.6, 0.55, 0])
        v2 = MathTex(r"V_2", font_size=28, color=GREEN).move_to([3.6, 1.35, 0])
        i2 = MathTex(r"\leftarrow\! I_2", font_size=28, color=GREEN)\
            .move_to([3.6, 0.55, 0])
        self.play(FadeIn(VGroup(v1, i1, v2, i2)))
        self.wait(0.6)

        # Z-matrix
        z_lbl = Text("Z-matrix (impedance parameters):",
                     font_size=22, color=GRAY).move_to([0, -1.0, 0])
        z_eq = MathTex(
            r"\begin{pmatrix}V_1\\V_2\end{pmatrix}"
            r"=\begin{pmatrix}Z_{11}&Z_{12}\\Z_{21}&Z_{22}\end{pmatrix}"
            r"\begin{pmatrix}I_1\\I_2\end{pmatrix}",
            font_size=34,
        ).move_to([0, -1.9, 0])
        self.play(Write(z_lbl), Write(z_eq))
        self.wait(1)

        # Mention Y, ABCD
        others = MathTex(
            r"\text{also:}\quad"
            r"[I]=[Y][V],\quad"
            r"\begin{pmatrix}V_1\\I_1\end{pmatrix}"
            r"=[ABCD]\begin{pmatrix}V_2\\-I_2\end{pmatrix}",
            font_size=24, color=GRAY,
        ).move_to([0, -3.0, 0])
        self.play(FadeIn(others))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, tp, v1, i1, v2, i2, z_lbl, z_eq, others)))

    # ── Act 3: Why S-Parameters? ──────────────────────────────────────────
    def act3_why_sparams(self):
        title = Text("Why S-Parameters?", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        items = [
            (r"Z\!/\!Y\text{ require open or short-circuit terminations}",
             WHITE),
            (r"\text{At GHz: open} \neq \text{ideal — stray capacitance}",
             RED),
            (r"\text{At GHz: short} \neq \text{ideal — stray inductance}",
             RED),
            (r"\text{Matched loads }(Z_0)\text{ are broadband and realisable}",
             GREEN),
            (r"\text{S-params measured directly with a VNA}",
             GREEN),
        ]

        rows = VGroup()
        for tex, color in items:
            rows.add(MathTex(tex, font_size=28, color=color))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.45).move_to([0, -0.2, 0])

        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.6)
            self.wait(0.4)

        self.wait(2)
        self.play(FadeOut(VGroup(title, rows)))

    # ── Act 4: S-Parameter Definitions ────────────────────────────────────
    def act4_sparam_definitions(self):
        title = Text("S-Parameter Definitions",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        tp = self._twoport_base(yc=0.6, label=r"\text{DUT}")
        self.play(Create(tp))

        # Wave arrows (thick, colour-coded)
        yc = 0.6
        a1_arr = Arrow([-4.2, yc + 0.38, 0], [-2.6, yc + 0.38, 0], buff=0,
                       color=BLUE, stroke_width=2.5, tip_length=0.22)
        b1_arr = Arrow([-2.6, yc - 0.38, 0], [-4.2, yc - 0.38, 0], buff=0,
                       color=RED,  stroke_width=2.5, tip_length=0.22)
        a2_arr = Arrow([4.2, yc + 0.38, 0], [2.6, yc + 0.38, 0], buff=0,
                       color=BLUE, stroke_width=2.5, tip_length=0.22)
        b2_arr = Arrow([2.6, yc - 0.38, 0], [4.2, yc - 0.38, 0], buff=0,
                       color=RED,  stroke_width=2.5, tip_length=0.22)

        a1_lbl = MathTex(r"a_1", font_size=28, color=BLUE)\
            .move_to([-3.6, yc + 0.80, 0])
        b1_lbl = MathTex(r"b_1", font_size=28, color=RED)\
            .move_to([-3.6, yc - 0.80, 0])
        a2_lbl = MathTex(r"a_2", font_size=28, color=BLUE)\
            .move_to([3.6,  yc + 0.80, 0])
        b2_lbl = MathTex(r"b_2", font_size=28, color=RED)\
            .move_to([3.6,  yc - 0.80, 0])

        self.play(
            LaggedStart(
                Create(a1_arr), Create(b1_arr),
                Create(a2_arr), Create(b2_arr),
                lag_ratio=0.25,
            ),
            FadeIn(VGroup(a1_lbl, b1_lbl, a2_lbl, b2_lbl)),
        )
        self.wait(0.6)

        # Normalised wave definitions
        def_a = MathTex(
            r"a_i = \frac{V_i^+}{\sqrt{Z_0}}", font_size=34, color=BLUE,
        ).move_to([-2.5, -1.2, 0])
        def_b = MathTex(
            r"b_i = \frac{V_i^-}{\sqrt{Z_0}}", font_size=34, color=RED,
        ).move_to([2.5, -1.2, 0])
        note_a = Text("incident", font_size=20, color=BLUE)\
            .next_to(def_a, DOWN, buff=0.1)
        note_b = Text("scattered / reflected", font_size=20, color=RED)\
            .next_to(def_b, DOWN, buff=0.1)

        power_note = MathTex(
            r"|a_i|^2 \;=\; \text{power incident at port }i",
            font_size=26, color=GRAY,
        ).move_to([0, -2.5, 0])

        self.play(Write(def_a), Write(def_b))
        self.play(FadeIn(note_a), FadeIn(note_b))
        self.play(FadeIn(power_note))
        self.wait(1.5)
        self.play(FadeOut(VGroup(def_a, def_b, note_a, note_b, power_note)))

        # Animated wave bounce: a1 pulse hits the DUT → S11 reflects, S21 transmits
        bounce_lbl = Text("incident a₁ pulse  →  S₁₁ reflects,  S₂₁ transmits",
                          font_size=22, color=GRAY).to_edge(DOWN, buff=1.2)
        self.play(FadeIn(bounce_lbl))

        XL, XR = -4.2, 4.2
        X_PORT1, X_PORT2 = -1.6, 1.6
        y_top, y_bot = yc + 0.38, yc - 0.38

        # Three amplitude plots (Bode-like) under the diagram
        ax_cfg = dict(
            x_range=[0, 1.01, 0.5], y_range=[-1.2, 1.2, 1],
            x_length=3.8, y_length=1.5, tips=False,
            axis_config={"stroke_width": 1.5, "include_numbers": False},
        )
        ax_a1 = Axes(**ax_cfg).move_to([-4.2, -2.2, 0])
        ax_b1 = Axes(**ax_cfg).move_to([0.0,  -2.2, 0])
        ax_b2 = Axes(**ax_cfg).move_to([4.2,  -2.2, 0])
        lbl_a1 = MathTex(r"a_1(t)", font_size=20, color=BLUE).next_to(ax_a1, UP, buff=0.05)
        lbl_b1 = MathTex(r"b_1(t)=S_{11}a_1", font_size=20, color=RED).next_to(ax_b1, UP, buff=0.05)
        lbl_b2 = MathTex(r"b_2(t)=S_{21}a_1", font_size=20, color=GREEN).next_to(ax_b2, UP, buff=0.05)

        # Trackers
        t = ValueTracker(0.0)
        s11_mag = ValueTracker(0.3)
        s21_mag = ValueTracker(0.85)

        def pulse(x, x0, width=0.35, amp=1.0):
            return amp * np.exp(-((x - x0) / width) ** 2) * np.cos(6 * (x - x0))

        def a1_curve():
            t_val = t.get_value()
            x0 = XL + (X_PORT1 - XL) * min(t_val, 1.0)
            return ax_a1.plot(
                lambda x: pulse(x, x0) if t_val <= 1.0 else 0,
                x_range=[0, 1.0, 0.01], color=BLUE, stroke_width=2.5,
            ) if t_val <= 1.0 else VMobject()

        # Spatial view pulses along the wires (on the diagram itself)
        def a1_shape():
            tv = t.get_value()
            if tv > 1.0:
                return VMobject()
            x0 = XL + (X_PORT1 - XL) * tv
            return ParametricFunction(
                lambda s: np.array([s, y_top + 0.4 * pulse(s, x0), 0]),
                t_range=[XL, X_PORT1, 0.02], color=BLUE, stroke_width=3,
            )

        def b1_shape():
            tv = t.get_value()
            if tv < 1.0:
                return VMobject()
            frac = min(tv - 1.0, 1.0)
            x0 = X_PORT1 - (X_PORT1 - XL) * frac
            amp = s11_mag.get_value()
            return ParametricFunction(
                lambda s: np.array([s, y_bot - 0.4 * amp * pulse(s, x0), 0]),
                t_range=[XL, X_PORT1, 0.02], color=RED, stroke_width=3,
            )

        def b2_shape():
            tv = t.get_value()
            if tv < 1.0:
                return VMobject()
            frac = min(tv - 1.0, 1.0)
            x0 = X_PORT2 + (XR - X_PORT2) * frac
            amp = s21_mag.get_value()
            return ParametricFunction(
                lambda s: np.array([s, y_bot - 0.4 * amp * pulse(s, x0), 0]),
                t_range=[X_PORT2, XR, 0.02], color=GREEN, stroke_width=3,
            )

        a1_mob = always_redraw(a1_shape)
        b1_mob = always_redraw(b1_shape)
        b2_mob = always_redraw(b2_shape)

        readout = always_redraw(lambda: MathTex(
            rf"|S_{{11}}|={s11_mag.get_value():.2f},\ "
            rf"|S_{{21}}|={s21_mag.get_value():.2f}",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.3))

        self.play(Create(ax_a1), Create(ax_b1), Create(ax_b2),
                  FadeIn(lbl_a1), FadeIn(lbl_b1), FadeIn(lbl_b2))

        # Bar-height markers on the three panels
        def bar_a1():
            return Line(ax_a1.c2p(0.5, 0), ax_a1.c2p(0.5, 1.0),
                        color=BLUE, stroke_width=4)
        def bar_b1():
            return Line(ax_b1.c2p(0.5, 0),
                        ax_b1.c2p(0.5, s11_mag.get_value()),
                        color=RED, stroke_width=4)
        def bar_b2():
            return Line(ax_b2.c2p(0.5, 0),
                        ax_b2.c2p(0.5, s21_mag.get_value()),
                        color=GREEN, stroke_width=4)

        bar_a = always_redraw(bar_a1)
        bar_br = always_redraw(bar_b1)
        bar_bt = always_redraw(bar_b2)

        self.add(a1_mob, b1_mob, b2_mob, bar_a, bar_br, bar_bt, readout)

        # Phase 1: pulse travels to port 1
        self.play(t.animate.set_value(1.0), run_time=1.8, rate_func=linear)
        # Phase 2: reflection + transmission depart
        self.play(t.animate.set_value(2.0), run_time=1.8, rate_func=linear)
        self.wait(0.4)

        # Sweep S11: more reflective (|S11| up) → lossless tradeoff
        self.play(s11_mag.animate.set_value(0.7),
                  s21_mag.animate.set_value(0.6), run_time=1.8)
        self.wait(0.5)
        self.play(s11_mag.animate.set_value(0.1),
                  s21_mag.animate.set_value(0.95), run_time=1.8)
        self.wait(0.8)

        for m in (a1_mob, b1_mob, b2_mob, bar_a, bar_br, bar_bt, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(
            title, tp, a1_arr, b1_arr, a2_arr, b2_arr,
            a1_lbl, b1_lbl, a2_lbl, b2_lbl,
            bounce_lbl, ax_a1, ax_b1, ax_b2, lbl_a1, lbl_b1, lbl_b2,
            a1_mob, b1_mob, b2_mob, bar_a, bar_br, bar_bt, readout,
        )))

    # ── Act 5: S-Matrix Equations ──────────────────────────────────────────
    def act5_smatrix_elements(self):
        title = Text("The S-Matrix", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Full [b] = [S][a]
        matrix_eq = MathTex(
            r"\underbrace{\begin{pmatrix}b_1\\b_2\end{pmatrix}}_{[b]}"
            r"=\underbrace{\begin{pmatrix}S_{11}&S_{12}\\S_{21}&S_{22}"
            r"\end{pmatrix}}_{[S]}"
            r"\underbrace{\begin{pmatrix}a_1\\a_2\end{pmatrix}}_{[a]}",
            font_size=36,
        ).move_to([0, 1.9, 0])
        self.play(Write(matrix_eq))
        self.wait(1)

        # Each element: equation + description, revealed one at a time
        data = [
            (r"S_{11}=\left.\frac{b_1}{a_1}\right|_{a_2=0}",
             "input reflection coefficient",  BLUE),
            (r"S_{21}=\left.\frac{b_2}{a_1}\right|_{a_2=0}",
             "forward transmission  (gain)",  GREEN),
            (r"S_{12}=\left.\frac{b_1}{a_2}\right|_{a_1=0}",
             "reverse transmission",          YELLOW),
            (r"S_{22}=\left.\frac{b_2}{a_2}\right|_{a_1=0}",
             "output reflection coefficient", RED),
        ]

        rows = VGroup()
        for tex_str, desc, col in data:
            eq  = MathTex(tex_str, font_size=28, color=col)
            txt = Text(f"— {desc}", font_size=20, color=GRAY)
            row = VGroup(eq, txt).arrange(RIGHT, buff=0.35)
            rows.add(row)
        rows.arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to([0.3, -0.5, 0])

        for row in rows:
            self.play(FadeIn(row, shift=UP * 0.15), run_time=0.65)
            self.wait(0.55)

        # Measurement condition note
        meas = MathTex(
            r"a_i = 0 \;\Leftrightarrow\; \text{port }i"
            r"\text{ terminated in }Z_0",
            font_size=22, color=GRAY,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(meas))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, matrix_eq, rows, meas)))

    # ── Act 6: S-Matrix Properties ─────────────────────────────────────────
    def act6_smatrix_properties(self):
        title = Text("Properties of the S-Matrix",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        props = [
            (
                "Reciprocal",
                r"[S]=[S]^T \;\Longrightarrow\; S_{12}=S_{21}",
                r"\text{e.g. passive linear networks}",
                YELLOW,
            ),
            (
                "Lossless",
                r"[S]^\dagger [S] = [I] \quad\text{(unitary)}",
                r"\text{no power is absorbed by the network}",
                GREEN,
            ),
            (
                "Passive",
                r"\sum_j |S_{ij}|^2 \leq 1 \quad \forall\,i",
                r"\text{output power} \leq \text{input power}",
                RED,
            ),
        ]

        groups = VGroup()
        for name, eq_str, note_str, col in props:
            header = Text(name + ":", font_size=26, color=col)
            eq     = MathTex(eq_str,   font_size=32)
            note   = MathTex(note_str, font_size=22, color=GRAY)
            block  = VGroup(header, eq, note).arrange(RIGHT, buff=0.45)
            groups.add(block)
        groups.arrange(DOWN, buff=0.65, aligned_edge=LEFT).move_to([0, -0.3, 0])

        for g in groups:
            self.play(FadeIn(g, shift=UP * 0.2), run_time=0.8)
            self.wait(1.5)

        self.wait(2)
        self.play(FadeOut(VGroup(title, groups)))
