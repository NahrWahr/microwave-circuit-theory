from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Scene 2 — Reflection Coefficient, Standing Waves, Power Flow
# Topics 5-7
# ---------------------------------------------------------------------------

class ReflectionAndImpedance(Scene):

    def construct(self):
        self.act1_title()
        self.act2_reflection_coefficient()
        self.act3_input_impedance()
        self.act4_standing_waves()
        self.act5_power_flow()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Reflection, Impedance & Power", font_size=44)
        sub = Text("what happens at a mismatched load",
                   font_size=26, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2 — Reflection Coefficient ────────────────────────────────────
    def act2_reflection_coefficient(self):
        title = Text("Reflection Coefficient  Γ",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Transmission line diagram with load
        self._draw_tline_with_load()

        # Boundary condition at z = 0
        bc_label = Text("Boundary condition at the load  (z = 0):",
                        font_size=24, color=GRAY).move_to([0, 0.2, 0])
        bc_eq = MathTex(
            r"V(0) = Z_L\, I(0)",
            font_size=36
        ).move_to([0, -0.5, 0])

        self.play(Write(bc_label))
        self.play(Write(bc_eq))
        self.wait(1)
        self.play(FadeOut(bc_label), FadeOut(bc_eq))

        # Derive Γ
        derive1 = MathTex(
            r"\Gamma_L \;=\; \frac{V^-}{V^+}\bigg|_{z=0}",
            font_size=38
        ).move_to([0, 0.5, 0])
        derive2 = MathTex(
            r"\Gamma_L \;=\; \frac{Z_L - Z_0}{Z_L + Z_0}",
            font_size=44
        ).move_to([0, -0.7, 0])
        box = SurroundingRectangle(derive2, color=BLUE, buff=0.2, corner_radius=0.1)

        self.play(Write(derive1))
        self.play(TransformFromCopy(derive1, derive2))
        self.play(Create(box))
        self.wait(1)

        # Special cases
        cases = VGroup(
            MathTex(r"Z_L = Z_0 \;\Rightarrow\; \Gamma_L = 0 \quad\text{(matched)}",
                    font_size=26, color=GREEN),
            MathTex(r"Z_L = 0 \;\Rightarrow\; \Gamma_L = -1 \quad\text{(short)}",
                    font_size=26, color=RED),
            MathTex(r"Z_L \to \infty \;\Rightarrow\; \Gamma_L = +1 \quad\text{(open)}",
                    font_size=26, color=YELLOW),
        ).arrange(DOWN, buff=0.3).move_to([0, -2.2, 0])

        self.play(FadeIn(cases, shift=UP * 0.15))
        self.wait(2)
        self.play(FadeOut(VGroup(derive1, derive2, box, cases, self.tline_diagram)))

        # Γ-plane mapping: sweep Z_L and trace Γ in the unit disk
        map_title = Text("Z_L sweep  →  Γ in the unit disk",
                         font_size=24, color=GRAY).next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(map_title))

        ax_zl = Axes(
            x_range=[0, 4, 1], y_range=[-2, 2, 1],
            x_length=5.2, y_length=4.2, tips=False,
            axis_config={"stroke_width": 2},
        ).move_to([-3.4, -0.6, 0])
        ax_g = Axes(
            x_range=[-1.2, 1.2, 0.5], y_range=[-1.2, 1.2, 0.5],
            x_length=4.2, y_length=4.2, tips=False,
            axis_config={"stroke_width": 2},
        ).move_to([3.0, -0.6, 0])

        zl_title = MathTex(r"Z_L/Z_0\ \text{plane}", font_size=22).next_to(ax_zl, UP, buff=0.05)
        g_title  = MathTex(r"\Gamma\ \text{plane}",    font_size=22).next_to(ax_g,  UP, buff=0.05)
        unit_circ = Circle(radius=ax_g.c2p(1, 0)[0] - ax_g.c2p(0, 0)[0],
                           color=GRAY, stroke_width=1.5).move_to(ax_g.c2p(0, 0))

        # Trackers: ZL = r + j x (normalized)
        r_t = ValueTracker(1.0)
        x_t = ValueTracker(0.0)

        def gamma_of(r, x):
            zl = complex(r, x)
            return (zl - 1) / (zl + 1)

        zl_dot = always_redraw(lambda: Dot(
            ax_zl.c2p(r_t.get_value(), x_t.get_value()),
            color=BLUE, radius=0.08,
        ))
        g_dot = always_redraw(lambda: Dot(
            ax_g.c2p(gamma_of(r_t.get_value(), x_t.get_value()).real,
                     gamma_of(r_t.get_value(), x_t.get_value()).imag),
            color=YELLOW, radius=0.08,
        ))
        readout = always_redraw(lambda: MathTex(
            rf"Z_L/Z_0={r_t.get_value():.2f}"
            rf"{'+' if x_t.get_value()>=0 else '-'}j{abs(x_t.get_value()):.2f},\ "
            rf"|\Gamma|={abs(gamma_of(r_t.get_value(), x_t.get_value())):.2f}",
            font_size=22, color=YELLOW
        ).to_edge(DOWN, buff=0.3))

        self.play(Create(ax_zl), Create(ax_g),
                  FadeIn(zl_title), FadeIn(g_title), Create(unit_circ))
        self.add(zl_dot, g_dot, readout)

        # Sweep real part (short → matched → open)
        self.play(r_t.animate.set_value(0.01), run_time=1.5)
        self.play(r_t.animate.set_value(1.0), run_time=1.5)
        self.play(r_t.animate.set_value(3.5), run_time=1.5)
        self.play(r_t.animate.set_value(1.0), run_time=1)
        # Sweep reactive part
        self.play(x_t.animate.set_value(2.0), run_time=1.5)
        self.play(x_t.animate.set_value(-2.0), run_time=2)
        self.play(x_t.animate.set_value(0.0), run_time=1)

        for m in (zl_dot, g_dot, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, map_title, ax_zl, ax_g,
                                 zl_title, g_title, unit_circ,
                                 zl_dot, g_dot, readout)))
        self.tline_diagram = None

    def _draw_tline_with_load(self):
        """Schematic: source — Z0 line — ZL load, drawn at top of frame."""
        y = 2.2
        # Wires
        top_wire = Line([-4.5, y, 0], [3.5, y, 0], color=WHITE, stroke_width=2)
        bot_wire = Line([-4.5, y - 0.9, 0], [3.5, y - 0.9, 0],
                        color=WHITE, stroke_width=2)

        # Z0 label along the line
        z0_label = MathTex(r"Z_0", font_size=26, color=GRAY).move_to([0, y + 0.35, 0])
        arrow_l = Arrow([-4.5, y + 0.05, 0], [0, y + 0.05, 0],
                        buff=0, color=BLUE, stroke_width=1.5,
                        tip_length=0.18, max_tip_length_to_length_ratio=0.15)
        arrow_r = Arrow([0, y - 0.95, 0], [-4.5, y - 0.95, 0],
                        buff=0, color=RED, stroke_width=1.5,
                        tip_length=0.18, max_tip_length_to_length_ratio=0.15)
        fwd_tex = MathTex(r"V^+", font_size=22, color=BLUE).next_to(arrow_l, UP, buff=0.05)
        bwd_tex = MathTex(r"V^-", font_size=22, color=RED ).next_to(arrow_r, DOWN, buff=0.05)

        # Load box
        load_rect = Rectangle(width=0.7, height=0.9, color=WHITE, stroke_width=2)
        load_rect.move_to([3.5, y - 0.45, 0])
        load_tex = MathTex(r"Z_L", font_size=26).move_to(load_rect)

        # z = 0 label
        z0_tick = Line([3.5, y + 0.15, 0], [3.5, y - 1.05, 0],
                       color=GRAY, stroke_width=1.5)
        z0_pos  = MathTex(r"z{=}0", font_size=20, color=GRAY).move_to([3.5, y - 1.3, 0])

        diagram = VGroup(top_wire, bot_wire, z0_label, arrow_l, arrow_r,
                         fwd_tex, bwd_tex, load_rect, load_tex, z0_tick, z0_pos)
        self.play(Create(diagram), run_time=1.5)
        self.tline_diagram = diagram

    # ── Act 3 — Input Impedance ───────────────────────────────────────────
    def act3_input_impedance(self):
        title = Text("Input Impedance", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # General reflection at distance l from load
        gamma_l = MathTex(
            r"\Gamma(z) \;=\; \Gamma_L\, e^{+2\gamma z}",
            font_size=36
        ).move_to([0, 1.6, 0])

        note = Text("at distance  ℓ  from the load  (z = −ℓ):",
                    font_size=24, color=GRAY).move_to([0, 0.7, 0])

        z_in = MathTex(
            r"Z_{\mathrm{in}} \;=\; Z_0\;\frac{Z_L + j Z_0 \tan\beta\ell}{"
            r"Z_0 + j Z_L \tan\beta\ell}",
            font_size=38
        ).move_to([0, -0.3, 0])
        box_zin = SurroundingRectangle(z_in, color=BLUE, buff=0.2, corner_radius=0.1)

        self.play(Write(gamma_l))
        self.play(Write(note))

        # Derivation steps: V(z), I(z) ratio → Z_in
        d1 = MathTex(
            r"Z_{\mathrm{in}}(-\ell) = \frac{V(-\ell)}{I(-\ell)} "
            r"= Z_0\,\frac{1+\Gamma_L e^{-2\gamma\ell}}{1-\Gamma_L e^{-2\gamma\ell}}",
            font_size=26,
        ).move_to([0, 0.2, 0])
        self.play(Write(d1))
        self.wait(0.8)
        self.play(FadeOut(d1))

        self.play(Write(z_in))
        self.play(Create(box_zin))
        self.wait(1)

        # Quarter-wave transformer
        qw = MathTex(
            r"\ell = \frac{\lambda}{4} \;\Rightarrow\; "
            r"Z_{\mathrm{in}} = \frac{Z_0^2}{Z_L}",
            font_size=32, color=YELLOW
        ).move_to([0, -1.8, 0])
        self.play(FadeIn(qw, shift=UP * 0.15))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, gamma_l, note, z_in, box_zin, qw)))

    # ── Act 4 — Animated Standing Waves & VSWR ────────────────────────────
    def act4_standing_waves(self):
        GAMMA = 0.5
        XMIN, XMAX = -2 * PI, 0.0

        title = Text("Standing Waves & VSWR", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── Three stacked panels ─────────────────────────────────────────
        ax_cfg = dict(
            x_range=[XMIN, XMAX + 0.01, PI / 2],
            y_range=[-1.7, 1.7, 1],
            x_length=10,
            y_length=1.7,
            axis_config={"color": GRAY, "include_numbers": False,
                         "stroke_width": 1.5},
            tips=False,
        )
        ax_inc = Axes(**ax_cfg)
        ax_ref = Axes(**ax_cfg)
        ax_tot = Axes(**ax_cfg)

        panel = VGroup(ax_inc, ax_ref, ax_tot).arrange(DOWN, buff=0.35)
        panel.move_to(ORIGIN)

        # Row labels
        lbl_inc = MathTex(r"V^+", font_size=26, color=BLUE)\
            .next_to(ax_inc, LEFT, buff=0.25)
        lbl_ref = MathTex(r"\Gamma\, V^-", font_size=26, color=RED)\
            .next_to(ax_ref, LEFT, buff=0.25)
        lbl_tot = MathTex(r"V_{\rm tot}", font_size=26, color=YELLOW)\
            .next_to(ax_tot, LEFT, buff=0.25)

        # Load boundary (z = 0) — dashed vertical on each panel
        def load_line(ax):
            return DashedLine(
                ax.c2p(0, -1.5), ax.c2p(0, 1.5),
                color=GRAY, stroke_width=1.2, dash_length=0.12,
            )
        load_lines = VGroup(*[load_line(ax) for ax in (ax_inc, ax_ref, ax_tot)])
        z0_lbl = MathTex(r"z{=}0", font_size=18, color=GRAY)\
            .next_to(load_lines[0], UP, buff=0.08)

        self.play(
            Create(panel),
            FadeIn(VGroup(lbl_inc, lbl_ref, lbl_tot, load_lines, z0_lbl)),
        )

        # ── Time-domain curves (always_redraw) ───────────────────────────
        t = ValueTracker(0)

        inc_curve = always_redraw(lambda: ax_inc.plot(
            lambda z: np.cos(t.get_value() - z),
            x_range=[XMIN, XMAX, 0.07],
            color=BLUE, stroke_width=2.5,
        ))
        ref_curve = always_redraw(lambda: ax_ref.plot(
            lambda z: GAMMA * np.cos(t.get_value() + z),
            x_range=[XMIN, XMAX, 0.07],
            color=RED, stroke_width=2.5,
        ))
        tot_curve = always_redraw(lambda: ax_tot.plot(
            lambda z: np.cos(t.get_value() - z) + GAMMA * np.cos(t.get_value() + z),
            x_range=[XMIN, XMAX, 0.07],
            color=YELLOW, stroke_width=2.5,
        ))

        # ── Phase 1: incident wave alone ─────────────────────────────────
        inc_dir = Text("→  propagates in +z", font_size=18, color=BLUE)\
            .next_to(ax_inc, RIGHT, buff=0.15)
        self.add(inc_curve)
        self.play(FadeIn(inc_dir))
        self.play(t.animate.set_value(2 * PI), run_time=4, rate_func=linear)
        self.play(FadeOut(inc_dir))

        # ── Phase 2: reflected wave appears ──────────────────────────────
        ref_dir = Text("←  propagates in −z", font_size=18, color=RED)\
            .next_to(ax_ref, RIGHT, buff=0.15)
        self.add(ref_curve)
        self.play(FadeIn(ref_dir))
        self.play(t.animate.set_value(4 * PI), run_time=4, rate_func=linear)
        self.play(FadeOut(ref_dir))

        # ── Phase 3: superposition ────────────────────────────────────────
        sum_note = Text("sum of both waves", font_size=18, color=YELLOW)\
            .next_to(ax_tot, RIGHT, buff=0.15)
        self.add(tot_curve)
        self.play(FadeIn(sum_note))
        self.play(t.animate.set_value(6 * PI), run_time=4, rate_func=linear)
        self.play(FadeOut(sum_note))

        # ── Phase 4: freeze and draw VSWR envelope ───────────────────────
        # Envelope of V_tot(z,t) = cos(ωt-z) + Γcos(ωt+z)
        # Amplitude at position z: sqrt((1+Γ)²cos²z + (1-Γ)²sin²z)
        def envelope(z):
            return np.sqrt(
                (1 + GAMMA) ** 2 * np.cos(z) ** 2
                + (1 - GAMMA) ** 2 * np.sin(z) ** 2
            )

        env_upper = DashedVMobject(
            ax_tot.plot(lambda z:  envelope(z), x_range=[XMIN, XMAX, 0.07],
                        color=GREEN, stroke_width=2.2),
            num_dashes=28,
        )
        env_lower = DashedVMobject(
            ax_tot.plot(lambda z: -envelope(z), x_range=[XMIN, XMAX, 0.07],
                        color=GREEN, stroke_width=2.2),
            num_dashes=28,
        )

        vmax_lbl = MathTex(r"1+|\Gamma|", font_size=20, color=GREEN)\
            .next_to(ax_tot, RIGHT, buff=0.12).shift(UP * 0.55)
        vmin_lbl = MathTex(r"1-|\Gamma|", font_size=20, color=GREEN)\
            .next_to(ax_tot, RIGHT, buff=0.12).shift(DOWN * 0.3)

        self.play(Create(env_upper), Create(env_lower))
        self.play(FadeIn(vmax_lbl), FadeIn(vmin_lbl))
        self.wait(0.8)

        vswr_eq = MathTex(
            r"\mathrm{VSWR} \;=\; \frac{V_{\max}}{V_{\min}} \;=\; "
            r"\frac{1+|\Gamma|}{1-|\Gamma|}",
            font_size=30, color=GREEN,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(vswr_eq))
        self.wait(3)

        # Clear — stop updaters before FadeOut
        inc_curve.clear_updaters()
        ref_curve.clear_updaters()
        tot_curve.clear_updaters()
        self.play(FadeOut(VGroup(
            title, panel, lbl_inc, lbl_ref, lbl_tot, load_lines, z0_lbl,
            inc_curve, ref_curve, tot_curve,
            env_upper, env_lower, vmax_lbl, vmin_lbl, vswr_eq,
        )))

    # ── Act 5 — Power Flow ────────────────────────────────────────────────
    def act5_power_flow(self):
        title = Text("Power Flow on a Transmission Line",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Time-average power
        p_eq = MathTex(
            r"P_{\mathrm{avg}} \;=\; \frac{|V^+|^2}{2 Z_0}"
            r"\!\left(1 - |\Gamma|^2\right)",
            font_size=44
        ).move_to([0, 0.9, 0])
        box_p = SurroundingRectangle(p_eq, color=BLUE, buff=0.2, corner_radius=0.1)

        self.play(Write(p_eq))
        self.play(Create(box_p))
        self.wait(1)

        # Break-down annotation
        avail = MathTex(
            r"\underbrace{\frac{|V^+|^2}{2 Z_0}}_{\text{available power}}",
            font_size=32, color=GREEN
        ).move_to([-1.8, -0.8, 0])
        mismatch = MathTex(
            r"\underbrace{(1 - |\Gamma|^2)}_{\text{mismatch factor}}",
            font_size=32, color=RED
        ).move_to([1.8, -0.8, 0])

        self.play(FadeIn(avail), FadeIn(mismatch))
        self.wait(1)

        # Insight: Γ = 0 → all power delivered; |Γ| = 1 → none
        insight = MathTex(
            r"|\Gamma| = 0 \;\Rightarrow\; P = P_{\mathrm{avs}}, \qquad"
            r"|\Gamma| = 1 \;\Rightarrow\; P = 0",
            font_size=28, color=YELLOW
        ).move_to([0, -2.2, 0])

        self.play(Write(insight))
        self.wait(1.5)
        self.play(FadeOut(VGroup(p_eq, box_p, avail, mismatch, insight)))

        # P/P_avs vs |Γ| — parabola with moving dot
        ax = Axes(
            x_range=[0, 1.05, 0.25], y_range=[0, 1.1, 0.25],
            x_length=8, y_length=3.6, tips=False,
            axis_config={"stroke_width": 2, "include_numbers": True,
                         "decimal_number_config": {"num_decimal_places": 2}},
        ).move_to([0, -0.5, 0])
        xlab = MathTex(r"|\Gamma|",            font_size=26).next_to(ax, DOWN, buff=0.15)
        ylab = MathTex(r"P/P_{\mathrm{avs}}",  font_size=26).next_to(ax, LEFT, buff=0.15)

        curve = ax.plot(lambda g: 1 - g ** 2, x_range=[0, 1, 0.01],
                        color=BLUE, stroke_width=3)

        g_t = ValueTracker(0.0)
        dot = always_redraw(lambda: Dot(
            ax.c2p(g_t.get_value(), 1 - g_t.get_value() ** 2),
            color=YELLOW, radius=0.08,
        ))
        vline = always_redraw(lambda: DashedLine(
            ax.c2p(g_t.get_value(), 0),
            ax.c2p(g_t.get_value(), 1 - g_t.get_value() ** 2),
            color=GRAY, stroke_width=1.5,
        ))
        readout = always_redraw(lambda: MathTex(
            rf"|\Gamma|={g_t.get_value():.2f},\ "
            rf"P/P_{{\rm avs}}={1 - g_t.get_value()**2:.2f}",
            font_size=24, color=YELLOW
        ).to_edge(DOWN, buff=0.3))

        self.play(Create(ax), FadeIn(xlab), FadeIn(ylab))
        self.play(Create(curve))
        self.add(dot, vline, readout)
        self.play(g_t.animate.set_value(1.0), run_time=3.5, rate_func=linear)
        self.play(g_t.animate.set_value(0.0), run_time=2, rate_func=linear)

        for m in (dot, vline, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, ax, xlab, ylab, curve,
                                 dot, vline, readout)))
