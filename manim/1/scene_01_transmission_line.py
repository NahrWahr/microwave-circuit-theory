from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def component_box(label: str, color=WHITE, width=0.9, height=0.38, font_size=22):
    """Labelled rectangle representing a lumped element."""
    rect = Rectangle(width=width, height=height, color=color, stroke_width=2)
    tex = MathTex(label, font_size=font_size, color=color)
    tex.move_to(rect)
    return VGroup(rect, tex)


def ground_symbol(tip: np.ndarray, color=WHITE) -> VGroup:
    """Three-line ground symbol with tip at the given point."""
    lines = VGroup(
        Line(tip, tip + DOWN * 0.25, color=color, stroke_width=2),
        Line(tip + DOWN * 0.25 + LEFT * 0.25, tip + DOWN * 0.25 + RIGHT * 0.25,
             color=color, stroke_width=2),
        Line(tip + DOWN * 0.45 + LEFT * 0.15, tip + DOWN * 0.45 + RIGHT * 0.15,
             color=color, stroke_width=2),
        Line(tip + DOWN * 0.60 + LEFT * 0.06, tip + DOWN * 0.60 + RIGHT * 0.06,
             color=color, stroke_width=2),
    )
    return lines


def rlcg_section(x_center=0.0, y=0.0, width=3.2,
                 show_labels=True, font_size=20, color=WHITE) -> VGroup:
    """
    One RLCG lumped-element section centred at (x_center, y).

    Layout (top-down view of one two-wire section):

      ──[R·Δz]──[L·Δz]──────────
                         |
                      [G·Δz]
                      [C·Δz]
                         |
                        ─┴─ (ground)
    """
    g = VGroup()

    # ── geometry constants ────────────────────────────────────────────────
    box_w, box_h = 0.82, 0.36
    half = width / 2.0

    # x positions of key points along the top wire
    x_left   = x_center - half
    x_r      = x_center - half + width * 0.22   # centre of R box
    x_l      = x_center - half + width * 0.52   # centre of L box
    x_shunt  = x_center - half + width * 0.80   # shunt junction
    x_right  = x_center + half

    # ── top wire segments ─────────────────────────────────────────────────
    g.add(Line([x_left,  y, 0], [x_r - box_w/2, y, 0],
               color=color, stroke_width=2))
    g.add(Line([x_r + box_w/2, y, 0], [x_l - box_w/2, y, 0],
               color=color, stroke_width=2))
    g.add(Line([x_l + box_w/2, y, 0], [x_shunt, y, 0],
               color=color, stroke_width=2))
    g.add(Line([x_shunt, y, 0], [x_right, y, 0],
               color=color, stroke_width=2))

    # ── series components ─────────────────────────────────────────────────
    r_box = component_box(r"R\Delta z", color=BLUE,
                          width=box_w, height=box_h, font_size=font_size)
    r_box.move_to([x_r, y, 0])

    l_box = component_box(r"L\Delta z", color=GREEN,
                          width=box_w, height=box_h, font_size=font_size)
    l_box.move_to([x_l, y, 0])

    g.add(r_box, l_box)

    # ── shunt branch ──────────────────────────────────────────────────────
    y_g = y - 0.55   # centre of G box
    y_c = y - 1.15   # centre of C box
    y_gnd = y - 1.60 # tip of ground symbol

    g.add(Line([x_shunt, y, 0],       [x_shunt, y_g + box_h/2, 0],
               color=color, stroke_width=2))
    g.add(Line([x_shunt, y_g - box_h/2, 0], [x_shunt, y_c + box_h/2, 0],
               color=color, stroke_width=2))
    g.add(Line([x_shunt, y_c - box_h/2, 0], [x_shunt, y_gnd, 0],
               color=color, stroke_width=2))

    gnd_box = component_box(r"G\Delta z", color=YELLOW,
                            width=box_w, height=box_h, font_size=font_size)
    gnd_box.move_to([x_shunt, y_g, 0])

    cap_box = component_box(r"C\Delta z", color=RED,
                            width=box_w, height=box_h, font_size=font_size)
    cap_box.move_to([x_shunt, y_c, 0])

    g.add(gnd_box, cap_box)
    g.add(ground_symbol(np.array([x_shunt, y_gnd, 0]), color=color))

    # ── Δz dimension brace (optional, used only for first section) ────────
    if show_labels:
        brace = Brace(
            Line([x_left, y + 0.05, 0], [x_right, y + 0.05, 0]),
            direction=UP, buff=0.05, color=GRAY
        )
        dz = MathTex(r"\Delta z", font_size=22, color=GRAY)
        dz.next_to(brace, UP, buff=0.05)
        g.add(brace, dz)

    return g


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class TransmissionLine(Scene):

    def construct(self):
        self.act1_title()
        self.act2_single_section()
        self.act3_cascade()
        self.act4_telegrapher_eqs()
        self.act4b_wave_equation_derivation()
        self.act5_wave_solution()
        self.act6_char_impedance()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Transmission Line Theory", font_size=44)
        sub = Text("from lumped elements to wave equations",
                   font_size=26, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2 ─────────────────────────────────────────────────────────────
    def act2_single_section(self):
        title = Text("The Lumped-Element Model",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        sec = rlcg_section(x_center=0.5, y=0.8, width=5.0, show_labels=True)
        self.play(Create(sec), run_time=2)

        # Annotation callouts
        desc_r = Text("wire resistance", font_size=18, color=BLUE)
        desc_l = Text("wire inductance", font_size=18, color=GREEN)
        desc_g = Text("dielectric loss",  font_size=18, color=YELLOW)
        desc_c = Text("capacitance",      font_size=18, color=RED)

        desc_r.move_to([-3.0, -1.8, 0])
        desc_l.move_to([-0.8, -1.8, 0])
        desc_g.move_to([ 1.4, -1.8, 0])
        desc_c.move_to([ 3.4, -1.8, 0])

        self.play(FadeIn(VGroup(desc_r, desc_l, desc_g, desc_c)))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, sec, desc_r, desc_l, desc_g, desc_c)))

    # ── Act 3 ─────────────────────────────────────────────────────────────
    def act3_cascade(self):
        title = Text("Distributed Model: cascade of sections",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        n = 3
        sec_w = 2.6
        total = n * sec_w
        y_top = 1.0

        sections = VGroup()
        for i in range(n):
            xc = -total / 2 + sec_w / 2 + i * sec_w
            s = rlcg_section(x_center=xc, y=y_top, width=sec_w,
                             show_labels=(i == 1), font_size=18)
            sections.add(s)

        dots_l = MathTex(r"\cdots", font_size=36).move_to([-total/2 - 0.5, y_top, 0])
        dots_r = MathTex(r"\cdots", font_size=36).move_to([ total/2 + 0.5, y_top, 0])

        self.play(Create(sections), run_time=2.5)
        self.play(FadeIn(VGroup(dots_l, dots_r)))
        self.wait(1)

        # Continuum limit
        limit = MathTex(r"\Delta z \to 0 \quad\Longrightarrow\quad \text{continuous line}",
                        font_size=34, color=YELLOW)
        limit.move_to([0, -1.6, 0])
        self.play(Write(limit))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, sections, dots_l, dots_r, limit)))

    # ── Act 4 ─────────────────────────────────────────────────────────────
    def act4_telegrapher_eqs(self):
        title = Text("Telegrapher's Equations",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # KVL derivation step
        kvl_head = Text("KVL around one section:", font_size=24, color=GRAY)
        kvl_head.move_to([-2.0, 1.8, 0])
        kvl_raw = MathTex(
            r"V(z+\Delta z) - V(z) = -(R\,\Delta z)\,I - (j\omega L\,\Delta z)\,I",
            font_size=26
        ).move_to([0, 1.1, 0])

        kcl_head = Text("KCL at the shunt node:", font_size=24, color=GRAY)
        kcl_head.move_to([-2.1, 0.3, 0])
        kcl_raw = MathTex(
            r"I(z+\Delta z) - I(z) = -(G\,\Delta z)\,V - (j\omega C\,\Delta z)\,V",
            font_size=26
        ).move_to([0, -0.4, 0])

        self.play(Write(kvl_head))
        self.play(Write(kvl_raw))
        self.play(Write(kcl_head))
        self.play(Write(kcl_raw))
        self.wait(1)

        arrow = Text("Divide by Δz, take Δz → 0 :", font_size=24, color=YELLOW)
        arrow.move_to([0, -1.3, 0])
        self.play(Write(arrow))
        self.wait(0.8)
        self.play(FadeOut(VGroup(kvl_head, kvl_raw, kcl_head, kcl_raw, arrow)))

        # Final boxed equations
        eq1 = MathTex(
            r"\frac{\partial V}{\partial z} = -(R + j\omega L)\,I",
            font_size=42
        ).move_to([0, 0.9, 0])
        eq2 = MathTex(
            r"\frac{\partial I}{\partial z} = -(G + j\omega C)\,V",
            font_size=42
        ).move_to([0, -0.6, 0])

        box1 = SurroundingRectangle(eq1, color=BLUE,  buff=0.18, corner_radius=0.1)
        box2 = SurroundingRectangle(eq2, color=GREEN, buff=0.18, corner_radius=0.1)

        self.play(Write(eq1), Write(eq2))
        self.play(Create(box1), Create(box2))
        self.wait(3)
        self.play(FadeOut(VGroup(title, eq1, eq2, box1, box2)))

    # ── Act 4b ────────────────────────────────────────────────────────────
    def act4b_wave_equation_derivation(self):
        title = Text("From Coupled PDEs to the Wave Equation",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        step1 = MathTex(
            r"\frac{\partial V}{\partial z} = -(R+j\omega L)\,I",
            font_size=32
        ).move_to([0, 1.7, 0])
        step2 = MathTex(
            r"\frac{\partial}{\partial z}\!\Big[\tfrac{\partial V}{\partial z}\Big] "
            r"= -(R+j\omega L)\,\frac{\partial I}{\partial z}",
            font_size=32
        ).move_to([0, 0.8, 0])
        step3 = MathTex(
            r"= -(R+j\omega L)\big[-(G+j\omega C)\,V\big]",
            font_size=32
        ).move_to([0, -0.1, 0])
        step4 = MathTex(
            r"\boxed{\;\frac{\partial^2 V}{\partial z^2} = \gamma^2\, V,\quad"
            r"\gamma^2 = (R+j\omega L)(G+j\omega C)\;}",
            font_size=34, color=YELLOW
        ).move_to([0, -1.3, 0])

        self.play(Write(step1))
        self.play(Write(step2))
        self.play(Write(step3))
        self.play(Write(step4))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, step1, step2, step3, step4)))

    # ── Act 5 ─────────────────────────────────────────────────────────────
    def act5_wave_solution(self):
        title = Text("Traveling Waves  V(z,t) = e^{-αz} cos(ωt − βz)",
                     font_size=28).to_edge(UP, buff=0.3)
        self.play(Write(title))

        gamma_def = MathTex(
            r"\gamma = \alpha + j\beta,\quad "
            r"V(z,t)=V^{+}e^{-\alpha z}\cos(\omega t-\beta z)+V^{-}e^{+\alpha z}\cos(\omega t+\beta z)",
            font_size=24, color=GRAY
        ).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(gamma_def))

        # Trackers
        alpha = ValueTracker(0.0)
        beta  = ValueTracker(1.2)
        t     = ValueTracker(0.0)

        axes = Axes(
            x_range=[0, 6, 1], y_range=[-1.5, 1.5, 1],
            x_length=10, y_length=3.2,
            tips=False,
            axis_config={"stroke_width": 2, "include_numbers": False},
        ).move_to([0, -0.4, 0])
        x_lbl = MathTex("z", font_size=26).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_lbl = MathTex("V", font_size=26).next_to(axes.y_axis, UP, buff=0.1)
        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl))

        omega = 2.0

        fwd = always_redraw(lambda: axes.plot(
            lambda z: np.exp(-alpha.get_value() * z) *
                      np.cos(omega * t.get_value() - beta.get_value() * z),
            x_range=[0, 6, 0.02], color=BLUE, stroke_width=3,
        ))
        bwd = always_redraw(lambda: axes.plot(
            lambda z: 0.5 * np.exp(+alpha.get_value() * z) *
                      np.cos(omega * t.get_value() + beta.get_value() * z),
            x_range=[0, 6, 0.02], color=RED, stroke_width=2,
        ))
        env_p = always_redraw(lambda: axes.plot(
            lambda z: np.exp(-alpha.get_value() * z),
            x_range=[0, 6, 0.02], color=BLUE_E, stroke_width=1.5,
        ))
        env_n = always_redraw(lambda: axes.plot(
            lambda z: -np.exp(-alpha.get_value() * z),
            x_range=[0, 6, 0.02], color=BLUE_E, stroke_width=1.5,
        ))

        self.add(env_p, env_n, fwd, bwd)

        readout = always_redraw(lambda: MathTex(
            rf"\alpha={alpha.get_value():.2f},\ \beta={beta.get_value():.2f}",
            font_size=26, color=YELLOW
        ).to_corner(DR, buff=0.4))
        self.add(readout)

        # Phase 1: lossless propagation (animate time)
        self.play(t.animate.set_value(2 * PI / omega), run_time=3, rate_func=linear)
        # Phase 2: crank up attenuation
        self.play(alpha.animate.set_value(0.35), run_time=2)
        self.play(t.animate.set_value(4 * PI / omega), run_time=3, rate_func=linear)
        # Phase 3: change beta (wavelength)
        self.play(beta.animate.set_value(2.4), run_time=2)
        self.play(t.animate.set_value(6 * PI / omega), run_time=3, rate_func=linear)

        for m in (fwd, bwd, env_p, env_n, readout):
            m.clear_updaters()
        self.wait(0.5)
        self.play(FadeOut(VGroup(title, gamma_def, axes, x_lbl, y_lbl,
                                 fwd, bwd, env_p, env_n, readout)))

    # ── Act 6 ─────────────────────────────────────────────────────────────
    def act6_char_impedance(self):
        title = Text("Characteristic Impedance  Z₀",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # General Z0
        z0_eq = MathTex(
            r"Z_0 \;=\; \frac{V^+}{I^+} \;=\; "
            r"\sqrt{\frac{R + j\omega L}{G + j\omega C}}",
            font_size=42
        ).move_to([0, 0.9, 0])

        box_z0 = SurroundingRectangle(z0_eq, color=BLUE, buff=0.18, corner_radius=0.1)

        self.play(Write(z0_eq))
        self.play(Create(box_z0))
        self.wait(1)

        # Lossless special case
        lossless_note = Text("Lossless line  (R = G = 0):",
                             font_size=26, color=GRAY).move_to([-1.5, -0.5, 0])
        z0_lossless = MathTex(
            r"Z_0 = \sqrt{\dfrac{L}{C}}",
            font_size=42, color=YELLOW
        ).move_to([0, -1.5, 0])

        box_ll = SurroundingRectangle(z0_lossless, color=YELLOW, buff=0.18, corner_radius=0.1)

        self.play(Write(lossless_note))
        self.play(Write(z0_lossless))
        self.play(Create(box_ll))
        self.wait(1.5)
        self.play(FadeOut(VGroup(z0_eq, box_z0, lossless_note,
                                 z0_lossless, box_ll)))

        # Z0 vs frequency: |Z0| and arg(Z0) transition to real √(L/C) at high f
        subtitle = Text("|Z₀(ω)| and ∠Z₀(ω): lossy line → real √(L/C) at high ω",
                        font_size=22, color=GRAY).next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(subtitle))

        R_, L_, G_, C_ = 2.0, 1.0, 0.02, 1.0
        z0_inf = np.sqrt(L_ / C_)

        def z0_of(w):
            num = complex(R_, w * L_)
            den = complex(G_, w * C_)
            return np.sqrt(num / den)

        ax_mag = Axes(
            x_range=[-1, 2.2, 1], y_range=[0.0, 2.5, 1],
            x_length=5.6, y_length=2.6, tips=False,
            axis_config={"stroke_width": 2},
        ).move_to([-3.2, -1.3, 0])
        ax_phase = Axes(
            x_range=[-1, 2.2, 1], y_range=[-50, 10, 15],
            x_length=5.6, y_length=2.6, tips=False,
            axis_config={"stroke_width": 2},
        ).move_to([3.2, -1.3, 0])

        mag_lbl = MathTex(r"|Z_0|", font_size=24).next_to(ax_mag, UP, buff=0.05)
        ph_lbl  = MathTex(r"\angle Z_0\ (^\circ)", font_size=24).next_to(ax_phase, UP, buff=0.05)
        wx_lbl1 = MathTex(r"\log_{10}\omega", font_size=20).next_to(ax_mag, DOWN, buff=0.1)
        wx_lbl2 = MathTex(r"\log_{10}\omega", font_size=20).next_to(ax_phase, DOWN, buff=0.1)

        mag_curve = ax_mag.plot(
            lambda x: float(np.abs(z0_of(10 ** x))),
            x_range=[-1, 2.2, 0.02], color=BLUE, stroke_width=3,
        )
        asymptote = DashedLine(
            ax_mag.c2p(-1, z0_inf), ax_mag.c2p(2.2, z0_inf),
            color=YELLOW, stroke_width=2,
        )
        asymp_lbl = MathTex(r"\sqrt{L/C}", font_size=22, color=YELLOW).next_to(
            ax_mag.c2p(2.2, z0_inf), LEFT, buff=0.1)

        ph_curve = ax_phase.plot(
            lambda x: float(np.degrees(np.angle(z0_of(10 ** x)))),
            x_range=[-1, 2.2, 0.02], color=GREEN, stroke_width=3,
        )
        zero_ph = DashedLine(
            ax_phase.c2p(-1, 0), ax_phase.c2p(2.2, 0),
            color=YELLOW, stroke_width=1.5,
        )

        self.play(Create(ax_mag), Create(ax_phase),
                  FadeIn(mag_lbl), FadeIn(ph_lbl),
                  FadeIn(wx_lbl1), FadeIn(wx_lbl2))
        self.play(Create(mag_curve), Create(ph_curve))
        self.play(Create(asymptote), FadeIn(asymp_lbl), Create(zero_ph))
        self.wait(3)
        self.play(FadeOut(VGroup(title, subtitle, ax_mag, ax_phase,
                                 mag_lbl, ph_lbl, wx_lbl1, wx_lbl2,
                                 mag_curve, ph_curve, asymptote, asymp_lbl, zero_ph)))
