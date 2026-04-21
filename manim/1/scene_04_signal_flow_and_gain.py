from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sfg_node(pos, label, label_dir=UP, color=WHITE, r=0.18, font_size=24):
    dot = Dot(pos, radius=r, color=color, fill_opacity=1)
    lbl = MathTex(label, font_size=font_size, color=color)
    lbl.next_to(dot, label_dir, buff=0.14)
    return VGroup(dot, lbl)


def sfg_arrow(start, end, label, label_side=UP, color=WHITE, font_size=20):
    arr = Arrow(start, end, buff=0.22, tip_length=0.18,
                max_tip_length_to_length_ratio=0.35,
                color=color, stroke_width=2)
    lbl = MathTex(label, font_size=font_size, color=color)
    lbl.next_to(arr.get_center(), label_side, buff=0.14)
    return VGroup(arr, lbl)


def block(center, label, color=WHITE, w=2.0, h=0.85):
    r = Rectangle(width=w, height=h, color=color, stroke_width=2)
    r.move_to(center)
    t = MathTex(label, font_size=24, color=color).move_to(r)
    return VGroup(r, t)


# ---------------------------------------------------------------------------
# Scene 4 — Signal Flow Graphs & Power Gain
# Topics 13–15
# ---------------------------------------------------------------------------

class SignalFlowAndGain(Scene):

    def construct(self):
        self.act1_title()
        self.act2_sfg_basics()
        self.act3_two_port_sfg()
        self.act4_masons_rule()
        self.act5_gain_definitions()
        self.act6_gain_expressions()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Signal Flow Graphs & Power Gain", font_size=40)
        sub = Text("from wave variables to gain definitions",
                   font_size=26, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2: SFG Basics ─────────────────────────────────────────────────
    def act2_sfg_basics(self):
        title = Text("Signal Flow Graphs", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        rules = VGroup(
            VGroup(
                Text("Node",         font_size=26, color=BLUE),
                Text(" — a wave variable  (a₁, b₂, …)", font_size=26),
            ).arrange(RIGHT, buff=0),
            VGroup(
                Text("Branch",       font_size=26, color=GREEN),
                Text(" — a directed gain connecting two nodes", font_size=26),
            ).arrange(RIGHT, buff=0),
            VGroup(
                Text("Mason's rule", font_size=26, color=YELLOW),
                Text(" — any ratio of nodes from first principles", font_size=26),
            ).arrange(RIGHT, buff=0),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to([0, 0.6, 0])

        for rule in rules:
            self.play(FadeIn(rule, shift=RIGHT * 0.2), run_time=0.6)
            self.wait(0.4)

        # Tiny demo: x → [G] → y = G·x
        n1 = sfg_node([-2.5, -2.0, 0], r"x", color=BLUE)
        n2 = sfg_node([ 2.5, -2.0, 0], r"y", color=BLUE)
        br = sfg_arrow(np.array([-2.5, -2.0, 0]),
                       np.array([ 2.5, -2.0, 0]),
                       r"G", color=GREEN)
        eq = MathTex(r"\Rightarrow\quad y = G\cdot x", font_size=28)\
            .move_to([5.5, -2.0, 0])

        self.play(Create(n1), Create(n2))
        self.play(Create(br))
        self.play(Write(eq))
        self.wait(2)
        self.play(FadeOut(VGroup(title, rules, n1, n2, br, eq)))

    # ── Act 3: Two-Port SFG ────────────────────────────────────────────────
    def act3_two_port_sfg(self):
        title = Text("Two-Port Signal Flow Graph",
                     font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Node positions — rectangular layout
        p_bs = np.array([-5.5,  0.5, 0])
        p_a1 = np.array([-2.0,  0.5, 0])
        p_b2 = np.array([ 2.0,  0.5, 0])
        p_b1 = np.array([-2.0, -2.0, 0])
        p_a2 = np.array([ 2.0, -2.0, 0])

        n_bs = sfg_node(p_bs, r"b_s", label_dir=UP,   color=GRAY,   font_size=22)
        n_a1 = sfg_node(p_a1, r"a_1", label_dir=UP,   font_size=22)
        n_b2 = sfg_node(p_b2, r"b_2", label_dir=UP,   font_size=22)
        n_b1 = sfg_node(p_b1, r"b_1", label_dir=DOWN, font_size=22)
        n_a2 = sfg_node(p_a2, r"a_2", label_dir=DOWN, font_size=22)
        nodes = VGroup(n_bs, n_a1, n_b2, n_b1, n_a2)
        self.play(LaggedStart(*[Create(n) for n in nodes], lag_ratio=0.18))

        # Source feed: b_s → a_1
        br_src = sfg_arrow(p_bs, p_a1, r"1", label_side=UP, color=GRAY, font_size=18)
        self.play(Create(br_src))

        # Main path: a_1 →[S21]→ b_2  (highlighted yellow)
        br_s21 = sfg_arrow(p_a1, p_b2, r"S_{21}", label_side=UP,
                           color=YELLOW, font_size=22)
        self.play(Create(br_s21))

        # S-matrix branches
        br_s11 = sfg_arrow(p_a1, p_b1, r"S_{11}", label_side=LEFT,  font_size=20)
        br_s22 = sfg_arrow(p_a2, p_b2, r"S_{22}", label_side=RIGHT, font_size=20)
        br_s12 = sfg_arrow(p_a2, p_b1, r"S_{12}", label_side=DOWN,  font_size=20)
        self.play(Create(br_s11), Create(br_s22), Create(br_s12))
        self.wait(0.5)

        # Feedback loops (curved, outward)
        # Γ_S: b_1 → a_1, curves LEFT (angle > 0 = left of upward direction)
        gs_arc = CurvedArrow(p_b1, p_a1, angle=PI / 2.0,
                             color=BLUE, stroke_width=2.2, tip_length=0.18)
        gs_lbl = MathTex(r"\Gamma_S", font_size=24, color=BLUE)\
            .move_to(np.array([-3.6, -0.75, 0]))

        # Γ_L: b_2 → a_2, curves RIGHT (angle > 0 = left of downward = rightward)
        gl_arc = CurvedArrow(p_b2, p_a2, angle=PI / 2.0,
                             color=RED, stroke_width=2.2, tip_length=0.18)
        gl_lbl = MathTex(r"\Gamma_L", font_size=24, color=RED)\
            .move_to(np.array([3.6, -0.75, 0]))

        self.play(
            Create(gs_arc), FadeIn(gs_lbl),
            Create(gl_arc), FadeIn(gl_lbl),
        )
        self.wait(1)

        # Pulse traversal — forward path then a feedback loop
        note = MathTex(r"\text{forward path: }b_s \to a_1 \to b_2",
                       font_size=28, color=YELLOW).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(note))

        pulse_fwd = Dot(p_bs, color=YELLOW, radius=0.12)
        self.play(FadeIn(pulse_fwd))
        self.play(pulse_fwd.animate.move_to(p_a1), run_time=1.0, rate_func=linear)
        self.play(pulse_fwd.animate.move_to(p_b2), run_time=1.2, rate_func=linear)
        self.play(FadeOut(pulse_fwd), FadeOut(note))

        note2 = MathTex(r"\text{loop: }a_1 \to b_1 \to \Gamma_S \to a_1",
                        font_size=28, color=BLUE).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(note2))
        pulse_loop = Dot(p_a1, color=BLUE, radius=0.12)
        self.play(FadeIn(pulse_loop))
        self.play(pulse_loop.animate.move_to(p_b1), run_time=0.9, rate_func=linear)
        self.play(MoveAlongPath(pulse_loop, gs_arc), run_time=1.4, rate_func=linear)
        self.play(FadeOut(pulse_loop), FadeOut(note2))

        note3 = MathTex(r"\text{loop: }b_2 \to \Gamma_L \to a_2 \to b_1 \to \Gamma_S \to a_1",
                        font_size=26, color=RED).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(note3))
        pulse_big = Dot(p_b2, color=RED, radius=0.12)
        self.play(FadeIn(pulse_big))
        self.play(MoveAlongPath(pulse_big, gl_arc), run_time=1.2, rate_func=linear)
        self.play(pulse_big.animate.move_to(p_b1), run_time=1.0, rate_func=linear)
        self.play(MoveAlongPath(pulse_big, gs_arc), run_time=1.2, rate_func=linear)
        self.play(FadeOut(pulse_big), FadeOut(note3))

        self.wait(0.5)
        self.play(FadeOut(VGroup(
            title, nodes, br_src, br_s21, br_s11, br_s22, br_s12,
            gs_arc, gs_lbl, gl_arc, gl_lbl,
        )))

    # ── Act 4: Mason's Rule (brief) ────────────────────────────────────────
    def act4_masons_rule(self):
        title = Text("Mason's Gain Rule", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        formula = MathTex(
            r"\frac{b_2}{b_s} \;=\; "
            r"\frac{\displaystyle\sum_k P_k\,\Delta_k}{\Delta}",
            font_size=44,
        ).move_to([0, 1.6, 0])
        self.play(Write(formula))
        self.wait(0.8)

        items = VGroup(
            MathTex(r"P_k \;=\;\text{gain of forward path }k",
                    font_size=28),
            MathTex(r"\Delta \;=\; 1 - \sum_i L_i + \sum_{i<j} L_i L_j - \cdots"
                    r"\quad\text{(graph determinant)}",
                    font_size=28),
            MathTex(r"\Delta_k \;=\; \Delta\text{ with all loops touching path }k"
                    r"\text{ removed}",
                    font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to([0.3, -0.3, 0])

        for item in items:
            self.play(FadeIn(item, shift=UP * 0.15), run_time=0.6)
            self.wait(0.5)

        loops = MathTex(
            r"\text{Two-port loops:}\;"
            r"L_1 = S_{11}\Gamma_S,\quad{}"
            r"L_2 = S_{22}\Gamma_L,\quad{}"
            r"L_3 = S_{21}\Gamma_L S_{12}\Gamma_S",
            font_size=26, color=YELLOW,
        ).move_to([0, -2.5, 0])
        self.play(FadeIn(loops))
        self.wait(2.5)
        self.play(FadeOut(VGroup(title, formula, items, loops)))

    # ── Act 5: Power Gain Definitions ──────────────────────────────────────
    def act5_gain_definitions(self):
        title = Text("Power Gain Definitions", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Source → Device → Load cascade
        src  = block([-4.5, 1.3, 0], r"Z_S",         color=GRAY, w=1.6, h=0.9)
        dut  = block([ 0.0, 1.3, 0], r"\text{Device}", color=WHITE, w=2.4, h=0.9)
        load = block([ 4.5, 1.3, 0], r"Z_L",         color=GRAY, w=1.6, h=0.9)
        wl = Line([-3.7, 1.3, 0], [-1.2, 1.3, 0], color=WHITE, stroke_width=2)
        wr = Line([ 1.2, 1.3, 0], [ 3.7, 1.3, 0], color=WHITE, stroke_width=2)
        diagram = VGroup(src, dut, load, wl, wr)
        self.play(Create(diagram))

        # Power labels
        pavs = MathTex(r"P_{\rm AVS}", font_size=22, color=GRAY)\
            .move_to([-4.5, 0.4, 0])
        pin  = MathTex(r"P_{\rm in}",  font_size=22, color=BLUE)\
            .move_to([-1.0, 0.4, 0])
        pavn = MathTex(r"P_{\rm AVN}", font_size=22, color=GREEN)\
            .move_to([ 1.2, 0.4, 0])
        pl   = MathTex(r"P_L",         font_size=22, color=RED)\
            .move_to([ 4.5, 0.4, 0])
        self.play(FadeIn(VGroup(pavs, pin, pavn, pl)))
        self.wait(0.5)

        # Three definitions
        defs = VGroup(
            MathTex(
                r"G_T = \frac{P_L}{P_{\rm AVS}}",
                r"\quad\longrightarrow\quad\text{transducer gain}",
                font_size=30, color=RED,
            ),
            MathTex(
                r"G_A = \frac{P_{\rm AVN}}{P_{\rm AVS}}",
                r"\quad\longrightarrow\quad\text{available gain}",
                font_size=30, color=GREEN,
            ),
            MathTex(
                r"G_P = \frac{P_L}{P_{\rm in}}",
                r"\quad\longrightarrow\quad\text{operating gain}",
                font_size=30, color=BLUE,
            ),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([0, -1.8, 0])

        for d in defs:
            self.play(FadeIn(d, shift=UP * 0.15), run_time=0.65)
            self.wait(0.6)

        self.wait(2)
        self.play(FadeOut(VGroup(title, diagram, pavs, pin, pavn, pl, defs)))

    # ── Act 6: Gain Expressions ────────────────────────────────────────────
    def act6_gain_expressions(self):
        title = Text("Gain Expressions in Terms of S-Parameters",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── G_T (factored: source mismatch · |S21|² · load mismatch) ────
        gt_head = MathTex(r"G_T \;=", font_size=30, color=RED)\
            .move_to([-5.6, 2.2, 0])
        gt_gs = MathTex(
            r"\underbrace{\frac{1-|\Gamma_S|^2}{|1-S_{11}\Gamma_S|^2}}_{G_S}",
            font_size=28, color=BLUE,
        ).move_to([-2.8, 2.2, 0])
        gt_s21 = MathTex(r"|S_{21}|^2", font_size=30, color=YELLOW)\
            .move_to([ 0.2, 2.2, 0])
        gt_gl = MathTex(
            r"\underbrace{\frac{1-|\Gamma_L|^2}{|1-S_{22}'\,\Gamma_L|^2}}_{G_L}",
            font_size=28, color=GREEN,
        ).move_to([ 3.5, 2.2, 0])

        s22p = MathTex(
            r"S_{22}' = S_{22} + \frac{S_{21}S_{12}\Gamma_S}{1-S_{11}\Gamma_S}",
            font_size=24, color=GRAY,
        ).move_to([0, 0.6, 0])

        self.play(Write(gt_head))
        self.play(Write(gt_gs))
        self.play(Write(gt_s21))
        self.play(Write(gt_gl))
        self.play(FadeIn(s22p))
        self.wait(1)

        # ── G_A ──────────────────────────────────────────────────────────
        ga_eq = MathTex(
            r"G_A \;=\; |S_{21}|^2\;\frac{1-|\Gamma_S|^2}"
            r"{|1-S_{11}\Gamma_S|^2\,\bigl(1-|\Gamma_{\rm out}|^2\bigr)}",
            font_size=28, color=GREEN,
        ).move_to([0, -0.8, 0])
        ga_note = MathTex(
            r"\Gamma_{\rm out} = S_{22}+\frac{S_{12}S_{21}\Gamma_S}{1-S_{11}\Gamma_S}",
            font_size=22, color=GRAY,
        ).next_to(ga_eq, DOWN, buff=0.2)

        self.play(FadeIn(ga_eq), FadeIn(ga_note))
        self.wait(0.8)

        # ── G_P ──────────────────────────────────────────────────────────
        gp_eq = MathTex(
            r"G_P \;=\; |S_{21}|^2\;\frac{1-|\Gamma_L|^2}"
            r"{\bigl(1-|\Gamma_{\rm in}|^2\bigr)\,|1-S_{22}\Gamma_L|^2}",
            font_size=28, color=BLUE,
        ).move_to([0, -2.5, 0])
        gp_note = MathTex(
            r"\Gamma_{\rm in} = S_{11}+\frac{S_{12}S_{21}\Gamma_L}{1-S_{22}\Gamma_L}",
            font_size=22, color=GRAY,
        ).next_to(gp_eq, DOWN, buff=0.2)

        self.play(FadeIn(gp_eq), FadeIn(gp_note))
        self.wait(1.5)

        self.play(FadeOut(VGroup(
            gt_head, gt_gs, gt_s21, gt_gl, s22p,
            ga_eq, ga_note, gp_eq, gp_note,
        )))

        # G_T vs |Γ_L| for a unilateral device, fixed Γ_S
        # G_T = |S21|² · (1-|Γ_S|²)/|1-S11 Γ_S|² · (1-|Γ_L|²)/|1-S22 Γ_L|²
        subtitle = Text("G_T vs Γ_L  (unilateral, S₁₂=0; sweep |Γ_L|, phase chosen for match)",
                        font_size=22, color=GRAY).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(subtitle))

        S11, S21, S22 = 0.5, 2.0, 0.4
        GS_mag = 0.5  # |Γ_S| fixed; phase conjugate-matched to S11 → Γ_S = S11*
        gs = complex(S11, 0)  # conjugate match: Γ_S = S11*
        gs_factor = (1 - abs(gs)**2) / abs(1 - S11 * gs)**2
        s21_sq = S21 ** 2

        # Express G_T(|Γ_L|) assuming phase(Γ_L) aligned with S22* (optimum phase)
        def gt_of(mag):
            gl = complex(S22, 0) * (mag / max(abs(complex(S22, 0)), 1e-9))
            return s21_sq * gs_factor * (1 - abs(gl)**2) / abs(1 - S22 * gl)**2

        # Precompute y_max
        mags = np.linspace(0, 0.99, 200)
        vals = [gt_of(m) for m in mags]
        y_max = max(vals) * 1.15

        ax = Axes(
            x_range=[0, 1.0, 0.25], y_range=[0, y_max, y_max / 4],
            x_length=8, y_length=3.4, tips=False,
            axis_config={"stroke_width": 2, "include_numbers": True,
                         "decimal_number_config": {"num_decimal_places": 2}},
        ).move_to([0, -1.1, 0])
        xlab = MathTex(r"|\Gamma_L|", font_size=24).next_to(ax, DOWN, buff=0.15)
        ylab = MathTex(r"G_T",       font_size=24).next_to(ax, LEFT, buff=0.15)

        curve = ax.plot(gt_of, x_range=[0, 0.99, 0.01], color=RED, stroke_width=3)

        # Optimum at Γ_L = S22* → |Γ_L| = S22 (real positive in this case)
        gl_t = ValueTracker(0.0)
        dot = always_redraw(lambda: Dot(
            ax.c2p(gl_t.get_value(), gt_of(gl_t.get_value())),
            color=YELLOW, radius=0.08,
        ))
        vline = always_redraw(lambda: DashedLine(
            ax.c2p(gl_t.get_value(), 0),
            ax.c2p(gl_t.get_value(), gt_of(gl_t.get_value())),
            color=GRAY, stroke_width=1.5,
        ))
        readout = always_redraw(lambda: MathTex(
            rf"|\Gamma_L|={gl_t.get_value():.2f},\ "
            rf"G_T={gt_of(gl_t.get_value()):.2f}",
            font_size=22, color=YELLOW
        ).to_edge(DOWN, buff=0.25))

        opt_line = DashedLine(
            ax.c2p(S22, 0), ax.c2p(S22, y_max * 0.95),
            color=GREEN, stroke_width=1.5,
        )
        opt_lbl = MathTex(r"|\Gamma_L|=|S_{22}^*|\ \text{(conj. match)}",
                          font_size=20, color=GREEN).next_to(
            ax.c2p(S22, y_max * 0.95), UP, buff=0.05)

        self.play(Create(ax), FadeIn(xlab), FadeIn(ylab))
        self.play(Create(curve))
        self.play(Create(opt_line), FadeIn(opt_lbl))
        self.add(dot, vline, readout)
        self.play(gl_t.animate.set_value(0.95), run_time=3, rate_func=linear)
        self.play(gl_t.animate.set_value(S22), run_time=1.5)
        self.wait(1)

        for m in (dot, vline, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, subtitle, ax, xlab, ylab, curve,
                                 opt_line, opt_lbl, dot, vline, readout)))
