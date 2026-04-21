from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Scene 4.5 — Bilinear Transformations
# Why reflection-coefficient algebra is Möbius geometry
# (Placed before stability & unilateral gain since those rely on circle maps)
# ---------------------------------------------------------------------------

class BilinearTransformations(Scene):

    def construct(self):
        self.act1_title()
        self.act2_definition()
        self.act3_circuit_connection()
        self.act4_properties()
        self.act5_circles_to_circles()
        self.act5b_point_mapping()
        self.act6_applications()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        tag = Text("Interlude 4.5", font_size=22, color=GRAY)
        title = Text("Bilinear Transformations", font_size=42)
        sub = Text("the geometry behind stability & gain circles",
                   font_size=26, color=GRAY)
        tag.next_to(title, UP, buff=0.2)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(tag), Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(tag, title, sub)))

    # ── Act 2: Definition ──────────────────────────────────────────────────
    def act2_definition(self):
        title = Text("Definition — the Möbius Transformation",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Core formula
        defn = MathTex(
            r"w(z) \;=\; \frac{a z + b}{c z + d},\qquad ad - bc \neq 0",
            font_size=44,
        ).move_to([0, 1.7, 0])
        defn_box = SurroundingRectangle(defn, color=YELLOW, buff=0.22, corner_radius=0.1)
        self.play(Write(defn))
        self.play(Create(defn_box))
        self.wait(0.8)

        # Parameter annotations
        data = [
            (r"a,\,b,\,c,\,d \in \mathbb{C}", "four complex coefficients",   WHITE),
            (r"ad - bc \neq 0",               "non-degenerate (invertible)", GREEN),
            (r"3\text{ complex DOF}",
             "divide by d — only the ratio a:b:c:d matters",               YELLOW),
        ]
        rows = VGroup()
        for tex, desc, col in data:
            eq  = MathTex(tex, font_size=28, color=col)
            txt = Text(f"— {desc}", font_size=20, color=GRAY)
            rows.add(VGroup(eq, txt).arrange(RIGHT, buff=0.4))
        rows.arrange(DOWN, buff=0.38, aligned_edge=LEFT).move_to([0.5, -0.3, 0])

        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.55)
            self.wait(0.3)

        # Three-point uniqueness
        three_pt = MathTex(
            r"\text{Three point-pairs }(z_k \mapsto w_k)\text{ determine }w(z)\text{ uniquely.}",
            font_size=24, color=GRAY,
        ).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(three_pt))
        self.wait(2)
        self.play(FadeOut(VGroup(title, defn, defn_box, rows, three_pt)))

    # ── Act 3: Circuit Connection ──────────────────────────────────────────
    def act3_circuit_connection(self):
        title = Text("Why S-Parameter Networks Are Bilinear",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Wave equations
        waves = MathTex(
            r"b_1 = S_{11}a_1 + S_{12}a_2,\qquad{}"
            r"b_2 = S_{21}a_1 + S_{22}a_2",
            font_size=27, color=GRAY,
        ).move_to([0, 2.3, 0])
        self.play(Write(waves))
        self.wait(0.5)

        # Load condition
        load = MathTex(
            r"a_2 = \Gamma_L\,b_2 \qquad \text{(matched-load termination)}",
            font_size=27, color=GRAY,
        ).move_to([0, 1.55, 0])
        self.play(FadeIn(load))
        self.wait(0.5)

        # Eliminate a2, b2
        elim = Text("Eliminate a₂ and b₂ :", font_size=22, color=YELLOW)\
            .move_to([-2.5, 0.75, 0])
        self.play(FadeIn(elim))

        gin_eq = MathTex(
            r"\Gamma_{\rm in} \;=\; \frac{b_1}{a_1} \;=\; "
            r"S_{11} + \frac{S_{12}S_{21}\,\Gamma_L}{1 - S_{22}\Gamma_L}",
            font_size=32,
        ).move_to([0, -0.05, 0])
        self.play(Write(gin_eq))
        self.wait(0.6)

        # Compact bilinear form
        rewrite = MathTex(
            r"\Gamma_{\rm in} \;=\;"
            r"\frac{S_{11} - \Delta\,\Gamma_L}{1 - S_{22}\,\Gamma_L},\qquad"
            r"\Delta \equiv S_{11}S_{22} - S_{12}S_{21}",
            font_size=30, color=BLUE,
        ).move_to([0, -1.05, 0])
        rbox = SurroundingRectangle(rewrite, color=BLUE, buff=0.18, corner_radius=0.1)
        self.play(Write(rewrite))
        self.play(Create(rbox))
        self.wait(0.8)

        # Map to standard form
        mapping = MathTex(
            r"w = \frac{az+b}{cz+d}:\quad{}"
            r"a=-\Delta,\; b=S_{11},\; c=-S_{22},\; d=1,\; z=\Gamma_L",
            font_size=22, color=GRAY,
        ).move_to([0, -2.1, 0])
        self.play(FadeIn(mapping))

        conclusion = Text(
            "Every two-port maps reflection coefficients via a bilinear transformation.",
            font_size=20, color=GREEN,
        ).to_edge(DOWN, buff=0.38)
        self.play(FadeIn(conclusion))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, waves, load, elim,
                                  gin_eq, rewrite, rbox, mapping, conclusion)))

    # ── Act 4: Properties ─────────────────────────────────────────────────
    def act4_properties(self):
        title = Text("Properties of Bilinear Transformations",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        props = [
            (r"\text{Circles} \to \text{circles}",
             "maps any circle or line to a circle or line",
             YELLOW),
            (r"\text{Conformal (angle-preserving)}",
             "intersection angles between curves are unchanged",
             GREEN),
            (r"\text{3 point-pairs} \Rightarrow \text{unique map}",
             "three constraints fully specify all parameters",
             BLUE),
            (r"T_2 \circ T_1 \;\text{is bilinear}",
             "closed under composition — forms a group",
             RED),
            (r"T^{-1}(w) = \frac{dw - b}{a - cw}",
             "inverse exists and is also bilinear",
             WHITE),
        ]

        rows = VGroup()
        for tex, desc, col in props:
            eq  = MathTex(tex, font_size=26, color=col)
            txt = Text(f"— {desc}", font_size=18, color=GRAY)
            rows.add(VGroup(eq, txt).arrange(RIGHT, buff=0.35))
        rows.arrange(DOWN, buff=0.43, aligned_edge=LEFT).move_to([0.7, -0.25, 0])

        for row in rows:
            self.play(FadeIn(row, shift=UP * 0.15), run_time=0.65)
            self.wait(0.6)

        self.wait(2)
        self.play(FadeOut(VGroup(title, rows)))

    # ── Act 5: Circles → Circles (animated) ──────────────────────────────
    def act5_circles_to_circles(self):
        title = Text("Circles Map to Circles", font_size=30).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── Axes ────────────────────────────────────────────────────────────
        lc = np.array([-3.2, -0.1, 0])
        rc = np.array([3.2, -0.1, 0])
        ax_cfg = dict(color=GRAY, stroke_width=1.2, include_tip=False)

        # Equal x/y scale so circles render as circles, not ellipses
        axes_l = Axes(x_range=[-2.0, 2.0, 1], y_range=[-2.0, 2.0, 1],
                      x_length=3.6, y_length=3.6,
                      axis_config=ax_cfg).move_to(lc)
        axes_r = Axes(x_range=[-2.0, 2.0, 1], y_range=[-2.0, 2.0, 1],
                      x_length=3.6, y_length=3.6,
                      axis_config=ax_cfg).move_to(rc)

        z_lbl = Text("z-plane", font_size=20, color=GRAY).move_to(lc + UP * 2.3)
        w_lbl = Text("w-plane", font_size=20, color=GRAY).move_to(rc + UP * 2.3)
        self.play(Create(axes_l), Create(axes_r), FadeIn(z_lbl), FadeIn(w_lbl))

        # ── Input: unit circle ───────────────────────────────────────────────
        in_circle = ParametricFunction(
            lambda t: axes_l.c2p(np.cos(t), np.sin(t)),
            t_range=[0, TAU], color=YELLOW, stroke_width=2.5,
        )
        in_lbl = MathTex(r"|z|=1", font_size=18, color=YELLOW)\
            .move_to(lc + RIGHT * 1.4 + UP * 0.45)
        self.play(Create(in_circle), FadeIn(in_lbl))
        self.wait(0.3)

        # ── Map: w = (z + 0.2) / (c·z + 1),  c controlled by tracker ────────
        b_v = 0.2

        def bil(z, c):
            return (z + b_v) / (c * z + 1.0)

        c_tracker = ValueTracker(0.3)
        theta     = ValueTracker(0.0)

        # Output circle — redrawn via become() each frame so it morphs with c
        def make_out_curve(c):
            return ParametricFunction(
                lambda t: axes_r.c2p(
                    bil(np.exp(1j * t), c).real,
                    bil(np.exp(1j * t), c).imag,
                ),
                t_range=[0, TAU], color=BLUE, stroke_width=2.5,
            )

        out_circle = make_out_curve(c_tracker.get_value())
        out_circle.add_updater(
            lambda m: m.become(make_out_curve(c_tracker.get_value()))
        )

        # Formula label (fixed) — between the two planes
        param_eq = MathTex(
            r"w = \frac{z + 0.2}{c\,z + 1}",
            font_size=28, color=GRAY,
        ).move_to([0, 0.6, 0])

        self.play(Create(out_circle), FadeIn(param_eq))
        self.wait(0.3)

        # ── Phase 1: dot traces z-circle → w-circle ─────────────────────────
        in_dot = always_redraw(lambda: Dot(
            axes_l.c2p(np.cos(theta.get_value()), np.sin(theta.get_value())),
            color=RED, radius=0.1,
        ))
        out_dot = always_redraw(lambda: Dot(
            axes_r.c2p(
                bil(np.exp(1j * theta.get_value()), c_tracker.get_value()).real,
                bil(np.exp(1j * theta.get_value()), c_tracker.get_value()).imag,
            ),
            color=RED, radius=0.1,
        ))
        trace_note = Text(
            "as z traces the circle,  w traces the image circle",
            font_size=18, color=GRAY,
        ).to_edge(DOWN, buff=0.38)

        self.add(in_dot, out_dot)
        self.play(FadeIn(trace_note))
        self.play(theta.animate.set_value(TAU), run_time=4, rate_func=linear)
        self.wait(0.4)

        # ── Phase 2: vary c — circle centre and radius shift ─────────────────
        in_dot.clear_updaters()
        out_dot.clear_updaters()
        self.remove(in_dot, out_dot)

        c_lbl = always_redraw(lambda: MathTex(
            r"c = " + f"{c_tracker.get_value():.2f}",
            font_size=26, color=GREEN,
        ).next_to(param_eq, UP, buff=0.22))

        vary_note = Text(
            "varying  c  shifts the centre and scales the radius",
            font_size=18, color=GREEN,
        ).to_edge(DOWN, buff=0.38)

        self.play(FadeOut(trace_note), FadeIn(vary_note), FadeIn(c_lbl))
        self.play(c_tracker.animate.set_value(0.65), run_time=3, rate_func=smooth)
        self.play(c_tracker.animate.set_value(0.05), run_time=3, rate_func=smooth)
        self.play(c_tracker.animate.set_value(0.3),  run_time=2, rate_func=smooth)
        self.wait(1)

        # Cleanup
        out_circle.clear_updaters()
        c_lbl.clear_updaters()
        self.play(FadeOut(VGroup(
            title, axes_l, axes_r, z_lbl, w_lbl,
            in_circle, in_lbl, out_circle, param_eq,
            vary_note, c_lbl,
        )))

    # ── Act 5b: Point-by-point z → w mapping ─────────────────────────────
    def act5b_point_mapping(self):
        title = Text("Point-by-Point Mapping  z  →  w  =  (z − i) / (z + i)",
                     font_size=26).to_edge(UP, buff=0.4)
        self.play(Write(title))

        lc = np.array([-3.4, -0.3, 0])
        rc = np.array([ 3.4, -0.3, 0])
        ax_cfg = dict(color=GRAY, stroke_width=1.2, include_tip=False)

        axes_l = Axes(x_range=[-2.2, 2.2, 1], y_range=[-2.0, 2.0, 1],
                      x_length=3.8, y_length=3.8, axis_config=ax_cfg).move_to(lc)
        axes_r = Axes(x_range=[-1.6, 1.6, 0.5], y_range=[-1.6, 1.6, 0.5],
                      x_length=3.8, y_length=3.8, axis_config=ax_cfg).move_to(rc)
        z_lbl = Text("z-plane", font_size=20, color=GRAY).move_to(lc + UP * 2.3)
        w_lbl = Text("w-plane", font_size=20, color=GRAY).move_to(rc + UP * 2.3)
        self.play(Create(axes_l), Create(axes_r), FadeIn(z_lbl), FadeIn(w_lbl))

        # Cayley transform: upper half-plane → unit disk
        def bil(z):
            return (z - 1j) / (z + 1j)

        # Unit circle on the w-plane (boundary) for reference
        unit_w = ParametricFunction(
            lambda t: axes_r.c2p(np.cos(t), np.sin(t)),
            t_range=[0, TAU], color=GRAY, stroke_width=1.5,
        )
        self.play(Create(unit_w))

        # ── Morphing grid: horizontal and vertical lines in z → curves in w ──
        grid_z = VGroup()
        grid_w = VGroup()
        h_vals = np.linspace(0.1, 1.8, 5)     # Im(z) = const > 0
        v_vals = np.linspace(-1.8, 1.8, 7)    # Re(z) = const

        for h in h_vals:
            col = interpolate_color(BLUE_E, BLUE, h / 1.8)
            lz = ParametricFunction(
                lambda x, h=h: axes_l.c2p(x, h),
                t_range=[-2.0, 2.0, 0.05], color=col, stroke_width=2,
            )
            lw = ParametricFunction(
                lambda x, h=h: axes_r.c2p(bil(x + 1j * h).real,
                                          bil(x + 1j * h).imag),
                t_range=[-2.0, 2.0, 0.03], color=col, stroke_width=2,
            )
            grid_z.add(lz)
            grid_w.add(lw)

        for v in v_vals:
            col = interpolate_color(RED_E, RED, (v + 1.8) / 3.6)
            lz = ParametricFunction(
                lambda y, v=v: axes_l.c2p(v, y),
                t_range=[0.0, 2.0, 0.05], color=col, stroke_width=2,
            )
            lw = ParametricFunction(
                lambda y, v=v: axes_r.c2p(bil(v + 1j * y).real,
                                          bil(v + 1j * y).imag),
                t_range=[0.01, 2.0, 0.03], color=col, stroke_width=2,
            )
            grid_z.add(lz)
            grid_w.add(lw)

        self.play(LaggedStart(*[Create(l) for l in grid_z], lag_ratio=0.05),
                  run_time=2)
        self.wait(0.4)
        # Morph z-grid into its w-image (shows the nonlinear warping)
        grid_copy = grid_z.copy()
        self.add(grid_copy)
        self.play(Transform(grid_copy, grid_w), run_time=3)
        self.wait(0.5)

        # ── Moving point z(t) with image w(t) ───────────────────────────────
        note = Text("drag the point in z  →  watch the image in w",
                    font_size=20, color=YELLOW).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note))

        t = ValueTracker(0.0)

        def z_of(s):
            # Figure-eight / Lissajous path that wanders the upper half-plane
            return complex(1.3 * np.sin(s), 0.9 + 0.7 * np.sin(2 * s))

        z_dot = always_redraw(lambda: Dot(
            axes_l.c2p(z_of(t.get_value()).real, z_of(t.get_value()).imag),
            color=YELLOW, radius=0.10,
        ))
        w_dot = always_redraw(lambda: Dot(
            axes_r.c2p(bil(z_of(t.get_value())).real,
                       bil(z_of(t.get_value())).imag),
            color=YELLOW, radius=0.10,
        ))

        # Persistent trails
        z_trail = TracedPath(z_dot.get_center, stroke_color=YELLOW,
                             stroke_width=2, dissipating_time=None)
        w_trail = TracedPath(w_dot.get_center, stroke_color=YELLOW,
                             stroke_width=2, dissipating_time=None)

        readout = always_redraw(lambda: MathTex(
            rf"z={z_of(t.get_value()).real:.2f}"
            rf"{'+' if z_of(t.get_value()).imag>=0 else '-'}"
            rf"j{abs(z_of(t.get_value()).imag):.2f},\ \ "
            rf"w={bil(z_of(t.get_value())).real:.2f}"
            rf"{'+' if bil(z_of(t.get_value())).imag>=0 else '-'}"
            rf"j{abs(bil(z_of(t.get_value())).imag):.2f}",
            font_size=22, color=YELLOW,
        ).next_to(note, UP, buff=0.15))

        self.add(z_trail, w_trail, z_dot, w_dot, readout)
        self.play(t.animate.set_value(2 * TAU), run_time=7, rate_func=linear)
        self.wait(0.5)

        # Highlight three canonical images: 0 → -1, ∞ → +1, i → 0
        canon = VGroup(
            MathTex(r"z=0\ \mapsto\ w=-1", font_size=20, color=RED),
            MathTex(r"z=i\ \mapsto\ w=0",  font_size=20, color=GREEN),
            MathTex(r"z\to\infty\ \mapsto\ w=+1", font_size=20, color=BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(title, DOWN, buff=0.2).to_edge(RIGHT, buff=0.4)
        self.play(FadeIn(canon))
        self.wait(2)

        for m in (z_dot, w_dot, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(
            title, axes_l, axes_r, z_lbl, w_lbl, unit_w,
            grid_z, grid_w, grid_copy,
            z_dot, w_dot, z_trail, w_trail,
            note, readout, canon,
        )))

    # ── Act 6: Applications in Microwave Circuits ─────────────────────────
    def act6_applications(self):
        title = Text("Bilinear Geometry in Microwave Circuits",
                     font_size=30).to_edge(UP, buff=0.4)
        self.play(Write(title))

        apps = [
            (
                "Smith Chart",
                r"z_n = \frac{1+\Gamma}{1-\Gamma}",
                r"|\Gamma|\leq 1 \;\longrightarrow\; \text{Re}\{z_n\}\geq 0",
                YELLOW,
            ),
            (
                "Stability Circles",
                r"|\Gamma_{\rm in}(\Gamma_L)| = 1",
                r"\text{unit circle in }\Gamma_{\rm in}"
                r"\;\to\;\text{circle in }\Gamma_L\text{-plane}",
                RED,
            ),
            (
                "Constant-Gain Circles",
                r"G_T(\Gamma_L) = G_0",
                r"\text{bilinear rational level sets are circles}",
                BLUE,
            ),
            (
                "Noise-Figure Circles",
                r"F(\Gamma_S) = F_0",
                r"\text{same structure — circles in }\Gamma_S\text{-plane}",
                GREEN,
            ),
        ]

        groups = VGroup()
        for name, eq_str, note_str, col in apps:
            hdr   = Text(name + ":", font_size=21, color=col)
            eq    = MathTex(eq_str,   font_size=25, color=col)
            note  = MathTex(note_str, font_size=19, color=GRAY)
            block = VGroup(hdr, eq, note).arrange(RIGHT, buff=0.38)
            groups.add(block)
        groups.arrange(DOWN, buff=0.52, aligned_edge=LEFT).move_to([0.5, -0.15, 0])

        for g in groups:
            self.play(FadeIn(g, shift=UP * 0.18), run_time=0.75)
            self.wait(0.8)

        self.wait(2)
        self.play(FadeOut(VGroup(title, groups)))
