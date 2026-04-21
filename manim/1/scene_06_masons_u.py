from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Scene 6 — Stability, Mason's U, and f_max
# Topics 16 (consolidated) + 19–20
# ---------------------------------------------------------------------------

class MasonsU(Scene):

    def construct(self):
        self.act1_title()
        self.act2_time_domain_stability()
        self.act3_smith_chart_stability()
        self.act4_masons_u()
        self.act5_fmax()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Stability, Mason's U, and f_max", font_size=40)
        sub = Text("from time-domain instability to the intrinsic device limit",
                   font_size=24, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2: Time-Domain Stability ──────────────────────────────────────
    def act2_time_domain_stability(self):
        title = Text("Stability in the Time Domain",
                     font_size=32).to_edge(UP, buff=0.32)
        self.play(Write(title))

        # ── Feedback-loop picture ────────────────────────────────────────
        gs_box = Rectangle(width=1.15, height=0.65, color=BLUE,
                           stroke_width=2).move_to([-4.5, 2.35, 0])
        gs_lbl = MathTex(r"\Gamma_S", color=BLUE, font_size=26).move_to(gs_box)
        dev_box = Rectangle(width=1.55, height=0.75, color=WHITE,
                            stroke_width=2).move_to([0, 2.35, 0])
        dev_lbl = MathTex(r"[S]", color=WHITE, font_size=24).move_to(dev_box)
        gl_box = Rectangle(width=1.15, height=0.65, color=RED,
                           stroke_width=2).move_to([4.5, 2.35, 0])
        gl_lbl = MathTex(r"\Gamma_L", color=RED, font_size=26).move_to(gl_box)

        a1 = Arrow([-3.9, 2.50, 0], [-0.85, 2.50, 0], buff=0,
                   color=YELLOW, stroke_width=2, tip_length=0.18)
        b1 = Arrow([-0.85, 2.20, 0], [-3.9, 2.20, 0], buff=0,
                   color=BLUE, stroke_width=2, tip_length=0.18)
        a2 = Arrow([0.85, 2.20, 0], [3.9, 2.20, 0], buff=0,
                   color=RED, stroke_width=2, tip_length=0.18)
        b2 = Arrow([3.9, 2.50, 0], [0.85, 2.50, 0], buff=0,
                   color=YELLOW, stroke_width=2, tip_length=0.18)

        self.play(Create(VGroup(gs_box, gs_lbl, dev_box, dev_lbl, gl_box, gl_lbl)),
                  run_time=0.7)
        self.play(Create(VGroup(a1, b1, a2, b2)), run_time=0.8)

        loop_eq = MathTex(
            r"\text{loop gain }L \;=\; \Gamma_S\,\Gamma_{\rm in}",
            font_size=26, color=YELLOW,
        ).move_to([0, 1.25, 0])
        self.play(FadeIn(loop_eq))
        self.wait(0.3)

        # ── Time-domain axes ──────────────────────────────────────────────
        ax = Axes(
            x_range=[0, 6, 1], y_range=[-2.6, 2.6, 1],
            x_length=9.0, y_length=2.2,
            axis_config=dict(color=GRAY, stroke_width=1.3, include_tip=True,
                             include_numbers=False),
        ).move_to([0, -1.0, 0])
        t_lbl = MathTex(r"t", color=GRAY, font_size=22)\
            .next_to(ax.x_axis.get_right(), RIGHT, buff=0.12)
        v_lbl = MathTex(r"V(t)", color=GRAY, font_size=22)\
            .next_to(ax.y_axis.get_top(), UP, buff=0.08)
        self.play(Create(ax), FadeIn(t_lbl), FadeIn(v_lbl))

        # ── Wave with σ tracker ───────────────────────────────────────────
        sigma = ValueTracker(-0.55)
        omega = 4.5

        def wave_color(s):
            if s > 0.04: return RED
            if s < -0.04: return GREEN
            return YELLOW

        def make_wave(s):
            col = wave_color(s)
            return ax.plot(
                lambda t: float(np.clip(np.exp(s * t) * np.cos(omega * t),
                                        -2.5, 2.5)),
                x_range=[0, 6, 0.01],
                color=col, stroke_width=2.5,
            )

        wave = make_wave(sigma.get_value())
        wave.add_updater(lambda m: m.become(make_wave(sigma.get_value())))

        def regime_tex(s):
            if s > 0.04:
                return (r"|L|>1 \;\Rightarrow\; \sigma > 0 "
                        r"\;\Rightarrow\; \text{UNSTABLE}")
            if s < -0.04:
                return (r"|L|<1 \;\Rightarrow\; \sigma < 0 "
                        r"\;\Rightarrow\; \text{STABLE}")
            return (r"|L|=1 \;\Rightarrow\; \sigma = 0 "
                    r"\;\Rightarrow\; \text{MARGINAL}")

        regime_lbl = always_redraw(lambda: MathTex(
            regime_tex(sigma.get_value()),
            font_size=24,
            color=wave_color(sigma.get_value()),
        ).to_edge(DOWN, buff=0.3))

        self.add(wave)
        self.play(FadeIn(regime_lbl), run_time=0.5)
        self.wait(1.2)
        self.play(sigma.animate.set_value(0.0), run_time=2.5, rate_func=smooth)
        self.wait(0.7)
        self.play(sigma.animate.set_value(0.3), run_time=2.5, rate_func=smooth)
        self.wait(1.5)

        wave.clear_updaters()
        regime_lbl.clear_updaters()
        self.play(FadeOut(VGroup(
            title, gs_box, gs_lbl, dev_box, dev_lbl, gl_box, gl_lbl,
            a1, b1, a2, b2, loop_eq,
            ax, t_lbl, v_lbl, wave, regime_lbl,
        )))

    # ── Act 3: Stability Circles on Smith Chart ──────────────────────────
    def act3_smith_chart_stability(self):
        title = Text("Stability Circles on the Smith Chart",
                     font_size=30).to_edge(UP, buff=0.32)
        self.play(Write(title))

        cond = MathTex(
            r"\text{Unconditional stability: }K>1\;\text{ and }\;|\Delta|<1",
            font_size=22, color=GREEN,
        ).next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(cond))
        self.wait(0.3)

        # ── Smith chart (left side) ───────────────────────────────────────
        R = 1.9
        center = np.array([-2.3, -0.85, 0])

        uc = Circle(radius=R, color=WHITE, stroke_width=2.2).move_to(center)
        ax_h = Line(center + LEFT*R, center + RIGHT*R,
                    color=GRAY, stroke_width=1.1)
        ax_v = Line(center + DOWN*R, center + UP*R,
                    color=GRAY, stroke_width=1.1)
        origin_dot = Dot(center, color=GRAY, radius=0.035)
        u_lbl = MathTex(r"|\Gamma|=1", color=GRAY, font_size=17)\
            .move_to(center + DOWN*(R + 0.22) + RIGHT*0.7)
        plane_lbl = MathTex(r"\Gamma\text{-plane}", color=GRAY, font_size=20)\
            .move_to(center + UP*(R + 0.25))

        self.play(Create(uc), Create(VGroup(ax_h, ax_v)),
                  FadeIn(VGroup(origin_dot, u_lbl, plane_lbl)))

        # ── Load stability circle ────────────────────────────────────────
        cL_rel = np.array([1.05, 0.45, 0])
        rL_rel = 0.80
        cL_pos = center + cL_rel * R
        rL_vis = rL_rel * R

        load_SC = Circle(radius=rL_vis, color=BLUE, stroke_width=2.4)\
            .move_to(cL_pos)
        load_shade = Circle(radius=rL_vis, color=BLUE, fill_opacity=0.22,
                            stroke_opacity=0).move_to(cL_pos)
        load_lbl = MathTex(r"|\Gamma_{\rm in}(\Gamma_L)|=1",
                           color=BLUE, font_size=17)\
            .move_to(center + np.array([1.25, 2.05, 0]))

        self.play(Create(load_SC), run_time=0.9)
        self.play(FadeIn(load_shade), FadeIn(load_lbl), run_time=0.5)

        # ── Source stability circle ──────────────────────────────────────
        cS_rel = np.array([-0.55, -1.05, 0])
        rS_rel = 0.85
        cS_pos = center + cS_rel * R
        rS_vis = rS_rel * R

        src_SC = Circle(radius=rS_vis, color=RED, stroke_width=2.4)\
            .move_to(cS_pos)
        src_shade = Circle(radius=rS_vis, color=RED, fill_opacity=0.22,
                           stroke_opacity=0).move_to(cS_pos)
        src_lbl = MathTex(r"|\Gamma_{\rm out}(\Gamma_S)|=1",
                          color=RED, font_size=17)\
            .move_to(center + np.array([-1.75, -2.3, 0]))

        self.play(Create(src_SC), run_time=0.9)
        self.play(FadeIn(src_shade), FadeIn(src_lbl), run_time=0.5)

        # Keep outline on top of the shades
        self.bring_to_front(uc, ax_h, ax_v, origin_dot)

        # ── Formulas (right side) ────────────────────────────────────────
        formulas = VGroup(
            Text("Load stability circle:", font_size=18, color=BLUE),
            MathTex(r"C_L = \frac{(S_{22}-\Delta S_{11}^*)^*}{|S_{22}|^2-|\Delta|^2}",
                    font_size=22, color=BLUE),
            MathTex(r"r_L = \left|\frac{S_{12}S_{21}}{|S_{22}|^2-|\Delta|^2}\right|",
                    font_size=22, color=BLUE),
            Text("Source stability circle:", font_size=18, color=RED),
            MathTex(r"C_S = \frac{(S_{11}-\Delta S_{22}^*)^*}{|S_{11}|^2-|\Delta|^2}",
                    font_size=22, color=RED),
            MathTex(r"r_S = \left|\frac{S_{12}S_{21}}{|S_{11}|^2-|\Delta|^2}\right|",
                    font_size=22, color=RED),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([3.8, -0.45, 0])

        self.play(FadeIn(formulas), run_time=0.7)
        self.wait(0.4)

        # ── Safe matching region callout ────────────────────────────────
        safe_pt = center + np.array([-0.45, 0.55, 0])
        safe_text = Text("stable matching region\n(both constraints satisfied)",
                         font_size=14, color=GREEN, line_spacing=0.8)\
            .move_to(center + np.array([-1.9, 1.55, 0]))
        safe_arrow = Arrow(safe_text.get_bottom() + RIGHT*0.15, safe_pt,
                           color=GREEN, stroke_width=2, tip_length=0.13, buff=0.12)
        safe_dot = Dot(safe_pt, color=GREEN, radius=0.07)

        self.play(FadeIn(safe_text), Create(safe_arrow), FadeIn(safe_dot))
        self.wait(1.2)

        # ── Animate: circles move as S-params vary with frequency ────────
        theta_tr = ValueTracker(0.0)

        def rot_center(rel, theta, sign=1.0):
            t = sign * theta
            c, s = np.cos(t), np.sin(t)
            return center + R * np.array([
                rel[0]*c - rel[1]*s,
                rel[0]*s + rel[1]*c,
                0,
            ])

        load_SC.add_updater(
            lambda m: m.move_to(rot_center(cL_rel, theta_tr.get_value(), +1)))
        load_shade.add_updater(
            lambda m: m.move_to(rot_center(cL_rel, theta_tr.get_value(), +1)))
        src_SC.add_updater(
            lambda m: m.move_to(rot_center(cS_rel, theta_tr.get_value(), -1)))
        src_shade.add_updater(
            lambda m: m.move_to(rot_center(cS_rel, theta_tr.get_value(), -1)))

        vary_note = Text("S-parameters vary with frequency  →  circles move",
                         font_size=18, color=YELLOW)\
            .to_edge(DOWN, buff=0.3)

        self.play(
            FadeOut(VGroup(load_lbl, src_lbl, safe_text, safe_arrow, safe_dot)),
            FadeIn(vary_note),
        )

        self.play(theta_tr.animate.set_value(0.75), run_time=2.2, rate_func=smooth)
        self.play(theta_tr.animate.set_value(-0.40), run_time=2.2, rate_func=smooth)
        self.play(theta_tr.animate.set_value(0.0), run_time=1.4, rate_func=smooth)

        for m in (load_SC, load_shade, src_SC, src_shade):
            m.clear_updaters()

        self.wait(0.5)
        self.play(FadeOut(VGroup(
            title, cond, uc, ax_h, ax_v, origin_dot, u_lbl, plane_lbl,
            load_SC, load_shade, src_SC, src_shade,
            formulas, vary_note,
        )))

    # ── Act 4: Mason's U ──────────────────────────────────────────────────
    def act4_masons_u(self):
        title = Text("Mason's Unilateral Power Gain  U",
                     font_size=32).to_edge(UP, buff=0.32)
        self.play(Write(title))

        # ── S-parameter form ─────────────────────────────────────────────
        u_eq = MathTex(
            r"U \;=\; \frac{\bigl|S_{21}/S_{12} - 1\bigr|^2}"
            r"{2K\,\bigl|S_{21}/S_{12}\bigr| \;-\; 2\,\mathrm{Re}\{S_{21}/S_{12}\}}",
            font_size=32,
        ).move_to([0, 1.95, 0])
        u_box = SurroundingRectangle(u_eq, color=YELLOW, buff=0.18,
                                     corner_radius=0.1)
        self.play(Write(u_eq))
        self.play(Create(u_box))
        self.wait(0.5)

        # ── Y-parameter equivalent ───────────────────────────────────────
        y_hint = Text("equivalently in Y-parameters:",
                      font_size=19, color=GRAY).move_to([0, 0.8, 0])
        u_y = MathTex(
            r"U \;=\; \frac{|Y_{21}-Y_{12}|^2}"
            r"{4\bigl(\mathrm{Re}\{Y_{11}\}\,\mathrm{Re}\{Y_{22}\}"
            r"-\,\mathrm{Re}\{Y_{12}\}\,\mathrm{Re}\{Y_{21}\}\bigr)}",
            font_size=28,
        ).move_to([0, -0.05, 0])
        self.play(FadeIn(y_hint), Write(u_y))
        self.wait(0.6)

        # ── Invariance statement ─────────────────────────────────────────
        inv_header = Text("Key property — lossless-embedding invariance:",
                          font_size=22, color=GREEN).move_to([0, -1.25, 0])

        dev = Rectangle(width=1.2, height=0.6, color=WHITE, stroke_width=2)\
            .move_to([-3.2, -2.25, 0])
        dev_t = MathTex(r"\text{device}", font_size=18).move_to(dev)
        emb = Rectangle(width=1.9, height=0.7, color=YELLOW, stroke_width=2)\
            .move_to([-0.4, -2.25, 0])
        emb_t = MathTex(r"\text{any lossless }[S']", font_size=16).move_to(emb)
        arr_embed = Arrow([-2.55, -2.25, 0], [-1.38, -2.25, 0],
                          buff=0, color=WHITE, stroke_width=2, tip_length=0.14)
        eq_sign = MathTex(r"\Rightarrow\;U\text{ unchanged}",
                          font_size=22, color=GREEN).move_to([2.9, -2.25, 0])

        self.play(FadeIn(inv_header))
        self.play(Create(dev), FadeIn(dev_t),
                  Create(arr_embed),
                  Create(emb), FadeIn(emb_t),
                  FadeIn(eq_sign))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, u_eq, u_box, y_hint, u_y,
                                  inv_header, dev, dev_t, arr_embed,
                                  emb, emb_t, eq_sign)))

    # ── Act 5: Frequency Rolloff & f_max ─────────────────────────────────
    def act5_fmax(self):
        title = Text("Frequency Rolloff and  f_max",
                     font_size=32).to_edge(UP, buff=0.32)
        self.play(Write(title))

        extrap = MathTex(
            r"U(f)\;\approx\;\left(\frac{f_{\max}}{f}\right)^{\!2}"
            r"\;\Longrightarrow\;"
            r"f_{\max} = f\,\sqrt{U(f)}",
            font_size=27, color=GREEN,
        ).move_to([0, 2.35, 0])
        self.play(FadeIn(extrap))
        self.wait(0.2)

        ax = Axes(
            x_range=[0, 5.5, 1], y_range=[0, 5.0, 1],
            x_length=7.5, y_length=3.5,
            axis_config=dict(color=GRAY, stroke_width=1.3, include_tip=True,
                             include_numbers=False),
        ).move_to([-0.5, -0.45, 0])
        x_lbl = MathTex(r"\log_{10} f", color=GRAY, font_size=22)\
            .next_to(ax.x_axis.get_right(), RIGHT, buff=0.08)
        y_lbl = MathTex(r"\log_{10} U", color=GRAY, font_size=22)\
            .next_to(ax.y_axis.get_top(), UP, buff=0.08)
        self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl))

        fmax_x = 4.6

        def u_log(x):
            return max(0.0, 2.0 * (fmax_x - x))

        rolloff = ax.plot(u_log, x_range=[2.2, fmax_x],
                          color=YELLOW, stroke_width=3)
        self.play(Create(rolloff), run_time=1.8)

        slope_lbl = MathTex(r"-20\,\mathrm{dB/dec}", font_size=20, color=YELLOW)\
            .move_to(ax.c2p(3.1, u_log(3.1) + 0.6))
        self.play(FadeIn(slope_lbl))

        u1_line = DashedLine(
            ax.c2p(0, 0), ax.c2p(fmax_x + 0.2, 0),
            color=RED, stroke_width=1.4, dash_length=0.12,
        )
        u1_lbl = MathTex(r"U=1", font_size=20, color=RED)\
            .next_to(ax.c2p(0, 0), LEFT, buff=0.1)
        fmax_dot = Dot(ax.c2p(fmax_x, 0), color=RED, radius=0.1)
        fmax_lbl = MathTex(r"f_{\max}", font_size=22, color=RED)\
            .next_to(fmax_dot, DOWN, buff=0.15)
        self.play(Create(u1_line), FadeIn(u1_lbl))
        self.play(Create(fmax_dot), FadeIn(fmax_lbl))

        meas_x = 3.2
        meas_dot = Dot(ax.c2p(meas_x, u_log(meas_x)), color=WHITE, radius=0.08)
        meas_lbl = MathTex(r"(f_0,\,U_0)", font_size=18, color=WHITE)\
            .next_to(meas_dot, UR, buff=0.06)
        hash_h = DashedLine(ax.c2p(0, u_log(meas_x)), ax.c2p(meas_x, u_log(meas_x)),
                             color=GRAY, stroke_width=1.1, dash_length=0.1)
        hash_v = DashedLine(ax.c2p(meas_x, 0), ax.c2p(meas_x, u_log(meas_x)),
                             color=GRAY, stroke_width=1.1, dash_length=0.1)
        self.play(Create(meas_dot), FadeIn(meas_lbl),
                  Create(hash_h), Create(hash_v))
        self.wait(0.6)

        # Physical interpretation
        interp_top = VGroup(
            MathTex(r"f > f_{\max}", font_size=22, color=RED),
            Text("no embedding", font_size=13, color=GRAY),
            Text("yields gain", font_size=13, color=GRAY),
        ).arrange(DOWN, buff=0.1)
        interp_bot = VGroup(
            MathTex(r"f < f_{\max}", font_size=22, color=GREEN),
            Text("amplification", font_size=13, color=GRAY),
            Text("is possible", font_size=13, color=GRAY),
        ).arrange(DOWN, buff=0.1)
        interp = VGroup(interp_top, interp_bot)\
            .arrange(DOWN, buff=0.32)\
            .move_to([5.3, -0.55, 0])
        self.play(FadeIn(interp))
        self.wait(0.5)

        bjt = MathTex(
            r"\text{BJT estimate: }",
            r"f_{\max}\;\approx\;\sqrt{\dfrac{f_T}{8\pi\, r_b\, C_{cb}}}",
            font_size=24, color=YELLOW,
        ).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(bjt))
        self.wait(2.5)

        self.play(FadeOut(VGroup(
            title, extrap, ax, x_lbl, y_lbl, rolloff, slope_lbl,
            u1_line, u1_lbl, fmax_dot, fmax_lbl,
            meas_dot, meas_lbl, hash_h, hash_v,
            interp, bjt,
        )))
