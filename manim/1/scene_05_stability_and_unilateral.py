from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Scene 5 — Stability & Unilateral Gain
# Topics 16–18
# ---------------------------------------------------------------------------

class StabilityAndUnilateral(Scene):

    def construct(self):
        self.act1_title()
        self.act2_stability_conditions()
        self.act3_stability_circles()
        self.act4_unilateral_approx()
        self.act5_max_unilateral_gain()
        self.act6_quadratic_form_derivation()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Stability & Unilateral Gain", font_size=40)
        sub = Text("K-factor, stability circles, and maximum gain",
                   font_size=26, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2: Stability Conditions ───────────────────────────────────────
    def act2_stability_conditions(self):
        title = Text("Stability Conditions", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Δ definition
        delta_lbl = Text("Determinant of the S-matrix:", font_size=24, color=GRAY)\
            .move_to([-1.3, 2.2, 0])
        delta_eq = MathTex(
            r"\Delta = S_{11}S_{22} - S_{12}S_{21}",
            font_size=36,
        ).move_to([0, 1.4, 0])
        self.play(Write(delta_lbl))
        self.play(Write(delta_eq))
        self.wait(0.8)

        # K-factor
        k_lbl = Text("Rollett stability factor:", font_size=24, color=GRAY)\
            .move_to([-2.0, 0.45, 0])
        k_eq = MathTex(
            r"K = \frac{1 - |S_{11}|^2 - |S_{22}|^2 + |\Delta|^2}"
            r"{2\,|S_{12}S_{21}|}",
            font_size=36,
        ).move_to([0, -0.5, 0])
        k_box = SurroundingRectangle(k_eq, color=YELLOW, buff=0.18, corner_radius=0.1)
        self.play(Write(k_lbl))
        self.play(Write(k_eq))
        self.play(Create(k_box))
        self.wait(1)

        # Unconditional stability
        cond_lbl = Text("Unconditional stability requires both:",
                        font_size=24, color=GREEN).move_to([0, -1.8, 0])
        cond = MathTex(
            r"K > 1 \qquad \text{and} \qquad |\Delta| < 1",
            font_size=36, color=GREEN,
        ).move_to([0, -2.6, 0])
        cond_box = SurroundingRectangle(cond, color=GREEN, buff=0.18, corner_radius=0.1)
        self.play(FadeIn(cond_lbl))
        self.play(Write(cond))
        self.play(Create(cond_box))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, delta_lbl, delta_eq,
                                  k_lbl, k_eq, k_box,
                                  cond_lbl, cond, cond_box)))

    # ── Act 3: Stability Circles ──────────────────────────────────────────
    def act3_stability_circles(self):
        title = Text("Stability Circles", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Callback to the bilinear-transformation interlude
        recall = Text(
            "Recall (§4.5): a bilinear map sends the |Γ_in|=1 circle to another circle "
            "in the Γ_L-plane — that image is the stability boundary.",
            font_size=18, color=GRAY,
        ).to_edge(UP, buff=1.05)
        self.play(FadeIn(recall))
        self.wait(0.8)
        self.play(FadeOut(recall))

        # ── Formulas on the right ──────────────────────────────────────────
        load_hdr = Text("Load stability circle:", font_size=22, color=BLUE)\
            .move_to([3.2, 2.3, 0])
        cl_eq = MathTex(
            r"C_L = \frac{(S_{22} - \Delta S_{11}^*)^*}{|S_{22}|^2 - |\Delta|^2}",
            font_size=24, color=BLUE,
        ).move_to([3.4, 1.4, 0])
        rl_eq = MathTex(
            r"r_L = \left|\frac{S_{12}S_{21}}{|S_{22}|^2 - |\Delta|^2}\right|",
            font_size=24, color=BLUE,
        ).move_to([3.4, 0.4, 0])

        src_hdr = Text("Source stability circle:", font_size=22, color=RED)\
            .move_to([3.2, -0.75, 0])
        cs_eq = MathTex(
            r"C_S = \frac{(S_{11} - \Delta S_{22}^*)^*}{|S_{11}|^2 - |\Delta|^2}",
            font_size=24, color=RED,
        ).move_to([3.4, -1.65, 0])
        rs_eq = MathTex(
            r"r_S = \left|\frac{S_{12}S_{21}}{|S_{11}|^2 - |\Delta|^2}\right|",
            font_size=24, color=RED,
        ).move_to([3.4, -2.6, 0])

        self.play(FadeIn(VGroup(load_hdr, cl_eq, rl_eq)))
        self.play(FadeIn(VGroup(src_hdr, cs_eq, rs_eq)))
        self.wait(0.8)

        # ── Γ-plane diagram on the left — tracker-driven ────────────────────
        center_l = np.array([-2.4, 0.2, 0])
        R_UNIT = 1.7  # pixel radius for |Γ|=1

        plane_lbl = Text("Γ_L-plane — sweep |S₁₂|", font_size=18, color=GRAY)\
            .move_to([-2.4, 2.65, 0])
        unit_circle = Circle(radius=R_UNIT, color=WHITE, stroke_width=2)\
            .move_to(center_l)
        unit_lbl = MathTex(r"|\Gamma|=1", font_size=18, color=GRAY)\
            .move_to(center_l + DOWN * 1.95)

        # Fixed S-params; sweep |S12|
        S11, S22, S21 = 0.6 + 0j, 0.5 + 0j, 2.0 + 0j
        s12_t = ValueTracker(0.05)

        def stab_load(s12_mag):
            s12 = s12_mag + 0j
            D = S11 * S22 - s12 * S21
            denom = abs(S22) ** 2 - abs(D) ** 2
            C = np.conj(S22 - D * np.conj(S11)) / denom
            r = abs(s12 * S21 / denom)
            K = (1 - abs(S11) ** 2 - abs(S22) ** 2 + abs(D) ** 2) / (2 * abs(s12 * S21))
            return C, r, K, D

        def sc_circle():
            C, r, _, _ = stab_load(s12_t.get_value())
            px = center_l + np.array([C.real * R_UNIT, C.imag * R_UNIT, 0])
            c = Circle(radius=r * R_UNIT, color=BLUE, stroke_width=2.5).move_to(px)
            return c

        def sc_shade():
            C, r, _, _ = stab_load(s12_t.get_value())
            px = center_l + np.array([C.real * R_UNIT, C.imag * R_UNIT, 0])
            return Circle(radius=r * R_UNIT, color=RED,
                          fill_opacity=0.2, stroke_opacity=0).move_to(px)

        stab_circle = always_redraw(sc_circle)
        shading     = always_redraw(sc_shade)

        readout = always_redraw(lambda: MathTex(
            rf"|S_{{12}}|={s12_t.get_value():.2f},\ "
            rf"K={stab_load(s12_t.get_value())[2]:.2f},\ "
            rf"|\Delta|={abs(stab_load(s12_t.get_value())[3]):.2f}",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.25))
        regime = always_redraw(lambda: Text(
            "unconditionally stable"
            if (stab_load(s12_t.get_value())[2] > 1
                and abs(stab_load(s12_t.get_value())[3]) < 1)
            else "conditionally stable",
            font_size=20,
            color=GREEN if (stab_load(s12_t.get_value())[2] > 1
                            and abs(stab_load(s12_t.get_value())[3]) < 1)
                  else ORANGE,
        ).next_to(readout, UP, buff=0.15))

        self.play(Create(unit_circle), FadeIn(unit_lbl), FadeIn(plane_lbl))
        self.add(shading, stab_circle, readout, regime)
        self.wait(0.5)

        # Sweep |S12|: low (unilateral, unconditional) → high (circle intrudes)
        self.play(s12_t.animate.set_value(0.45), run_time=3.5, rate_func=linear)
        self.wait(0.5)
        self.play(s12_t.animate.set_value(0.05), run_time=2.5, rate_func=linear)
        self.wait(0.5)

        for m in (stab_circle, shading, readout, regime):
            m.clear_updaters()
        self.play(FadeOut(VGroup(
            title, load_hdr, cl_eq, rl_eq, src_hdr, cs_eq, rs_eq,
            plane_lbl, unit_circle, unit_lbl,
            stab_circle, shading, readout, regime,
        )))

    # ── Act 4: Unilateral Approximation ──────────────────────────────────
    def act4_unilateral_approx(self):
        title = Text("Unilateral Approximation", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # S12 ≈ 0
        approx = MathTex(r"S_{12} \approx 0", font_size=52, color=YELLOW)\
            .move_to([0, 2.0, 0])
        approx_box = SurroundingRectangle(approx, color=YELLOW,
                                          buff=0.22, corner_radius=0.1)
        desc = Text("reverse transmission is negligible",
                    font_size=24, color=GRAY).move_to([0, 1.05, 0])
        self.play(Write(approx))
        self.play(Create(approx_box))
        self.play(FadeIn(desc))
        self.wait(0.8)

        # Input/output decouple
        decouple = MathTex(
            r"\Gamma_{\rm in} = S_{11},\qquad \Gamma_{\rm out} = S_{22}",
            font_size=32,
        ).move_to([0, 0.05, 0])
        self.play(Write(decouple))
        self.wait(0.6)

        # Unilateral figure of merit
        uf_lbl = Text("Unilateral figure of merit:", font_size=24, color=GRAY)\
            .move_to([-1.5, -0.9, 0])
        uf_eq = MathTex(
            r"U_f = \frac{|S_{12}|\,|S_{21}|\,|S_{11}|\,|S_{22}|}"
            r"{(1-|S_{11}|^2)(1-|S_{22}|^2)}",
            font_size=32,
        ).move_to([0, -1.85, 0])
        self.play(Write(uf_lbl))
        self.play(Write(uf_eq))
        self.wait(0.8)

        # Error bound
        err = MathTex(
            r"\frac{1}{(1+U_f)^2} \;<\; \frac{G_T}{G_{TU}} \;<\; \frac{1}{(1-U_f)^2}",
            font_size=28, color=GRAY,
        ).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(err))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, approx, approx_box, desc,
                                  decouple, uf_lbl, uf_eq, err)))

    # ── Act 5: Maximum Unilateral Transducer Gain ─────────────────────────
    def act5_max_unilateral_gain(self):
        title = Text("Maximum Unilateral Transducer Gain",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # Factored form
        factor_lbl = Text("Unilateral gain in factored form:",
                          font_size=22, color=GRAY).move_to([0, 2.35, 0])
        g_tu = MathTex(
            r"G_{TU} = "
            r"\underbrace{\frac{1-|\Gamma_S|^2}{|1-S_{11}\Gamma_S|^2}}_{G_S}"
            r"\;|S_{21}|^2\;"
            r"\underbrace{\frac{1-|\Gamma_L|^2}{|1-S_{22}\Gamma_L|^2}}_{G_L}",
            font_size=30,
        ).move_to([0, 1.3, 0])
        self.play(Write(factor_lbl))
        self.play(Write(g_tu))
        self.wait(1)

        # Conjugate matching
        match_lbl = Text("Maximised by conjugate matching:",
                         font_size=22, color=GREEN).move_to([-0.8, 0.15, 0])
        match_eq = MathTex(
            r"\Gamma_S = S_{11}^*,\qquad \Gamma_L = S_{22}^*",
            font_size=32, color=GREEN,
        ).move_to([0, -0.65, 0])
        self.play(FadeIn(match_lbl))
        self.play(Write(match_eq))
        self.wait(0.8)

        # G_TU,max
        gtu_max = MathTex(
            r"G_{TU,\max} = \frac{|S_{21}|^2}{(1-|S_{11}|^2)(1-|S_{22}|^2)}",
            font_size=38,
        ).move_to([0, -1.85, 0])
        box_max = SurroundingRectangle(gtu_max, color=BLUE, buff=0.22, corner_radius=0.1)
        self.play(Write(gtu_max))
        self.play(Create(box_max))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, factor_lbl, g_tu,
                                  match_lbl, match_eq,
                                  gtu_max, box_max)))

    # ── Act 6: Quadratic-Form Derivation of Unilateral Gain ───────────────
    # Port power P = ½ V† Y V; real part keeps only Y_h = ½(Y + Y†);
    # passivity ⟺ Y_h ≽ 0, and Mason's U is built on det(Y_h).
    def act6_quadratic_form_derivation(self):
        title = Text("Where Does Unilateral Gain Come From?",
                     font_size=32).to_edge(UP, buff=0.32)
        sub = Text("Port power as a Hermitian quadratic form in V",
                   font_size=22, color=GRAY).next_to(title, DOWN, buff=0.15)
        self.play(Write(title))
        self.play(FadeIn(sub))
        self.wait(0.5)

        # ── Step 1: I = Y V ─────────────────────────────────────────────
        step1 = Text("Port currents from the Y-matrix:",
                     font_size=20, color=GRAY).move_to([-3.7, 1.55, 0])
        iv_eq = MathTex(r"\mathbf{I} \;=\; Y\,\mathbf{V}",
                        font_size=34).move_to([-3.7, 0.85, 0])
        self.play(FadeIn(step1), Write(iv_eq))
        self.wait(0.4)

        # ── Step 2: complex power ──────────────────────────────────────
        step2 = Text("Complex power delivered to the network:",
                     font_size=20, color=GRAY).move_to([2.7, 1.55, 0])
        p_eq = MathTex(
            r"P \;=\; \tfrac{1}{2}\,\mathbf{V}^{\dagger}\mathbf{I}"
            r"\;=\; \tfrac{1}{2}\,\mathbf{V}^{\dagger} Y\, \mathbf{V}",
            font_size=30,
        ).move_to([2.7, 0.85, 0])
        self.play(FadeIn(step2), Write(p_eq))
        self.wait(0.6)

        # ── Step 3: Hermitian / anti-Hermitian split ───────────────────
        decomp_lbl = Text(
            "Split Y into a Hermitian (dissipative) + anti-Hermitian (reactive) part:",
            font_size=20, color=GRAY,
        ).move_to([0, 0.0, 0])
        decomp_eq = MathTex(
            r"Y \;=\; \underbrace{\tfrac{1}{2}(Y+Y^{\dagger})}_{Y_h\;=\;Y_h^{\dagger}}"
            r"\;+\;\underbrace{\tfrac{1}{2}(Y-Y^{\dagger})}_{Y_a\;=\;-Y_a^{\dagger}}",
            font_size=30,
        ).move_to([0, -0.85, 0])
        self.play(FadeIn(decomp_lbl))
        self.play(Write(decomp_eq))
        self.wait(0.8)

        # ── Step 4: real part of P keeps only Y_h ─────────────────────
        re_lbl = Text(
            "Since (V† Y V)* = V† Y† V, the anti-Hermitian part is pure imaginary —",
            font_size=19, color=GRAY,
        ).move_to([0, -1.75, 0])
        re_eq = MathTex(
            r"\mathrm{Re}\{P\} \;=\; \tfrac{1}{2}\,\mathbf{V}^{\dagger}\,Y_h\,\mathbf{V}",
            font_size=36, color=YELLOW,
        ).move_to([0, -2.55, 0])
        re_box = SurroundingRectangle(re_eq, color=YELLOW, buff=0.2,
                                      corner_radius=0.08)
        self.play(FadeIn(re_lbl))
        self.play(Write(re_eq), Create(re_box))
        self.wait(2.4)

        self.play(FadeOut(VGroup(
            sub, step1, iv_eq, step2, p_eq,
            decomp_lbl, decomp_eq, re_lbl, re_eq, re_box,
        )))

        # ── Passive vs Active regime — spectral condition on Y_h ──────
        hdr = Text(
            "Dissipated power ≥ 0 for every drive V  ⟺  Y_h is positive semidefinite.",
            font_size=22, color=GRAY,
        ).move_to([0, 2.05, 0])
        self.play(FadeIn(hdr))
        self.wait(0.4)

        # Left panel: Passive
        p_box = RoundedRectangle(width=5.8, height=3.5, corner_radius=0.15,
                                 color=GREEN, stroke_width=2)\
            .move_to([-3.3, -0.55, 0])
        p_ttl = Text("Passive", font_size=24, color=GREEN)\
            .move_to([-3.3, 0.85, 0])
        p_cond = MathTex(
            r"\mathbf{V}^{\dagger} Y_h \mathbf{V} \;\ge\; 0 \;\;\forall\, \mathbf{V}",
            font_size=26, color=GREEN,
        ).move_to([-3.3, 0.1, 0])
        p_spec = MathTex(
            r"\lambda_1,\,\lambda_2 \;\ge\; 0",
            font_size=24, color=GREEN,
        ).move_to([-3.3, -0.7, 0])
        p_phys = Text("no net power generation",
                      font_size=18, color=GREEN).move_to([-3.3, -1.55, 0])
        # small eigenvalue bars
        p_bar1 = Rectangle(width=0.25, height=0.6, fill_color=GREEN,
                           fill_opacity=0.8, stroke_width=0)\
            .move_to([-3.6, -2.0, 0])
        p_bar2 = Rectangle(width=0.25, height=0.95, fill_color=GREEN,
                           fill_opacity=0.8, stroke_width=0)\
            .move_to([-3.0, -1.83, 0])

        # Right panel: Active
        a_box = RoundedRectangle(width=5.8, height=3.5, corner_radius=0.15,
                                 color=ORANGE, stroke_width=2)\
            .move_to([3.3, -0.55, 0])
        a_ttl = Text("Active", font_size=24, color=ORANGE)\
            .move_to([3.3, 0.85, 0])
        a_cond = MathTex(
            r"\exists\,\mathbf{V}:\;\;\mathbf{V}^{\dagger} Y_h \mathbf{V} \;<\; 0",
            font_size=26, color=ORANGE,
        ).move_to([3.3, 0.1, 0])
        a_spec = MathTex(
            r"\lambda_{\min}(Y_h) \;<\; 0",
            font_size=24, color=ORANGE,
        ).move_to([3.3, -0.7, 0])
        a_phys = Text("drive that extracts power = gain",
                      font_size=18, color=ORANGE).move_to([3.3, -1.55, 0])
        a_bar1 = Rectangle(width=0.25, height=0.5, fill_color=ORANGE,
                           fill_opacity=0.8, stroke_width=0)\
            .move_to([2.7, -1.88, 0])
        # negative eigenvalue — bar drawn BELOW the baseline
        a_bar2 = Rectangle(width=0.25, height=0.55, fill_color=RED,
                           fill_opacity=0.85, stroke_width=0)\
            .move_to([3.3, -2.4, 0])
        a_axis = Line([2.3, -2.13, 0], [3.9, -2.13, 0],
                      color=GRAY, stroke_width=1)

        self.play(Create(p_box), Create(a_box),
                  FadeIn(p_ttl), FadeIn(a_ttl))
        self.play(Write(p_cond), Write(a_cond))
        self.play(Write(p_spec), Write(a_spec),
                  FadeIn(p_phys), FadeIn(a_phys))
        self.play(FadeIn(VGroup(p_bar1, p_bar2, a_bar1, a_bar2, a_axis)))
        self.wait(2.2)

        self.play(FadeOut(VGroup(
            hdr,
            p_box, p_ttl, p_cond, p_spec, p_phys, p_bar1, p_bar2,
            a_box, a_ttl, a_cond, a_spec, a_phys, a_bar1, a_bar2, a_axis,
        )))

        # ── Mason's U as the boundary quantity ─────────────────────────
        lead = Text(
            "Mason's unilateral power gain measures exactly this spectral excess:",
            font_size=22, color=GRAY,
        ).move_to([0, 2.1, 0])
        self.play(FadeIn(lead))

        u_eq = MathTex(
            r"U \;=\; \frac{|\,Y_{21}-Y_{12}\,|^{\,2}}{4\,\det(Y_h)}",
            font_size=40,
        ).move_to([0, 1.05, 0])
        u_box = SurroundingRectangle(u_eq, color=YELLOW, buff=0.22,
                                     corner_radius=0.1)
        self.play(Write(u_eq), Create(u_box))
        self.wait(0.8)

        det_form = MathTex(
            r"\det(Y_h) \;=\; \mathrm{Re}\{Y_{11}\}\mathrm{Re}\{Y_{22}\}"
            r"\;-\;\tfrac{1}{4}|\,Y_{12}+Y_{21}^{*}\,|^{\,2}",
            font_size=24, color=GRAY,
        ).move_to([0, 0.05, 0])
        self.play(FadeIn(det_form))
        self.wait(0.6)

        interp = VGroup(
            MathTex(r"\det(Y_h) > 0:\;\;\text{passive  →  } 0 \le U < \infty",
                    font_size=24, color=GREEN),
            MathTex(r"\det(Y_h) = 0:\;\;\text{boundary  →  } U = 1 \;\;(f_{\max})",
                    font_size=24, color=YELLOW),
            MathTex(r"\det(Y_h) < 0:\;\;\text{active  →  } U > 1",
                    font_size=24, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -1.35, 0])
        for row in interp:
            self.play(FadeIn(row), run_time=0.6)
        self.wait(0.4)

        tag = Text(
            "U = 1 is the passivity boundary of Y_h  —  this defines f_max.",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.28)
        self.play(FadeIn(tag))
        self.wait(2.6)

        self.play(FadeOut(VGroup(
            title, lead, u_eq, u_box, det_form, interp, tag,
        )))
