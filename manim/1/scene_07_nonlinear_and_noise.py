from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Scene 7 — Beyond Small-Signal:
#   large-signal behaviour, non-linearity, noise, memory effects
# ---------------------------------------------------------------------------

class NonlinearAndNoise(Scene):

    def construct(self):
        self.act1_title()
        self.act2_why_small_signal_breaks()
        self.act3_gain_compression()
        self.act4_harmonic_distortion()
        self.act5_two_tone_ip3()
        self.act6_am_pm_distortion()
        self.act7_noise_figure()
        self.act8_beyond()

    # ── Act 1 ─────────────────────────────────────────────────────────────
    def act1_title(self):
        title = Text("Beyond Small-Signal", font_size=44)
        sub = Text("non-linearity, distortion, noise, and memory",
                   font_size=26, color=GRAY)
        sub.next_to(title, DOWN, buff=0.35)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── Act 2: why the linear model breaks ────────────────────────────────
    def act2_why_small_signal_breaks(self):
        title = Text("Why the Linear S-Parameter Model Breaks",
                     font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title))

        lin = MathTex(
            r"b_i = \sum_j S_{ij}\, a_j "
            r"\quad\text{valid only for}\quad |a_j|\ll 1",
            font_size=30, color=GRAY,
        ).move_to([0, 1.8, 0])
        self.play(Write(lin))
        self.wait(0.5)

        poly = MathTex(
            r"v_{\rm out}(t) = a_1 v_{\rm in}(t) + a_2 v_{\rm in}^2(t)"
            r" + a_3 v_{\rm in}^3(t) + \cdots",
            font_size=32,
        ).move_to([0, 0.7, 0])
        poly_box = SurroundingRectangle(poly, color=YELLOW, buff=0.2, corner_radius=0.1)
        self.play(Write(poly))
        self.play(Create(poly_box))
        self.wait(0.8)

        effects = VGroup(
            Text("•  gain compression   (|a₃| opposes a₁)", font_size=24, color=BLUE),
            Text("•  harmonic distortion   (nω₀ tones)",    font_size=24, color=GREEN),
            Text("•  intermodulation   (2ω₁−ω₂ in-band)",   font_size=24, color=RED),
            Text("•  AM-PM distortion   (phase vs power)",   font_size=24, color=PURPLE),
            Text("•  noise   (adds independent of signal)", font_size=24, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, -1.6, 0])

        for e in effects:
            self.play(FadeIn(e, shift=RIGHT * 0.2), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, lin, poly, poly_box, effects)))

    # ── Act 3: gain compression / P1dB ────────────────────────────────────
    def act3_gain_compression(self):
        title = Text("Gain Compression and the 1-dB Point",
                     font_size=30).to_edge(UP, buff=0.4)
        self.play(Write(title))

        model = MathTex(
            r"P_{\rm out}(P_{\rm in}) \approx G_0\,P_{\rm in}"
            r"\cdot\frac{1}{1 + P_{\rm in}/P_{\rm sat}}",
            font_size=28, color=GRAY,
        ).next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(model))

        # Pin in dBm (-30 .. 20), Pout in dBm
        G0_dB = 20.0
        Psat_dBm = 15.0

        def pout_dBm(pin_dBm):
            pin = 10 ** (pin_dBm / 10)
            psat = 10 ** (Psat_dBm / 10)
            pout = (10 ** (G0_dB / 10)) * pin / (1 + pin / psat)
            return 10 * np.log10(pout)

        ax = Axes(
            x_range=[-30, 20, 10], y_range=[-10, 40, 10],
            x_length=8, y_length=4, tips=False,
            axis_config={"stroke_width": 2, "include_numbers": True},
        ).move_to([0, -1.0, 0])
        xl = MathTex(r"P_{\rm in}\ (\text{dBm})", font_size=22).next_to(ax, DOWN, buff=0.2)
        yl = MathTex(r"P_{\rm out}\ (\text{dBm})", font_size=22).next_to(ax, LEFT, buff=0.2)

        linear_curve = ax.plot(lambda p: p + G0_dB, x_range=[-30, 20, 0.5],
                               color=GRAY, stroke_width=2)
        linear_curve.set_stroke(opacity=0.7)
        actual = ax.plot(pout_dBm, x_range=[-30, 20, 0.2],
                         color=BLUE, stroke_width=3)

        # Solve P1dB numerically
        grid = np.linspace(-30, 20, 2001)
        diff = np.array([(g + G0_dB) - pout_dBm(g) for g in grid])
        idx = np.argmin(np.abs(diff - 1.0))
        p1db_in = grid[idx]
        p1db_out = pout_dBm(p1db_in)

        p1db_dot = Dot(ax.c2p(p1db_in, p1db_out), color=RED, radius=0.08)
        p1db_lbl = MathTex(r"P_{1\rm dB}", font_size=22, color=RED)\
            .next_to(p1db_dot, UR, buff=0.1)
        p1db_v = DashedLine(ax.c2p(p1db_in, -10), ax.c2p(p1db_in, p1db_out),
                            color=RED, stroke_width=1.5)

        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(linear_curve), Create(actual))
        self.play(FadeIn(p1db_v), Create(p1db_dot), FadeIn(p1db_lbl))

        # Operating-point tracker
        op = ValueTracker(-20.0)
        dot = always_redraw(lambda: Dot(
            ax.c2p(op.get_value(), pout_dBm(op.get_value())),
            color=YELLOW, radius=0.09,
        ))
        gain_read = always_redraw(lambda: MathTex(
            rf"P_{{\rm in}}={op.get_value():.1f}\,\text{{dBm}},\ "
            rf"G={pout_dBm(op.get_value()) - op.get_value():.2f}\,\text{{dB}}",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.25))
        self.add(dot, gain_read)
        self.play(op.animate.set_value(18.0), run_time=4, rate_func=linear)
        self.wait(0.4)

        for m in (dot, gain_read):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, model, ax, xl, yl, linear_curve, actual,
                                 p1db_dot, p1db_lbl, p1db_v, dot, gain_read)))

    # ── Act 4: harmonic distortion ────────────────────────────────────────
    def act4_harmonic_distortion(self):
        title = Text("Harmonic Distortion",
                     font_size=30).to_edge(UP, buff=0.4)
        self.play(Write(title))

        eq = MathTex(
            r"v_{\rm in}=A\cos\omega t\ \Rightarrow\ "
            r"v_{\rm out}=\tfrac{a_2 A^2}{2}"
            r"+\!\left(a_1 A+\tfrac{3a_3 A^3}{4}\right)\!\cos\omega t"
            r"+\tfrac{a_2 A^2}{2}\cos 2\omega t"
            r"+\tfrac{a_3 A^3}{4}\cos 3\omega t+\cdots",
            font_size=22, color=GRAY,
        ).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(eq))

        # Bar-chart spectrum driven by amplitude tracker A
        A = ValueTracker(0.3)
        a1, a2, a3 = 1.0, 0.4, 0.3

        ax = Axes(
            x_range=[0, 4.5, 1], y_range=[0, 1.2, 0.25],
            x_length=9, y_length=3.6, tips=False,
            axis_config={"stroke_width": 2, "include_numbers": False},
        ).move_to([0, -1.0, 0])
        xl = MathTex(r"f/f_0", font_size=22).next_to(ax, DOWN, buff=0.2)
        yl = MathTex(r"|V|\ \text{(normalised)}", font_size=22).next_to(ax, LEFT, buff=0.2)

        tick_lbls = VGroup(*[
            MathTex(fr"{k}f_0", font_size=20).next_to(ax.c2p(k, 0), DOWN, buff=0.1)
            for k in (1, 2, 3, 4)
        ])

        def bar(freq, amp, color):
            top = min(amp, 1.15)
            return Line(ax.c2p(freq, 0), ax.c2p(freq, top),
                        color=color, stroke_width=8)

        def spectrum():
            a_v = A.get_value()
            h1 = abs(a1 * a_v + 0.75 * a3 * a_v ** 3)
            h2 = abs(0.5 * a2 * a_v ** 2)
            h3 = abs(0.25 * a3 * a_v ** 3)
            norm = max(h1, 1e-6)
            return VGroup(
                bar(1, h1 / norm, BLUE),
                bar(2, h2 / norm, GREEN),
                bar(3, h3 / norm, RED),
            )

        bars = always_redraw(spectrum)
        readout = always_redraw(lambda: MathTex(
            rf"A={A.get_value():.2f},\ "
            rf"\text{{THD}}=\sqrt{{H_2^2+H_3^2}}/H_1"
            rf"={self._thd(A.get_value(), a1, a2, a3) * 100:.1f}\%",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.25))

        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(tick_lbls))
        self.add(bars, readout)
        self.play(A.animate.set_value(1.4), run_time=4, rate_func=linear)
        self.wait(0.4)
        self.play(A.animate.set_value(0.1), run_time=2, rate_func=linear)
        self.wait(0.3)

        for m in (bars, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, eq, ax, xl, yl, tick_lbls, bars, readout)))

    @staticmethod
    def _thd(A, a1, a2, a3):
        h1 = abs(a1 * A + 0.75 * a3 * A ** 3)
        h2 = abs(0.5 * a2 * A ** 2)
        h3 = abs(0.25 * a3 * A ** 3)
        return np.sqrt(h2 ** 2 + h3 ** 2) / max(h1, 1e-9)

    # ── Act 5: two-tone test and IP3 ──────────────────────────────────────
    def act5_two_tone_ip3(self):
        title = Text("Two-Tone Intermodulation and IP3",
                     font_size=30).to_edge(UP, buff=0.4)
        self.play(Write(title))

        setup = MathTex(
            r"v_{\rm in}=A(\cos\omega_1 t + \cos\omega_2 t)\ \Rightarrow\ "
            r"\text{IM3 at }2\omega_1-\omega_2,\ 2\omega_2-\omega_1"
            r"\ \propto A^3",
            font_size=22, color=GRAY,
        ).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(setup))

        # Fundamental slope 1 (dB/dB), IM3 slope 3 → meet at IP3
        G_dB = 10.0
        IIP3_dBm = 10.0  # input intercept
        OIP3_dBm = IIP3_dBm + G_dB

        def p_fund(pin):
            return pin + G_dB
        def p_im3(pin):
            return 3 * pin - 2 * IIP3_dBm + G_dB

        ax = Axes(
            x_range=[-40, 20, 10], y_range=[-90, 30, 20],
            x_length=8, y_length=4, tips=False,
            axis_config={"stroke_width": 2, "include_numbers": True},
        ).move_to([0, -1.1, 0])
        xl = MathTex(r"P_{\rm in}\ (\text{dBm})", font_size=22).next_to(ax, DOWN, buff=0.2)
        yl = MathTex(r"P_{\rm out}\ (\text{dBm})", font_size=22).next_to(ax, LEFT, buff=0.2)

        fund = ax.plot(p_fund, x_range=[-40, 15, 0.5], color=BLUE, stroke_width=3)
        im3  = ax.plot(p_im3,  x_range=[-40, 15, 0.5], color=RED,  stroke_width=3)
        fund_lbl = MathTex(r"\text{fundamental, slope }1", font_size=20, color=BLUE)\
            .next_to(ax.c2p(-20, p_fund(-20)), UL, buff=0.1)
        im3_lbl  = MathTex(r"\text{IM3, slope }3", font_size=20, color=RED)\
            .next_to(ax.c2p(-10, p_im3(-10)), DR, buff=0.1)

        ip3_dot = Dot(ax.c2p(IIP3_dBm, OIP3_dBm), color=YELLOW, radius=0.1)
        ip3_lbl = MathTex(r"\text{IP3}", font_size=24, color=YELLOW)\
            .next_to(ip3_dot, UR, buff=0.1)
        ip3_v = DashedLine(ax.c2p(IIP3_dBm, -90), ax.c2p(IIP3_dBm, OIP3_dBm),
                           color=YELLOW, stroke_width=1.5)

        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(fund), FadeIn(fund_lbl))
        self.play(Create(im3), FadeIn(im3_lbl))
        self.play(Create(ip3_v), Create(ip3_dot), FadeIn(ip3_lbl))

        # Operating point with SFDR readout
        op = ValueTracker(-30.0)
        d_fund = always_redraw(lambda: Dot(
            ax.c2p(op.get_value(), p_fund(op.get_value())),
            color=BLUE, radius=0.08))
        d_im3 = always_redraw(lambda: Dot(
            ax.c2p(op.get_value(), p_im3(op.get_value())),
            color=RED, radius=0.08))
        gap = always_redraw(lambda: DashedLine(
            ax.c2p(op.get_value(), p_fund(op.get_value())),
            ax.c2p(op.get_value(), p_im3(op.get_value())),
            color=YELLOW, stroke_width=1.5))
        readout = always_redraw(lambda: MathTex(
            rf"P_{{\rm in}}={op.get_value():.1f}\,\text{{dBm}},\ "
            rf"\Delta=2(\text{{IIP3}}-P_{{\rm in}})"
            rf"={2*(IIP3_dBm-op.get_value()):.1f}\,\text{{dB}}",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.25))

        self.add(d_fund, d_im3, gap, readout)
        self.play(op.animate.set_value(5.0), run_time=4, rate_func=linear)
        self.wait(0.5)

        for m in (d_fund, d_im3, gap, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, setup, ax, xl, yl,
                                 fund, im3, fund_lbl, im3_lbl,
                                 ip3_dot, ip3_lbl, ip3_v,
                                 d_fund, d_im3, gap, readout)))

    # ── Act 6: AM-PM distortion ───────────────────────────────────────────
    def act6_am_pm_distortion(self):
        title = Text("AM–PM Distortion",
                     font_size=30).to_edge(UP, buff=0.4)
        self.play(Write(title))

        eq = MathTex(
            r"\phi_{\rm out}(P_{\rm in}) = \phi_0 + k_{\rm pm}\,P_{\rm in}"
            r"\quad(\text{deg}/\text{dB at large signal})",
            font_size=24, color=GRAY,
        ).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(eq))

        # Phase shift vs input power — saturating sigmoid
        def phi(pin_dBm):
            return 45.0 / (1 + np.exp(-(pin_dBm - 5.0) / 3.0))

        ax = Axes(
            x_range=[-20, 20, 5], y_range=[-5, 55, 10],
            x_length=8, y_length=3.8, tips=False,
            axis_config={"stroke_width": 2, "include_numbers": True},
        ).move_to([-1.8, -1.2, 0])
        xl = MathTex(r"P_{\rm in}\ (\text{dBm})", font_size=22).next_to(ax, DOWN, buff=0.2)
        yl = MathTex(r"\Delta\phi\ (^\circ)", font_size=22).next_to(ax, LEFT, buff=0.2)

        curve = ax.plot(phi, x_range=[-20, 20, 0.2], color=PURPLE, stroke_width=3)

        # Phasor diagram on the right that rotates with drive level
        pc = np.array([4.6, -1.2, 0])
        unit = Circle(radius=1.2, color=GRAY, stroke_width=1.5).move_to(pc)
        origin = Dot(pc, color=WHITE, radius=0.03)

        op = ValueTracker(-18.0)
        phasor = always_redraw(lambda: Arrow(
            pc, pc + 1.2 * np.array([np.cos(np.radians(phi(op.get_value()))),
                                     np.sin(np.radians(phi(op.get_value()))), 0]),
            color=PURPLE, stroke_width=4, buff=0,
            tip_length=0.18, max_tip_length_to_length_ratio=0.2,
        ))
        op_dot = always_redraw(lambda: Dot(
            ax.c2p(op.get_value(), phi(op.get_value())),
            color=YELLOW, radius=0.08))
        readout = always_redraw(lambda: MathTex(
            rf"P_{{\rm in}}={op.get_value():.1f}\,\text{{dBm}},\ "
            rf"\Delta\phi={phi(op.get_value()):.1f}^\circ",
            font_size=22, color=YELLOW,
        ).to_edge(DOWN, buff=0.25))

        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(unit), FadeIn(origin))
        self.play(Create(curve))
        self.add(phasor, op_dot, readout)
        self.play(op.animate.set_value(18.0), run_time=4, rate_func=linear)
        self.wait(0.4)

        for m in (phasor, op_dot, readout):
            m.clear_updaters()
        self.play(FadeOut(VGroup(title, eq, ax, xl, yl, curve,
                                 unit, origin, phasor, op_dot, readout)))

    # ── Act 7: noise figure ───────────────────────────────────────────────
    def act7_noise_figure(self):
        title = Text("Noise Figure and Cascaded Noise (Friis)",
                     font_size=28).to_edge(UP, buff=0.4)
        self.play(Write(title))

        defn = MathTex(
            r"F = \frac{\text{SNR}_{\rm in}}{\text{SNR}_{\rm out}}"
            r"= 1 + \frac{T_e}{T_0},\qquad NF = 10\log_{10} F",
            font_size=30,
        ).move_to([0, 1.8, 0])
        friis = MathTex(
            r"F_{\rm tot} = F_1 + \frac{F_2-1}{G_1}"
            r"+ \frac{F_3-1}{G_1 G_2} + \cdots",
            font_size=32, color=YELLOW,
        ).move_to([0, 0.7, 0])
        friis_box = SurroundingRectangle(friis, color=YELLOW, buff=0.2, corner_radius=0.1)

        self.play(Write(defn))
        self.play(Write(friis), Create(friis_box))
        self.wait(0.6)

        # Cascade diagram with swappable stage-1 NF
        def stage(center, g, f, col):
            r = Rectangle(width=1.6, height=1.0, color=col, stroke_width=2).move_to(center)
            t = VGroup(
                MathTex(rf"G_1={g}\,\text{{dB}}" if col == BLUE else rf"G_2={g}\,\text{{dB}}",
                        font_size=20, color=col),
                MathTex(rf"F={f}", font_size=20, color=col),
            ).arrange(DOWN, buff=0.08).move_to(r)
            return VGroup(r, t)

        s1 = stage([-2.2, -1.3, 0], 15, 2.0, BLUE)
        s2 = stage([ 2.2, -1.3, 0], 20, 5.0, GREEN)
        wires = VGroup(
            Line([-4.0, -1.3, 0], [-3.0, -1.3, 0], color=WHITE, stroke_width=2),
            Line([-1.4, -1.3, 0], [ 1.4, -1.3, 0], color=WHITE, stroke_width=2),
            Line([ 3.0, -1.3, 0], [ 4.0, -1.3, 0], color=WHITE, stroke_width=2),
        )
        self.play(FadeIn(s1), FadeIn(s2), Create(wires))

        F1, F2, G1_lin = 2.0, 5.0, 10 ** 1.5
        F_tot = F1 + (F2 - 1) / G1_lin
        NF_tot = 10 * np.log10(F_tot)
        concl = MathTex(
            rf"F_{{\rm tot}}={F1} + \frac{{{F2}-1}}{{10^{{1.5}}}}"
            rf"\approx {F_tot:.2f},\quad NF_{{\rm tot}}\approx{NF_tot:.2f}\,\text{{dB}}",
            font_size=26, color=GREEN,
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(concl))
        self.wait(2.5)

        self.play(FadeOut(VGroup(title, defn, friis, friis_box,
                                 s1, s2, wires, concl)))

    # ── Act 8: beyond — memory, X-parameters, load-pull ───────────────────
    def act8_beyond(self):
        title = Text("Beyond: what this video does not cover",
                     font_size=28).to_edge(UP, buff=0.4)
        self.play(Write(title))

        items = [
            ("Volterra series",
             r"y(t)=\sum_{n=1}^{\infty}\!\int\!\cdots\!\int "
             r"h_n(\tau_1,\dots,\tau_n)\prod_k x(t-\tau_k)\,d\tau_k",
             "captures both nonlinearity and memory",
             BLUE),
            ("X-parameters",
             r"B_{p,k}=X_{p,k}^{F}(|A_{1,1}|)P^{k}"
             r"+\sum_{q,l}X_{p,k,q,l}^{S}(|A_{1,1}|)P^{k-l}A_{q,l}",
             "extension of S-params to large-signal (Agilent)",
             GREEN),
            ("Load-pull",
             r"\text{measure } P_{\rm out},\ \eta,\ IM3\text{ vs }\Gamma_L",
             "maps device performance across Γ_L plane",
             RED),
            ("Behavioural models",
             r"\text{memory polynomial, neural, Wiener-Hammerstein}",
             "for DPD and system-level simulation",
             PURPLE),
        ]

        rows = VGroup()
        for name, eq_str, note, col in items:
            hdr = Text(name, font_size=22, color=col)
            eq  = MathTex(eq_str, font_size=22)
            nt  = Text("— " + note, font_size=18, color=GRAY)
            rows.add(VGroup(hdr, eq, nt).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT).move_to([0, -0.3, 0])

        for r in rows:
            self.play(FadeIn(r, shift=UP * 0.15), run_time=0.6)
            self.wait(0.4)

        end = Text("— end of video 1 —", font_size=22, color=YELLOW)\
            .to_edge(DOWN, buff=0.35)
        self.play(FadeIn(end))
        self.wait(3)
        self.play(FadeOut(VGroup(title, rows, end)))
