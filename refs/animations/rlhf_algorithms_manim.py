from __future__ import annotations

from manim import *


config.background_color = "#10131a"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30


POLICY = "#6ea8fe"
REF = "#9aa4b2"
REWARD = "#62d26f"
ADV = "#f5b14c"
GUARD = "#ff6b6b"
TEXT = "#edf2f7"
MUTED = "#aab4c2"
PANEL = "#171b24"
PANEL_2 = "#202636"


class RLHFAlgorithmsExplainer(MovingCameraScene):
    """Math-to-code explainer for DPO, GRPO, PPO, and RLOO.

    Render:
        uv run --no-project --with manim manim -pql refs/animations/rlhf_algorithms_manim.py RLHFAlgorithmsExplainer
    High quality:
        uv run --no-project --with manim manim -pqh refs/animations/rlhf_algorithms_manim.py RLHFAlgorithmsExplainer
    """

    def construct(self) -> None:
        self.camera.background_color = "#10131a"
        self.intro()
        self.dpo_scene()
        self.policy_rl_scene()
        self.algorithm_cards()
        self.project_map()

    def intro(self) -> None:
        title = Text("RLHF-style post-training", font_size=48, color=TEXT, weight=BOLD)
        subtitle = Text(
            "from preference math to Baby Whale v4 training loops",
            font_size=25,
            color=MUTED,
        )
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.18)
        header.to_edge(UP, buff=0.45)

        stages = VGroup(
            stage_box("SFT", "imitate good traces", POLICY),
            stage_box("DPO", "learn from chosen > rejected", REWARD),
            stage_box("Policy RL", "sample, score, update", ADV),
        ).arrange(RIGHT, buff=0.8)
        stages.move_to(ORIGIN + DOWN * 0.25)
        arrows = VGroup(
            Arrow(stages[0].get_right(), stages[1].get_left(), buff=0.18, color=MUTED),
            Arrow(stages[1].get_right(), stages[2].get_left(), buff=0.18, color=MUTED),
        )

        note = Text(
            "RLHF is the family. DPO skips online RL. GRPO/PPO/RLOO update the policy from rollouts.",
            font_size=24,
            color=TEXT,
        ).to_edge(DOWN, buff=0.65)

        self.play(FadeIn(header, shift=DOWN * 0.25), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in stages], lag_ratio=0.18), run_time=1.1)
        self.play(Create(arrows), FadeIn(note, shift=UP * 0.2), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(VGroup(header, stages, arrows, note), shift=UP * 0.2), run_time=0.7)

    def dpo_scene(self) -> None:
        title = section_title("1. DPO: preference optimization without rollout RL")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)

        left = panel(
            "Data",
            [
                "prompt x",
                "chosen y+",
                "rejected y-",
                "reference policy pi_ref",
            ],
            color=REWARD,
            width=3.55,
        )
        mid = panel(
            "Preference margin",
            [
                "m = beta * [",
                "  log pi_theta(y+|x) - log pi_theta(y-|x)",
                "  - log pi_ref(y+|x) + log pi_ref(y-|x)",
                "]",
            ],
            color=POLICY,
            width=5.85,
        )
        right = panel(
            "Loss",
            [
                "L_DPO = -log sigmoid(m)",
                "chosen higher -> m up -> loss down",
                "rejected higher -> m down -> loss up",
            ],
            color=ADV,
            width=3.55,
        )
        row = VGroup(left, mid, right).arrange(RIGHT, buff=0.34)
        row.next_to(title, DOWN, buff=0.45)
        arrow1 = Arrow(left.get_right(), mid.get_left(), buff=0.12, color=MUTED)
        arrow2 = Arrow(mid.get_right(), right.get_left(), buff=0.12, color=MUTED)

        self.play(LaggedStart(FadeIn(left), Create(arrow1), FadeIn(mid), Create(arrow2), FadeIn(right), lag_ratio=0.18), run_time=1.8)

        code = code_panel(
            "baby_whale_v4/training/dpo.py",
            [
                "pi_pos  = logp(model, prompt, chosen)",
                "pi_neg  = logp(model, prompt, rejected)",
                "ref_pos = logp(ref,   prompt, chosen)",
                "ref_neg = logp(ref,   prompt, rejected)",
                "margin = beta * ((pi_pos - pi_neg) - (ref_pos - ref_neg))",
                "loss = -log_sigmoid(margin).mean()",
            ],
            width=12.5,
        )
        code.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(code, shift=UP * 0.25), run_time=0.8)

        pointer = Arrow(right.get_bottom(), code.get_top() + RIGHT * 4.5, buff=0.14, color=ADV)
        label = Text("math -> code", font_size=23, color=ADV, font="Menlo")
        label.next_to(pointer, LEFT, buff=0.16).shift(UP * 0.35)
        self.play(Create(pointer), FadeIn(label), run_time=0.7)
        self.wait(1.0)
        self.play(FadeOut(VGroup(title, row, arrow1, arrow2, code, pointer, label), shift=LEFT * 0.3), run_time=0.8)

    def policy_rl_scene(self) -> None:
        title = section_title("2. Policy RL: generate, score, then push likely good tokens up")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)

        boxes = VGroup(
            stage_box("Policy pi_theta", "sample responses", POLICY, width=3.0),
            stage_box("RewardHost", "score responses", REWARD, width=3.0),
            stage_box("Advantage A", "center rewards", ADV, width=3.0),
            stage_box("Update", "push logprobs", GUARD, width=3.0),
        ).arrange(RIGHT, buff=0.25)
        boxes.next_to(title, DOWN, buff=0.65)
        arrows = VGroup(
            Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1, color=MUTED)
            for i in range(3)
        )

        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.15) for b in boxes], lag_ratio=0.12), run_time=1.1)
        self.play(Create(arrows), run_time=0.8)

        formula = panel(
            "Common policy-gradient shape",
            [
                "loss ~= - A * log pi_theta(token | prefix)",
                "A > 0: make sampled tokens more likely",
                "A < 0: make sampled tokens less likely",
                "KL keeps pi_theta near the reference policy",
            ],
            color=ADV,
            width=8.6,
        )
        formula.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(formula, shift=UP * 0.25), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(VGroup(title, boxes, arrows, formula), shift=UP * 0.2), run_time=0.75)

    def algorithm_cards(self) -> None:
        title = section_title("3. GRPO / PPO / RLOO: same rollout loop, different control")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)

        grpo = algo_card(
            "GRPO",
            REWARD,
            [
                "sample G responses for one prompt",
                "A_i = (r_i - mean(r_group)) / std(r_group)",
                "no critic/value model",
                "DeepSeek-style fit for verifier rewards",
            ],
            [
                "samples = rollout_group(prompt, G)",
                "rewards = reward_host.score_batch(samples)",
                "adv = normalize(rewards)",
                "loss = -mean(adv * new_logp)",
            ],
        )
        ppo = algo_card(
            "PPO",
            GUARD,
            [
                "compare new policy to rollout-time old policy",
                "ratio = exp(logp_new - logp_old)",
                "clip ratio to stop destructive jumps",
                "usually pairs with KL and sometimes a value model",
            ],
            [
                "ratio = exp(new_logp - old_logp)",
                "s1 = ratio * adv",
                "s2 = clip(ratio, 1-eps, 1+eps) * adv",
                "loss = -mean(min(s1, s2)) + beta_kl * kl",
            ],
        )
        rloo = algo_card(
            "RLOO",
            ADV,
            [
                "sample a group for the same prompt",
                "baseline for item i excludes item i",
                "A_i = r_i - mean(r_j for j != i)",
                "simple, critic-free variance reduction",
            ],
            [
                "for i in group:",
                "    base = mean(rewards except i)",
                "    adv_i = rewards[i] - base",
                "loss = -mean(adv * new_logp)",
            ],
        )
        cards = VGroup(grpo, ppo, rloo).arrange(RIGHT, buff=0.35)
        cards.next_to(title, DOWN, buff=0.45)

        self.play(FadeIn(grpo, shift=RIGHT * 0.35), run_time=0.8)
        self.play(Indicate(grpo[0], color=REWARD), run_time=0.55)
        self.play(FadeIn(ppo, shift=RIGHT * 0.35), run_time=0.8)
        self.play(Indicate(ppo[0], color=GUARD), run_time=0.55)
        self.play(FadeIn(rloo, shift=RIGHT * 0.35), run_time=0.8)
        self.play(Indicate(rloo[0], color=ADV), run_time=0.55)
        self.wait(1.1)
        self.play(FadeOut(VGroup(title, cards), shift=LEFT * 0.25), run_time=0.8)

    def project_map(self) -> None:
        title = section_title("4. Baby Whale v4 map: where each idea lives")
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.6)

        pipeline = VGroup(
            stage_box("SFT", "training/sft.py", POLICY, width=2.45),
            stage_box("DPO", "training/dpo.py", REWARD, width=2.45),
            stage_box("GRPO", "training/grpo.py", ADV, width=2.45),
            stage_box("PPO", "training/ppo.py", GUARD, width=2.45),
            stage_box("RLOO", "training/rloo.py", ADV, width=2.45),
        ).arrange(RIGHT, buff=0.18)
        pipeline.next_to(title, DOWN, buff=0.6)
        arrows = VGroup(
            Arrow(pipeline[i].get_right(), pipeline[i + 1].get_left(), buff=0.08, color=MUTED)
            for i in range(4)
        )

        infra = panel(
            "Shared RL infra",
            [
                "rl/rollout.py: chunked prefill + prefix cache + captured logprobs",
                "rl/reward_host.py: local or HTTP rewards",
                "rl/code_reward.py: verifier reward for generated code",
                "tools/: tool-call parsing and deterministic local tools",
            ],
            color=POLICY,
            width=12.8,
        )
        infra.to_edge(DOWN, buff=0.75)

        final = text_lines(
            [
                "Classic learned reward-model RLHF is not implemented yet.",
                "Current path: verifier rewards + DPO/GRPO/PPO/RLOO policy updates.",
            ],
            font_size=22,
            color=TEXT,
            font="Menlo",
        )
        if final.width > 12.7:
            final.scale_to_fit_width(12.7)
        final.next_to(infra, UP, buff=0.5)

        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.15) for p in pipeline], lag_ratio=0.08), run_time=1.0)
        self.play(Create(arrows), FadeIn(infra, shift=UP * 0.2), run_time=1.0)
        self.play(FadeIn(final), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, pipeline, arrows, infra, final)), run_time=0.8)


def section_title(text: str) -> Text:
    mob = Text(text, font_size=34, color=TEXT, weight=BOLD)
    if mob.width > 13.1:
        mob.scale_to_fit_width(13.1)
    return mob.to_edge(UP, buff=0.38)


def stage_box(title: str, subtitle: str, color: str, width: float = 3.7) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=0.18,
        width=width,
        height=1.55,
        fill_color=PANEL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=2.4,
    )
    title_m = Text(title, font_size=25, color=color, weight=BOLD)
    subtitle_kwargs = {"font_size": 18, "color": TEXT}
    if "/" in subtitle or "." in subtitle:
        subtitle_kwargs["font"] = "Menlo"
        subtitle_kwargs["font_size"] = 15
    subtitle_m = Text(subtitle, **subtitle_kwargs)
    content = VGroup(title_m, subtitle_m).arrange(DOWN, buff=0.16)
    if content.width > width - 0.35:
        content.scale_to_fit_width(width - 0.35)
    content.move_to(rect.get_center())
    return VGroup(rect, content)


def panel(title: str, lines: list[str], color: str, width: float = 5.2) -> VGroup:
    body = text_lines(lines, font_size=18, color=TEXT, font="Menlo")
    title_m = Text(title, font_size=24, color=color, weight=BOLD, font="Menlo")
    content = VGroup(title_m, body).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
    if content.width > width - 0.48:
        content.scale_to_fit_width(width - 0.48)
    rect = RoundedRectangle(
        corner_radius=0.16,
        width=width,
        height=max(1.8, content.height + 0.58),
        fill_color=PANEL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=2.0,
    )
    content.move_to(rect.get_center()).align_to(rect, LEFT).shift(RIGHT * 0.28)
    return VGroup(rect, content)


def code_panel(title: str, lines: list[str], width: float = 10.5) -> VGroup:
    title_m = Text(title, font_size=20, color=MUTED, font="Menlo")
    body = text_lines(lines, font_size=17, color=TEXT, font="Menlo")
    content = VGroup(title_m, body).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    if content.width > width - 0.55:
        content.scale_to_fit_width(width - 0.55)
    rect = RoundedRectangle(
        corner_radius=0.14,
        width=width,
        height=content.height + 0.6,
        fill_color=PANEL_2,
        fill_opacity=1.0,
        stroke_color=REF,
        stroke_width=1.5,
    )
    content.move_to(rect.get_center()).align_to(rect, LEFT).shift(RIGHT * 0.32)
    return VGroup(rect, content)


def algo_card(title: str, color: str, math_lines: list[str], code_lines: list[str]) -> VGroup:
    title_m = Text(title, font_size=28, color=color, weight=BOLD)
    math = text_lines(math_lines, font_size=16, color=TEXT)
    code = text_lines(code_lines, font_size=13, color="#dce7ff", font="Menlo")
    divider = Line(LEFT * 2.0, RIGHT * 2.0, color=REF, stroke_opacity=0.35)
    content = VGroup(title_m, math, divider, code).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    if content.width > 3.55:
        content.scale_to_fit_width(3.55)
    if content.height > 5.85:
        content.scale_to_fit_height(5.85)
    rect = RoundedRectangle(
        corner_radius=0.17,
        width=4.18,
        height=6.22,
        fill_color=PANEL,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=2.0,
    )
    content.move_to(rect.get_center()).align_to(rect, LEFT).shift(RIGHT * 0.28)
    return VGroup(rect, content)


def text_lines(lines: list[str], font_size: int, color: str, font: str | None = None) -> VGroup:
    rendered = []
    for line in lines:
        kwargs = {"font_size": font_size, "color": color, "disable_ligatures": True}
        if font is not None:
            kwargs["font"] = font
        rendered.append(Text(line, **kwargs))
    return VGroup(*rendered).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
