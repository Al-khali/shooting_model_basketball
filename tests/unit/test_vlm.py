"""
Tests for Phase 2: VLM module.

All tests are fully mocked — no real API calls, no API keys needed in CI.
"""

from __future__ import annotations

import pytest

from src.api.schemas.domain import (
    BiomechanicsReport,
    CoachingFeedback,
    JointAngle,
    PlayerLevel,
    SessionContext,
    ShotResult,
    ShotTimingMetrics,
)
from src.vlm.base import BaseVLMClient, Message, VLMConfig, VLMError, VLMParseError
from src.vlm.basketball_analyzer import BasketballVLMAnalyzer
from src.vlm.evaluator import DimensionScore, EvaluationResult, FeedbackEvaluator
from src.vlm.prompts.basketball import (
    BASKETBALL_COACH_SYSTEM_PROMPT,
    build_analysis_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_report(
    shot_result: ShotResult = ShotResult.MISSED,
    primary_issue: str | None = "Elbow angle too wide",
    alignment_score: float | None = 0.71,
) -> BiomechanicsReport:
    return BiomechanicsReport(
        shot_result=shot_result,
        joint_angles=[
            JointAngle(
                joint="right_elbow",
                value_deg=68.0,
                optimal_min=30.0,
                optimal_max=50.0,
            ),
            JointAngle(
                joint="right_knee",
                value_deg=162.0,
                optimal_min=150.0,
                optimal_max=170.0,
            ),
        ],
        timing=ShotTimingMetrics(
            jump_to_release_ms=320.0,
            early_release=False,
        ),
        alignment_score=alignment_score,
        primary_issue=primary_issue,
        issues_detected=["Elbow angle too wide at release"],
    )


def make_context(level: PlayerLevel = PlayerLevel.INTERMEDIATE) -> SessionContext:
    return SessionContext(
        player_id="player-42",
        player_level=level,
        session_count=3,
        recurring_issues=["Elbow flare on fatigue"],
        previous_drills=["Elbow tuck drill"],
    )


def make_good_feedback(report: BiomechanicsReport) -> CoachingFeedback:
    """A feedback that should pass all quality checks."""
    return CoachingFeedback(
        player_id="player-42",
        shot_result=report.shot_result,
        confidence=0.82,
        summary=(
            "Your elbow angle of 68° is significantly outside the optimal 30–50° range. "
            "This creates lateral deviation on the ball's flight path. "
            "Tuck your elbow to align with the basket and improve shot consistency."
        ),
        primary_correction=(
            "Tuck your shooting elbow under the ball at release — "
            "point your elbow toward the basket, not sideways."
        ),
        detailed_analysis=(
            "The right_elbow angle measured at 68° far exceeds the NBA-calibrated "
            "optimal range of 30–50°. This pushes the ball off the shooting line. "
            "Your right_knee at 162° is within the optimal range, meaning your "
            "lower body kinetic chain is solid. Alignment score of 71% confirms "
            "consistent mechanical compensation in the upper body. "
            "Focus all corrections on elbow alignment during the release phase."
        ),
        biomechanics=report,
        drills=[],
        model_used="gemini-2.0-flash|prompts@v1.0",
        processing_time_ms=850.0,
    )


# ---------------------------------------------------------------------------
# Concrete mock client
# ---------------------------------------------------------------------------


class MockVLMClient(BaseVLMClient):
    """Test double for BaseVLMClient."""

    def __init__(self, json_response: dict | None = None, text_response: str = "") -> None:
        super().__init__(VLMConfig())
        self._json = json_response or {}
        self._text = text_response

    def complete(self, messages: list[Message]) -> str:
        return self._text

    def complete_json(self, messages: list[Message]) -> dict:
        return self._json


class ErrorVLMClient(BaseVLMClient):
    """Client that always raises VLMError."""

    def complete(self, messages: list[Message]) -> str:
        raise VLMError("API unavailable")

    def complete_json(self, messages: list[Message]) -> dict:
        raise VLMError("API unavailable")


class ParseErrorVLMClient(BaseVLMClient):
    """Client that raises VLMParseError on JSON, returns text on text."""

    def complete(self, messages: list[Message]) -> str:
        return '{"summary": "ok", "primary_correction": "Tuck elbow down", "detailed_analysis": "The elbow flare at release reduces accuracy", "confidence": 0.75, "drills": []}'

    def complete_json(self, messages: list[Message]) -> dict:
        raise VLMParseError("Not JSON")


# ---------------------------------------------------------------------------
# Tests: prompts
# ---------------------------------------------------------------------------


class TestBuildAnalysisPrompt:
    def test_contains_shot_result(self) -> None:
        report = make_report(shot_result=ShotResult.MISSED)
        prompt = build_analysis_prompt(report)
        assert "MISSED" in prompt

    def test_contains_joint_angles(self) -> None:
        report = make_report()
        prompt = build_analysis_prompt(report)
        assert "right_elbow" in prompt
        assert "68.0" in prompt

    def test_contains_issues(self) -> None:
        report = make_report()
        prompt = build_analysis_prompt(report)
        assert "Elbow angle too wide" in prompt

    def test_contains_player_context(self) -> None:
        report = make_report()
        ctx = make_context()
        prompt = build_analysis_prompt(report, ctx)
        assert "player-42" in prompt
        assert "Elbow flare on fatigue" in prompt
        assert "Elbow tuck drill" in prompt

    def test_level_instruction_beginner(self) -> None:
        report = make_report()
        ctx = make_context(level=PlayerLevel.BEGINNER)
        prompt = build_analysis_prompt(report, ctx)
        assert "simple" in prompt.lower() or "fundamental" in prompt.lower()

    def test_level_instruction_elite(self) -> None:
        report = make_report()
        ctx = make_context(level=PlayerLevel.ELITE)
        prompt = build_analysis_prompt(report, ctx)
        assert "nba" in prompt.lower() or "kinetic" in prompt.lower()

    def test_no_context_runs_cleanly(self) -> None:
        report = make_report()
        prompt = build_analysis_prompt(report, context=None)
        assert "SHOT BIOMECHANICS" in prompt

    def test_alignment_score_formatted(self) -> None:
        report = make_report(alignment_score=0.71)
        prompt = build_analysis_prompt(report)
        assert "71%" in prompt

    def test_shot_number_in_prompt(self) -> None:
        report = make_report()
        prompt = build_analysis_prompt(report, shot_number=5)
        assert "shot #5" in prompt.lower()

    def test_system_prompt_non_empty(self) -> None:
        assert len(BASKETBALL_COACH_SYSTEM_PROMPT) > 100
        assert "biomechanics" in BASKETBALL_COACH_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Tests: BasketballVLMAnalyzer
# ---------------------------------------------------------------------------


class TestBasketballVLMAnalyzer:
    def _good_raw(self) -> dict:
        return {
            "summary": "Your elbow is too wide at 68 degrees.",
            "primary_correction": "Tuck your elbow under the ball at release.",
            "detailed_analysis": "The right elbow angle at 68° exceeds optimal 30–50°. Alignment 71%.",
            "confidence": 0.82,
            "drills": [
                {
                    "name": "Wall Drill",
                    "description": "Stand against wall, tuck elbow.",
                    "duration_minutes": 10,
                    "focus": "Elbow alignment",
                    "difficulty": "intermediate",
                }
            ],
        }

    def test_analyze_returns_coaching_feedback(self) -> None:
        report = make_report()
        client = MockVLMClient(json_response=self._good_raw())
        analyzer = BasketballVLMAnalyzer(client)
        feedback = analyzer.analyze(report)
        assert isinstance(feedback, CoachingFeedback)
        assert feedback.shot_result == ShotResult.MISSED
        assert feedback.confidence == 0.82
        assert len(feedback.drills) == 1

    def test_analyze_with_context(self) -> None:
        report = make_report()
        ctx = make_context()
        client = MockVLMClient(json_response=self._good_raw())
        analyzer = BasketballVLMAnalyzer(client)
        feedback = analyzer.analyze(report, ctx)
        assert feedback.player_id == "player-42"

    def test_confidence_clamped(self) -> None:
        raw = self._good_raw()
        raw["confidence"] = 99.0  # invalid — should be clamped to 1.0
        client = MockVLMClient(json_response=raw)
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        assert feedback.confidence == 1.0

    def test_drill_parsing(self) -> None:
        client = MockVLMClient(json_response=self._good_raw())
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        drill = feedback.drills[0]
        assert drill.name == "Wall Drill"
        assert drill.difficulty == PlayerLevel.INTERMEDIATE

    def test_max_three_drills(self) -> None:
        raw = self._good_raw()
        raw["drills"] = [raw["drills"][0]] * 5  # 5 drills → should cap at 3
        client = MockVLMClient(json_response=raw)
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        assert len(feedback.drills) <= 3

    def test_model_used_contains_version(self) -> None:
        client = MockVLMClient(json_response=self._good_raw())
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        assert "prompts@" in feedback.model_used

    def test_fallback_on_parse_error(self) -> None:
        """ParseErrorVLMClient fails JSON but succeeds on text — should fallback."""
        client = ParseErrorVLMClient(VLMConfig())
        analyzer = BasketballVLMAnalyzer(client)
        feedback = analyzer.analyze(make_report())
        assert isinstance(feedback, CoachingFeedback)

    def test_error_fallback_on_api_failure(self) -> None:
        """ErrorVLMClient always fails — should return error fallback, not crash."""
        client = ErrorVLMClient(VLMConfig())
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        assert feedback.confidence == 0.0
        assert feedback.model_used == "fallback"

    def test_analyze_session_processes_all_shots(self) -> None:
        reports = [make_report() for _ in range(3)]
        client = MockVLMClient(json_response=self._good_raw())
        analyzer = BasketballVLMAnalyzer(client)
        results = analyzer.analyze_session(reports)
        assert len(results) == 3

    def test_analyze_session_continues_on_error(self) -> None:
        """One shot failing should not abort the rest."""
        reports = [make_report() for _ in range(3)]
        client = ErrorVLMClient(VLMConfig())
        results = BasketballVLMAnalyzer(client).analyze_session(reports)
        assert len(results) == 3
        assert all(r.model_used == "fallback" for r in results)

    def test_missing_drills_key_graceful(self) -> None:
        raw = self._good_raw()
        del raw["drills"]
        client = MockVLMClient(json_response=raw)
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        assert feedback.drills == []

    def test_processing_time_set(self) -> None:
        client = MockVLMClient(json_response=self._good_raw())
        feedback = BasketballVLMAnalyzer(client).analyze(make_report())
        assert feedback.processing_time_ms is not None
        assert feedback.processing_time_ms >= 0


# ---------------------------------------------------------------------------
# Tests: FeedbackEvaluator
# ---------------------------------------------------------------------------


class TestFeedbackEvaluator:
    def test_good_feedback_passes(self) -> None:
        result = EvaluationResult(
            dimensions=[
                DimensionScore("completeness", 1.0, True, "ok"),
                DimensionScore("actionability", 1.0, True, "ok"),
                DimensionScore("specificity", 1.0, True, "ok"),
                DimensionScore("consistency", 1.0, True, "ok"),
                DimensionScore("no_hallucination", 1.0, True, "ok"),
            ]
        )
        assert result.passed
        assert result.overall_score == 1.0

    def test_evaluator_returns_evaluation_result(self) -> None:
        evaluator = FeedbackEvaluator()
        report = make_report()
        feedback = make_good_feedback(report)
        result = evaluator.evaluate(feedback)
        assert isinstance(result, EvaluationResult)
        assert len(result.dimensions) == 5

    def test_completeness_fails_on_short_summary(self) -> None:
        report = make_report()
        feedback = make_good_feedback(report)
        # Set all 3 text fields too short to guarantee completeness failure
        feedback.summary = "Bad shot."
        feedback.primary_correction = "Fix it."
        feedback.detailed_analysis = "Short."
        evaluator = FeedbackEvaluator()
        result = evaluator.evaluate(feedback)
        completeness = next(d for d in result.dimensions if d.name == "completeness")
        assert not completeness.passed

    def test_actionability_fails_without_verbs(self) -> None:
        report = make_report()
        feedback = make_good_feedback(report)
        feedback.primary_correction = "The elbow angle is problematic."
        feedback.summary = "This is an issue that exists in the shot."
        evaluator = FeedbackEvaluator()
        result = evaluator.evaluate(feedback)
        actionability = next(d for d in result.dimensions if d.name == "actionability")
        assert not actionability.passed

    def test_specificity_passes_with_numeric_refs(self) -> None:
        report = make_report()
        feedback = make_good_feedback(report)
        evaluator = FeedbackEvaluator()
        result = evaluator.evaluate(feedback)
        specificity = next(d for d in result.dimensions if d.name == "specificity")
        assert specificity.passed

    def test_hallucination_flag_on_ungrounded_claims(self) -> None:
        report = make_report()
        feedback = make_good_feedback(report)
        feedback.detailed_analysis = (
            "Studies show that the elbow should be tucked. Research indicates "
            "that NBA average is 45 degrees. Tuck your elbow at release."
        )
        evaluator = FeedbackEvaluator()
        result = evaluator.evaluate(feedback)
        hallucination = next(d for d in result.dimensions if d.name == "no_hallucination")
        assert not hallucination.passed

    def test_evaluation_result_summary_non_empty(self) -> None:
        evaluator = FeedbackEvaluator()
        result = evaluator.evaluate(make_good_feedback(make_report()))
        summary = result.summary()
        assert "Overall:" in summary
        assert len(summary) > 20

    def test_overall_score_is_average(self) -> None:
        dims = [
            DimensionScore("a", 1.0, True, ""),
            DimensionScore("b", 0.5, True, ""),
            DimensionScore("c", 0.0, False, ""),
        ]
        result = EvaluationResult(dimensions=dims)
        assert abs(result.overall_score - 0.5) < 0.01

    def test_passed_false_if_any_dimension_fails(self) -> None:
        dims = [
            DimensionScore("a", 1.0, True, ""),
            DimensionScore("b", 0.0, False, ""),
        ]
        result = EvaluationResult(dimensions=dims)
        assert not result.passed


# ---------------------------------------------------------------------------
# Tests: BaseVLMClient contract
# ---------------------------------------------------------------------------


class TestVLMClientContract:
    def test_model_id_returns_config_model(self) -> None:
        client = MockVLMClient()
        assert client.model_id == "gemini-2.0-flash"

    def test_vlm_error_is_exception(self) -> None:
        with pytest.raises(VLMError):
            raise VLMError("test error")

    def test_vlm_parse_error_is_vlm_error(self) -> None:
        with pytest.raises(VLMError):
            raise VLMParseError("parse failure")


# ---------------------------------------------------------------------------
# Tests: VLMConfig retry defaults
# ---------------------------------------------------------------------------


class TestVLMConfigRetry:
    def test_retry_defaults(self) -> None:
        cfg = VLMConfig()
        assert cfg.retry_attempts == 3
        assert cfg.retry_backoff_seconds == 1.0
        assert cfg.retry_max_backoff_seconds == 16.0

    def test_retry_overridable(self) -> None:
        cfg = VLMConfig(retry_attempts=5, retry_backoff_seconds=0.5, retry_max_backoff_seconds=4.0)
        assert cfg.retry_attempts == 5
        assert cfg.retry_backoff_seconds == 0.5
        assert cfg.retry_max_backoff_seconds == 4.0


# ---------------------------------------------------------------------------
# Tests: GeminiFlashClient retry behavior
#
# We don't need a real google-generativeai install — we patch sys.modules so
# the lazy import inside __init__ resolves to a stub. Tests target the
# `_call_with_retry` helper directly: that's where the resilience logic lives,
# and it's the right unit boundary (avoids coupling to SDK internals).
# ---------------------------------------------------------------------------


def _make_gemini_client(monkeypatch: pytest.MonkeyPatch, **cfg_kwargs: object) -> object:
    """Build a GeminiFlashClient with a stubbed google.generativeai SDK."""
    import sys
    from unittest.mock import MagicMock

    fake_genai = MagicMock()
    fake_genai.GenerativeModel.return_value = MagicMock()
    monkeypatch.setitem(sys.modules, "google", MagicMock())
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    # Import here so the patched sys.modules takes effect.
    from src.vlm.gemini_client import GeminiFlashClient  # noqa: PLC0415

    # Tight backoff so tests stay fast — 0.001s instead of 1s.
    base_kwargs: dict[str, object] = {
        "retry_attempts": 2,
        "retry_backoff_seconds": 0.001,
        "retry_max_backoff_seconds": 0.01,
    }
    base_kwargs.update(cfg_kwargs)
    config = VLMConfig(**base_kwargs)  # type: ignore[arg-type]
    return GeminiFlashClient(api_key="test-key", config=config)


class TestGeminiClientRetry:
    def test_retries_on_transient_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.vlm import gemini_client  # noqa: PLC0415

        # Substitute a known retryable type so we don't depend on google-api-core.
        monkeypatch.setattr(gemini_client, "_RETRYABLE_EXCEPTIONS", (RuntimeError,))

        client = _make_gemini_client(monkeypatch, retry_attempts=3)

        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient blip")
            return "ok"

        result = client._call_with_retry(flaky)  # type: ignore[attr-defined]
        assert result == "ok"
        assert attempts["n"] == 3

    def test_raises_vlmerror_after_attempts_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.vlm import gemini_client  # noqa: PLC0415

        monkeypatch.setattr(gemini_client, "_RETRYABLE_EXCEPTIONS", (RuntimeError,))

        client = _make_gemini_client(monkeypatch, retry_attempts=2)

        def always_transient() -> None:
            raise RuntimeError("never recovers")

        with pytest.raises(VLMError, match="3 attempts"):
            client._call_with_retry(always_transient)  # type: ignore[attr-defined]

    def test_non_retryable_propagates_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.vlm import gemini_client  # noqa: PLC0415

        monkeypatch.setattr(gemini_client, "_RETRYABLE_EXCEPTIONS", (RuntimeError,))

        client = _make_gemini_client(monkeypatch)

        attempts = {"n": 0}

        def auth_failure() -> None:
            attempts["n"] += 1
            raise PermissionError("invalid api key")

        with pytest.raises(PermissionError):
            client._call_with_retry(auth_failure)  # type: ignore[attr-defined]
        assert attempts["n"] == 1, "non-retryable errors must not be retried"

    def test_first_call_succeeds_no_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.vlm import gemini_client  # noqa: PLC0415

        monkeypatch.setattr(gemini_client, "_RETRYABLE_EXCEPTIONS", (RuntimeError,))

        client = _make_gemini_client(monkeypatch)

        attempts = {"n": 0}

        def happy() -> int:
            attempts["n"] += 1
            return 42

        assert client._call_with_retry(happy) == 42  # type: ignore[attr-defined]
        assert attempts["n"] == 1

    def test_request_options_carries_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_gemini_client(monkeypatch)
        opts = client._request_options()  # type: ignore[attr-defined]
        assert opts == {"timeout": client.config.timeout_seconds}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests: LiteLLMClient (T3-1, AI Gateway POC)
#
# All tests stub `litellm.completion` via sys.modules so the suite stays
# fully offline. The contract under test is the BaseVLMClient mapping
# (`complete` / `complete_json` -> stable VLMError/VLMParseError), not
# LiteLLM internals.
# ---------------------------------------------------------------------------


def _install_fake_litellm(monkeypatch: pytest.MonkeyPatch) -> object:
    """Patch sys.modules with a stub `litellm` exposing a configurable
    `completion()` mock. Returns the mock so each test can program it."""
    import sys
    from unittest.mock import MagicMock

    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def _ok_response(content: str) -> object:
    """Build a duck-typed object that mimics LiteLLM's ModelResponse path:
    response.choices[0].message.content. Avoids depending on the real type."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestLiteLLMClient:
    def test_from_env_requires_gemini_key_for_default_primary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_litellm(monkeypatch)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("LITELLM_PRIMARY_MODEL", raising=False)

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        with pytest.raises(VLMError, match="GEMINI_API_KEY"):
            LiteLLMClient.from_env()

    def test_from_env_reads_fallback_csv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_litellm(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv(
            "LITELLM_FALLBACK_MODELS",
            "claude-3-5-haiku-latest, openai/gpt-4o-mini",
        )

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient.from_env()
        assert client._fallback_models == [  # type: ignore[attr-defined]
            "claude-3-5-haiku-latest",
            "openai/gpt-4o-mini",
        ]

    def test_from_env_empty_fallback_is_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_litellm(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("LITELLM_FALLBACK_MODELS", raising=False)

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient.from_env()
        assert client._fallback_models == []  # type: ignore[attr-defined]

    def test_complete_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.return_value = _ok_response("hello world")

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        out = client.complete([Message(role="user", content="hi")])
        assert out == "hello world"

        # Verify the call payload — primary model, retries, timeout passed through
        kwargs = fake.completion.call_args.kwargs
        assert kwargs["model"] == "gemini/gemini-2.0-flash"
        assert kwargs["num_retries"] == client.config.retry_attempts
        assert kwargs["timeout"] == client.config.timeout_seconds
        assert "fallbacks" not in kwargs  # none configured

    def test_complete_passes_fallbacks_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.return_value = _ok_response("ok")

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(
            primary_model="gemini/gemini-2.0-flash",
            fallback_models=["claude-3-5-haiku-latest"],
        )
        client.complete([Message(role="user", content="hi")])
        assert fake.completion.call_args.kwargs["fallbacks"] == ["claude-3-5-haiku-latest"]

    def test_complete_wraps_litellm_error_as_vlmerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.side_effect = RuntimeError("upstream went away")

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        with pytest.raises(VLMError, match="litellm_call_failed:RuntimeError"):
            client.complete([Message(role="user", content="hi")])

    def test_complete_empty_messages_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_litellm(monkeypatch)

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        with pytest.raises(VLMError, match="No user messages provided"):
            client.complete([])

    def test_complete_json_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.return_value = _ok_response('{"summary": "ok"}')

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        out = client.complete_json([Message(role="user", content="give json")])
        assert out == {"summary": "ok"}
        # JSON mode must be requested
        assert fake.completion.call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_complete_json_raises_parse_error_on_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.return_value = _ok_response("not-json-at-all")

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        with pytest.raises(VLMParseError, match="non-JSON"):
            client.complete_json([Message(role="user", content="give json")])

    def test_complete_json_rejects_non_dict_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`json.loads` happily returns lists/bools/numbers — the contract is dict."""
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.return_value = _ok_response('["not", "a", "dict"]')

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        with pytest.raises(VLMParseError, match="expected dict"):
            client.complete_json([Message(role="user", content="give json")])

    def test_explicit_config_wins_over_primary_model_arg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gemini finding on PR #43: `config.model` is the single source of truth."""
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.return_value = _ok_response("ok")

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(
            primary_model="gemini/gemini-2.0-flash",  # convenience seed, ignored
            config=VLMConfig(model="claude-3-5-haiku-latest"),
        )
        assert client.config.model == "claude-3-5-haiku-latest"
        assert client.model_id == "claude-3-5-haiku-latest"

        client.complete([Message(role="user", content="hi")])
        assert fake.completion.call_args.kwargs["model"] == "claude-3-5-haiku-latest"

    def test_log_uses_actual_response_model_for_failover_visibility(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When LiteLLM falls back, the log must report the model that responded."""
        import logging  # noqa: PLC0415

        fake = _install_fake_litellm(monkeypatch)
        resp = _ok_response("ok")
        # Failover scenario: primary Gemini, fallback actually replied with Claude.
        resp.model = "claude-3-5-haiku-latest"
        fake.completion.return_value = resp

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(
            primary_model="gemini/gemini-2.0-flash",
            fallback_models=["claude-3-5-haiku-latest"],
        )
        with caplog.at_level(logging.DEBUG, logger="src.vlm.litellm_client"):
            client.complete([Message(role="user", content="hi")])

        debug_logs = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
        assert any("claude-3-5-haiku-latest" in msg for msg in debug_logs), (
            f"expected fallback model in debug logs, got: {debug_logs}"
        )

    def test_response_never_leaks_internal_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LiteLLM errors must surface as stable codes, no exception text."""
        fake = _install_fake_litellm(monkeypatch)
        fake.completion.side_effect = PermissionError("API key denied; secret=abc123")

        from src.vlm.litellm_client import LiteLLMClient  # noqa: PLC0415

        client = LiteLLMClient(primary_model="gemini/gemini-2.0-flash")
        try:
            client.complete([Message(role="user", content="hi")])
        except VLMError as err:
            msg = str(err)
            assert "abc123" not in msg, "secret leaked into VLMError message"
            assert "API key denied" not in msg, "raw exception text leaked"
            assert "litellm_call_failed:PermissionError" in msg, "expected stable code"
        else:
            pytest.fail("VLMError not raised")
