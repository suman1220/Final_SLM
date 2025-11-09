"""Lightweight security checks to complement the medical guardrails."""

from __future__ import annotations

import re
from typing import Iterable, Optional

from guardrails_medical import ValidationResult, MedicalResponsePolicy


class PromptInjectionDetector:
    """Detect common prompt-injection attempts in user questions."""

    _OVERRIDE_PATTERNS = (
        re.compile(r"\b(ignore|bypass|forget)\b.*\b(instruction|policy|rule|guardrail)s?\b", re.IGNORECASE),
        re.compile(r"\bdisregard\b.*\bprevious\b.*\bmessage\b", re.IGNORECASE),
        re.compile(r"\b(system|developer)\s+prompt\b", re.IGNORECASE),
        re.compile(r"\b(jailbreak|do anything now|dan mode)\b", re.IGNORECASE),
        re.compile(r"\bact\s+as\b.*\b(system|root|administrator)\b", re.IGNORECASE),
        re.compile(r"###\s*(system|assistant|developer)", re.IGNORECASE),
        re.compile(r"\bprevious\b.*\btester\b.*\btold you\b", re.IGNORECASE),
        re.compile(r"\brole[-\s]?play\b.*\bdoctor\b.*\bdiscuss anything\b", re.IGNORECASE),
        re.compile(r"\b(list|share|enumerate)\b.*\b(forbidden|restricted|banned)\b.*\btopics\b", re.IGNORECASE),
    )

    _OVERRIDE_KEYWORDS = {
        "ignore previous",
        "override instructions",
        "break character",
        "forget your training",
        "unfiltered",
        "no moral",
        "developer mode",
        "instruction override",
        "previous tester told you",
        "roleplay as a doctor",
        "list forbidden topics",
        "what are your forbidden topics",
        "discuss anything",
    }

    _SECRET_EXTRACTION_PATTERN = re.compile(
        r"\b(api key|api keys|passwords?|tokens?|secrets?|credentials?|private key)\b.*\b(machine|system|stored|device)\b",
        re.IGNORECASE,
    )

    def detect(
        self,
        question: str,
        *,
        recent_questions: Optional[Iterable[str]] = None,
    ) -> ValidationResult:
        if not question:
            return ValidationResult(passed=True)

        normalized = question.lower()

        for pattern in self._OVERRIDE_PATTERNS:
            if pattern.search(normalized):
                return ValidationResult(
                    passed=False,
                    message="Prompt injection signal detected (override instructions).",
                )

        if any(keyword in normalized for keyword in self._OVERRIDE_KEYWORDS):
            return ValidationResult(
                passed=False,
                message="Prompt injection signal detected (override keywords).",
            )

        if self._SECRET_EXTRACTION_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="Prompt injection signal detected (attempt to extract secrets).",
            )

        if recent_questions:
            recent_normalized = [q.lower() for q in recent_questions if q]
            injection_attempts = sum(
                1 for q in recent_normalized
                if any(pattern.search(q) for pattern in self._OVERRIDE_PATTERNS)
            )
            if injection_attempts >= 2:
                return ValidationResult(
                    passed=False,
                    message="Repeated prompt injection attempts detected in conversation history.",
                )

        return ValidationResult(passed=True)


class HighRiskMedicalRequestDetector:
    """Detect direct requests for dosing, prescriptions, or medication adjustments."""

    _INSULIN_REQUEST_PATTERN = re.compile(
        r"\b(insulin|dose|dosing|units?|bolus|basal|long[-\s]?acting|short[-\s]?acting|rapid[-\s]?acting)\b.*\b(should I|how much|how many|double|increase|decrease|take|give|administer)\b",
        re.IGNORECASE,
    )
    _MED_ADJUSTMENT_PATTERN = re.compile(
        r"\b(should I|can I|do I|is it ok|is it okay)\b.*\b(double|adjust|change|increase|decrease)\b.*\b(insulin|medication|dose)\b",
        re.IGNORECASE,
    )
    _SPECIFIC_UNITS_PATTERN = re.compile(r"\b(\d+)\s*(u|units?)\b", re.IGNORECASE)

    def detect(self, question: str) -> ValidationResult:
        if not question:
            return ValidationResult(passed=True)

        normalized = question.lower()

        if self._INSULIN_REQUEST_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="High-risk medication dosing request detected; refusing.",
            )

        if self._MED_ADJUSTMENT_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="Medication adjustment request detected; refusing.",
            )

        if self._SPECIFIC_UNITS_PATTERN.search(normalized) and "insulin" in normalized:
            return ValidationResult(
                passed=False,
                message="Specific insulin unit request detected; refusing.",
            )

        return ValidationResult(passed=True)


class SafetyContentFilter:
    """Detect offensive or unsafe model outputs."""

    _PROFANITY_PATTERN = re.compile(
        r"\b(fuck|shit|bastard|asshole|bitch|cunt|motherfucker|slut|whore|dick|piss|bollocks)\b",
        re.IGNORECASE,
    )
    _HATE_SPEECH_PATTERN = re.compile(
        r"\b(?:kill|harm|attack)\b.*\b(?:jews|muslims|christians|blacks|asians|gays|lesbians|immigrants)\b",
        re.IGNORECASE,
    )
    _SELF_HARM_PATTERN = re.compile(
        r"\b(?:suicide|kill myself|die by suicide|self harm|self-harm|cut myself)\b",
        re.IGNORECASE,
    )
    _DOSING_DIRECTIVE_PATTERNS = (
        re.compile(
            r"\b(double|take|administer|give|inject)\b[^.]{0,80}\b(insulin|insulin\s+dose|insulin\s+units?|dose|dosage|units?|bolus|basal|shot)s?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(increase|decrease|adjust)\b[^.]{0,80}\b(insulin\s+(?:dose|dosage|units?|bolus|basal|shot)|(?:dose|dosage|units?|bolus|basal|shot)\s+of\s+insulin)\b",
            re.IGNORECASE,
        ),
    )
    _DOSING_UNIT_PATTERN = re.compile(
        r"\b\d+\s*(?:u|units?)\b[^.]{0,40}\binsulin\b",
        re.IGNORECASE,
    )

    _REFUSAL_WHITELIST = {MedicalResponsePolicy.REFUSAL_LINE.lower()}

    def validate(self, response: str) -> ValidationResult:
        if not response:
            return ValidationResult(passed=True)

        normalized = response.lower()
        if normalized in self._REFUSAL_WHITELIST:
            return ValidationResult(passed=True)

        if self._PROFANITY_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="Detected profanity in generated response.",
            )
        if self._HATE_SPEECH_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="Detected hate/offensive content in generated response.",
            )
        if self._SELF_HARM_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="Detected self-harm content in generated response.",
            )
        if any(pattern.search(normalized) for pattern in self._DOSING_DIRECTIVE_PATTERNS):
            return ValidationResult(
                passed=False,
                message="Detected insulin dosing directive in generated response.",
            )
        if self._DOSING_UNIT_PATTERN.search(normalized):
            return ValidationResult(
                passed=False,
                message="Detected explicit insulin unit instruction in generated response.",
            )

        return ValidationResult(passed=True)


prompt_injection_detector = PromptInjectionDetector()
safety_filter = SafetyContentFilter()
high_risk_detector = HighRiskMedicalRequestDetector()
_CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way. I can't give crisis counseling, but please contact emergency services or a suicide hotline right away. "
    "In the United States, call or text 988. If you're elsewhere, reach out to local emergency numbers or mental health professionals immediately."
)


class SelfHarmCrisisDetector:
    """Detect user questions expressing intent to self-harm."""

    _CRISIS_PATTERNS = (
        re.compile(r"\b(i don'?t want to live|i don'?t want to be here)\b", re.IGNORECASE),
        re.compile(r"\b(end it|end my life|kill myself|suicide|take all my meds|take all my insulin)\b", re.IGNORECASE),
        re.compile(r"\b(stop (?:all )?my meds\b.*\b(?:to die|because i want to die))", re.IGNORECASE),
        re.compile(r"\b(?:self[-\s]?harm|hurt myself)\b", re.IGNORECASE),
    )

    def detect(self, question: str) -> ValidationResult:
        if not question:
            return ValidationResult(passed=True)

        normalized = question.lower()
        for pattern in self._CRISIS_PATTERNS:
            if pattern.search(normalized):
                return ValidationResult(
                    passed=False,
                    message="Self-harm intent detected; route to crisis response.",
                )

        return ValidationResult(passed=True)


def detect_prompt_injection(
    question: str,
    *,
    recent_questions: Optional[Iterable[str]] = None,
) -> ValidationResult:
    return prompt_injection_detector.detect(
        question,
        recent_questions=recent_questions,
    )


def detect_high_risk_medical_request(question: str) -> ValidationResult:
    return high_risk_detector.detect(question)


crisis_detector = SelfHarmCrisisDetector()


def detect_self_harm_crisis(question: str) -> ValidationResult:
    return crisis_detector.detect(question)


def crisis_response() -> str:
    return _CRISIS_RESPONSE


def enforce_safety_filters(response: str) -> ValidationResult:
    return safety_filter.validate(response)


__all__ = [
    "detect_prompt_injection",
    "detect_high_risk_medical_request",
    "detect_self_harm_crisis",
    "enforce_safety_filters",
    "crisis_response",
    "PromptInjectionDetector",
    "SafetyContentFilter",
    "HighRiskMedicalRequestDetector",
    "SelfHarmCrisisDetector",
]

