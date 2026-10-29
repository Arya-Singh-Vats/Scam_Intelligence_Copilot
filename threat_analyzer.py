
import re


def analyze_threat_indicators(message):
    """Identify suspicious patterns in an SMS or email."""

    text = message.lower()
    indicators = []
    urls = re.findall(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
        message,
        flags=re.IGNORECASE,
    )

    # 1. Suspicious links
    if urls:
        indicators.append({
            "category": "Suspicious link",
            "severity": "High",
            "details": (
                f"Found {len(urls)} URL(s). Verify the domain "
                "before opening any link."
            ),
            "evidence": urls[:5],
        })

    # 2. Requests for sensitive information
    credential_patterns = [
        r"\botp\b",
        r"\bone[- ]time password\b",
        r"\bpassword\b",
        r"\bpin\b",
        r"\bcvv\b",
        r"\baccount number\b",
        r"\blogin credentials\b",
    ]

    if any(re.search(pattern, text) for pattern in credential_patterns):
        indicators.append({
            "category": "Sensitive information request",
            "severity": "High",
            "details": (
                "The message mentions credentials or sensitive "
                "financial information. Never disclose these in "
                "response to an unexpected message."
            ),
            "evidence": [
                pattern.strip(r"\b")
                for pattern in credential_patterns
                if re.search(pattern, text)
            ],
        })

    # 3. Urgency and threats
    urgency_terms = [
        "urgent",
        "immediately",
        "act now",
        "account suspended",
        "account blocked",
        "will be blocked",
        "expires today",
        "within 24 hours",
        "verify now",
    ]

    found_urgency = [term for term in urgency_terms if term in text]

    if found_urgency:
        indicators.append({
            "category": "Urgency or threat",
            "severity": "Medium",
            "details": (
                "The message pressures the recipient to act quickly, "
                "which can be a social-engineering tactic."
            ),
            "evidence": found_urgency,
        })

    # 4. Prize and reward lures
    reward_terms = [
        "you have won",
        "winner",
        "claim your prize",
        "free gift",
        "cash prize",
        "lottery",
        "reward points",
    ]

    found_rewards = [term for term in reward_terms if term in text]

    if found_rewards:
        indicators.append({
            "category": "Prize or reward lure",
            "severity": "Medium",
            "details": (
                "Unexpected prizes or rewards may be used to lure "
                "people into clicking links or sharing information."
            ),
            "evidence": found_rewards,
        })

    # 5. Payment pressure
    payment_terms = [
        "pay now",
        "send money",
        "transfer money",
        "processing fee",
        "pay immediately",
        "upi payment",
    ]

    found_payment = [term for term in payment_terms if term in text]

    if found_payment:
        indicators.append({
            "category": "Payment pressure",
            "severity": "Medium",
            "details": (
                "The message requests or pressures the recipient "
                "to make a payment."
            ),
            "evidence": found_payment,
        })

    return {
        "indicators": indicators,
        "indicator_count": len(indicators),
        "urls_found": urls,
        "status": (
            "Potential warning signs detected"
            if indicators
            else "No configured warning patterns detected"
        ),
    }
