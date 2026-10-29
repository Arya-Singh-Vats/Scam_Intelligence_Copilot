
from llm_explainer import explain_message

try:
    result = explain_message(
        message=(
            "Congratulations! You won a prize. "
            "Click this link and provide your bank details to claim it."
        ),
        prediction="spam",
        spam_probability=0.95,
        risk="High",
    )

    print("\n--- AI SAFETY EXPLANATION ---\n")
    print(result)

except Exception as error:
    print(f"Test failed: {error}")
