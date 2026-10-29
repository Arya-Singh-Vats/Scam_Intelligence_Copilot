
from threat_analyzer import analyze_threat_indicators

sample_message = (
    "URGENT! Your bank account will be blocked. "
    "Verify now at http://example.com and enter your OTP "
    "and password immediately."
)

result = analyze_threat_indicators(sample_message)

print("Status:", result["status"])
print("Indicators found:", result["indicator_count"])

for item in result["indicators"]:
    print(f"\nCategory: {item['category']}")
    print(f"Severity: {item['severity']}")
    print(f"Details: {item['details']}")
    print(f"Evidence: {item['evidence']}")
