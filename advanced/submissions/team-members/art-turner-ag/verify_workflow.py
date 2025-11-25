from workflow import ResearchWorkflow
from unittest.mock import MagicMock
import sys

# Mocking dependencies to avoid actual API calls during verification if keys are missing or to save cost
# However, for true verification, we might want to run with actual keys if available.
# Given the instructions, I should probably try to run it. But if I don't have keys, it will fail.
# The user said they have keys in the .env file. So I will try to run it.

if __name__ == "__main__":
    try:
        workflow = ResearchWorkflow()
        # Use a simple topic to avoid long processing
        topic = "The benefits of drinking water"
        print(f"Running workflow for topic: {topic}")
        final_report, findings = workflow.run(topic)
        
        print("\n--- Final Report ---")
        print(final_report[:200] + "...") # Print first 200 chars
        print("\n--- Findings Keys ---")
        print(list(findings.keys()))
        print("\nVerification Successful!")
    except Exception as e:
        print(f"Verification Failed: {e}")
