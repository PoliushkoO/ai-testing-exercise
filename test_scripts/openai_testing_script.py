import os
import pandas as pd
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directories
INPUT_FILE = "test_input/input.csv"
OUTPUT_FILE = "test_results/output.csv"
PROMPT_FILE = "test_input/prompt.txt"

# Ensure output directory exists
os.makedirs("test_results", exist_ok=True)

def load_prompt():
    """Load system prompt from a file."""
    with open(PROMPT_FILE, "r", encoding="utf-8") as file:
        return file.read()

def extract_attributes(chat_history, prompt):
    """Send request to OpenAI API and extract attributes."""
    client = openai.OpenAI()  # Create an OpenAI client instance

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chat_history}
        ]
    )

    return response.choices[0].message.content.strip()

def evaluate_results(expected, actual):
    """Compare expected vs actual attributes and return match status."""
    return "TRUE" if expected == actual else "FALSE"

def main():
    """Main script to run the data-driven test."""
    prompt = load_prompt()
    results = []

    # Read input CSV
    df = pd.read_csv(INPUT_FILE)

    # Check for correct column names
    if "chat_history" not in df.columns or "expected_attributes" not in df.columns:
        raise KeyError("CSV file must contain 'chat_history' and 'expected_attributes' columns")

    for _, row in df.iterrows():
        chat_history = row["chat_history"]
        expected_attributes = row["expected_attributes"]

        actual_attributes = extract_attributes(chat_history, prompt)
        match_status = evaluate_results(expected_attributes, actual_attributes)

        results.append([chat_history, expected_attributes, actual_attributes, match_status])

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["chat_history", "expected_attributes", "actual_attributes", "status"])
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Test results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
