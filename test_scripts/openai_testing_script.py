import openai
import pandas as pd
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

def extract_attributes(chat_history, prompt):
    """Sends a request to OpenAI's ChatCompletion API to extract attributes."""
    client = openai.OpenAI()  # Create an OpenAI client instance

    response = client.chat.completions.create(
        model="gpt-4",  # Ensure you specify the correct model
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chat_history}
        ]
    )

    # Extract response content
    extracted_data = response.choices[0].message.content
    return extracted_data

def main():
    INPUT_FILE = "test_input/input.csv"
    OUTPUT_FILE = "test_results/output.csv"
    PROMPT_FILE = "test_input/prompt.txt"

    # Read input CSV
    df = pd.read_csv(INPUT_FILE)

    # Check for correct column names
    if "chat_history" not in df.columns or "expected_attributes" not in df.columns:
        raise KeyError("CSV file must contain 'chat_history' and 'expected_attributes' columns")

    # Read system prompt from prompt.txt
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt = f.read()

    results = []

    for _, row in df.iterrows():
        chat_history = row["chat_history"]
        expected_attributes = row["expected_attributes"]

        # Call OpenAI API to extract attributes
        actual_attributes = extract_attributes(chat_history, prompt)

        # Compare expected vs actual
        match_status = "TRUE" if actual_attributes == expected_attributes else "FALSE"

        # Store result
        results.append([chat_history, expected_attributes, actual_attributes, match_status])

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["chat_history", "expected_attributes", "actual_attributes", "status"])
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Test results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
