import os
import pandas as pd
import openai
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directories
INPUT_FILE = "test_input/input.csv"
PROMPT_FILE = "test_input/prompt.txt"
OUTPUT_DIR = "test_results"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_prompt():
    """Load system prompt from a file."""
    with open(PROMPT_FILE, "r", encoding="utf-8") as file:
        return file.read()

def extract_attributes(chat_history, prompt):
    """Send request to OpenAI API and extract attributes."""
    client = openai.OpenAI()  # Create OpenAI client instance

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chat_history}
        ]
    )

    extracted_text = response.choices[0].message.content.strip()

    # Convert JSON string to dictionary
    try:
        extracted_data = json.loads(extracted_text)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON response")
        return "Invalid format"

    # Extract individual attributes safely
    email = extracted_data.get("email", "N/A")

    # Ensure phone is always a list, even if only one number is provided
    phone_numbers = extracted_data.get("phone", [])
    if isinstance(phone_numbers, str):
        phone_numbers = [phone_numbers]

    # Format phone numbers correctly
    formatted_phone = ", ".join(phone_numbers) if phone_numbers else "N/A"

    move_date = extracted_data.get("move_date", "N/A")

    # Enforce "only one valid move date" rule
    if isinstance(move_date, list):  # If model returns a list of dates
        if len(set(move_date)) > 1:  # Multiple distinct dates detected
            move_date = "N/A"
        else:
            move_date = move_date[0]  # Keep the single valid date

    elif isinstance(move_date, str) and "," in move_date:  # If model returns comma-separated dates
        dates = set(move_date.split(","))
        if len(dates) > 1:
            move_date = "N/A"
        else:
            move_date = list(dates)[0].strip()

    # Format as required: Email: ... | Phone: ... | Move date: ...
    formatted_output = f"Email: {email} | Phone: {formatted_phone} | Move date: {move_date}"

    return formatted_output

def evaluate_results(expected, actual):
    """Compare expected vs actual attributes and return match status, ignoring case."""
    return "TRUE" if expected.lower() == actual.lower() else "FALSE"

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

    # Generate unique output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"output_{timestamp}.csv")

    # Save results to CSV
    results_df = pd.DataFrame(results, columns=["chat_history", "expected_attributes", "actual_attributes", "status"])
    results_df.to_csv(output_file, index=False)

    print(f"Test results saved to {output_file}")

if __name__ == "__main__":
    main()
