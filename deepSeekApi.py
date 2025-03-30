import re
import fitz  # PyMuPDF for PDF processing
import os
import io
from contextlib import redirect_stdout
from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

agent = Agent(
    model=Groq(id='deepseek-r1-distill-qwen-32b'),
    show_tool_calls=True,
    markdown=False,
    instructions=["Extract coding-related questions from the provided text and solve them. Return only the question followed by its solution."],
)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def solve_coding_problems(pdf_text):
    """Use Groq's LLaMA model to extract coding problems and provide solutions."""
    prompt = f"Extract coding-related problems from this text and solve them: {pdf_text}. Return each question followed by its solution."

    f = io.StringIO()
    with redirect_stdout(f):
        agent.print_response(prompt)
    response = f.getvalue().strip()

    # Remove unnecessary formatting and clean response
    clean_response = re.sub(r'\x1b\[[0-9;]*[mK]', '', response)  # Remove ANSI escape codes
    clean_response = re.sub(r'[┃┏┓┗━┛]+', '', clean_response)     # Remove decorative box-drawing characters
    return clean_response if clean_response else "No coding-related problems found or solved."

def save_output_to_file(output, output_file):
    """Save extracted problems and solutions to a text file using UTF-8 encoding."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output + "\n")

if __name__ == "__main__":
    pdf_path = "Questions.pdf"  # Change to your PDF file containing coding problems
    output_file = "solutions.txt"

    pdf_text = extract_text_from_pdf(pdf_path)
    solved_problems = solve_coding_problems(pdf_text)

    if solved_problems:
        save_output_to_file(solved_problems, output_file)
        print(f"Coding problems and their solutions saved to {output_file}")
    else:
        print("No coding problems were found or solved.")