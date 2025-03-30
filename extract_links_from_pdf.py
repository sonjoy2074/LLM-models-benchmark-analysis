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

# Initialize the Groq Agent
agent = Agent(
    model=Groq(id='llama-3.3-70b-versatile'),
    show_tool_calls=True,
    markdown=False,
    instructions=["Extract only LinkedIn and GitHub profile links from the provided text. Return only the links, nothing else."],
)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def extract_links_with_llama(pdf_text):
    """Use Groq's LLaMA model to extract LinkedIn and GitHub profile links, and clean the response."""
    prompt = f"Extract LinkedIn and GitHub profile links from this text: {pdf_text}. Return only the links, one per line, nothing else."

    f = io.StringIO()
    with redirect_stdout(f):
        agent.print_response(prompt)
    response = f.getvalue().strip()

    links = set(re.findall(r'https?://(?:www\.)?(linkedin\.com/in/[^\s]+|github\.com/[^\s]+)', response))

    # Return cleaned links as a string
    return "\n".join(["https://" + link for link in links]) if links else "No valid links found."


def save_links_to_file(links, output_file):
    """Save extracted links to a text file using UTF-8 encoding."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(links + "\n")

if __name__ == "__main__":
    pdf_path = "Resume.pdf" 
    output_file = "saved_info.txt"
    
    pdf_text = extract_text_from_pdf(pdf_path)
    extracted_links = extract_links_with_llama(pdf_text)
    
    if extracted_links:
        save_links_to_file(extracted_links, output_file)
        print(f"Extracted links saved to {output_file}")
    else:
        print("No links were extracted.")
