import pandas as pd
import re
import os
from collections import defaultdict

# --- CONFIGURATION ---

# This is the crucial list. We've expanded it to include all common
# headers found in clinical notes, including those from your sample.
TARGET_HEADERS = [
    'SUBJECTIVE',
    'OBJECTIVE',
    'CHIEF COMPLAINT',
    'HISTORY OF PRESENT ILLNESS',
    'PAST MEDICAL HISTORY',
    'FAMILY HISTORY',
    'SOCIAL HISTORY',
    'ALLERGIES',
    'MEDICATIONS',
    'REVIEW OF SYSTEMS',
    'PHYSICAL EXAMINATION',
    'PHYSICAL EXAM',
    'EXAM',
    'LABORATORY DATA',
    'LABS',
    'IMAGING',
    'PROCEDURE',
    'PROCEDURES',
    'ASSESSMENT',
    'IMPRESSION',
    'DIAGNOSIS',
    'PLAN',
    'RECOMMENDATIONS'
]

INPUT_FILE = 'mtsamples.csv'
OUTPUT_PREFIX = 'kb_'

# This regex finds any of the headers, followed by a colon.
# We build it dynamically from the list above.
# It's case-insensitive.
HEADER_REGEX = re.compile(
    # Use a non-word-boundary at the start (\B) or start-of-line (^)
    # to catch headers that might be part of another word if we're not careful,
    # but more importantly, to just be flexible.
    # The main part is the (HEADER_NAME)\s*:
    r'(^|\n)\s*(' + '|'.join(re.escape(h) for h in TARGET_HEADERS) + r')\s*:\s*',
    re.IGNORECASE
)

# --- SCRIPT LOGIC ---

def sanitize_filename(name: str) -> str:
    """Removes special characters to create a valid filename."""
    # Replace slashes, spaces, and other problematic chars with an underscore
    name = re.sub(r'[\s/\\:]+', '_', name)
    # Remove any other non-alphanumeric characters (except underscore)
    name = re.sub(r'[^\w_]', '', name)
    return name

def extract_sections(text: str) -> list[tuple[str, str]]:
    """
    Finds all target sections in a note and extracts their content.
    
    This logic works by:
    1. Finding all headers.
    2. For each header, grabbing the text from its start
       until the *next* header begins.
    """
    sections = []
    
    # Pre-process text to standardize newlines
    text = re.sub(r'(\r\n|\r)', '\n', text)
    
    # Find all matches for our headers
    matches = list(HEADER_REGEX.finditer(text))
    
    if not matches:
        return []

    for i, match in enumerate(matches):
        # The header name (e.g., "PLAN") is the 2nd captured group
        header_name = match.group(2).upper().strip()
        
        # The content starts right after the full match (e.g., after "PLAN: ")
        content_start = match.end()
        
        # The content ends where the *next* section's match begins
        # or at the end of the file if this is the last section.
        content_end = len(text)
        if i + 1 < len(matches):
            # The next match's start is where this content ends
            content_end = matches[i+1].start()
            
        # Extract the content
        content = text[content_start:content_end].strip()
        
        # Clean up common artifacts, like "1. ..." or "2. ..."
        # and remove excessive newlines.
        content = re.sub(r'^\s*(\d+\.|-)\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n\s*\n', '\n', content) # Consolidate blank lines
        
        if content:
            sections.append((header_name, content))
            
    return sections

def main():
    """
    Reads the CSV, processes all notes, and writes
    specialty-specific .md files.
    """
    print(f"Attempting to read '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILE}' not found.")
        print("Please make sure it's in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure required columns exist
    if 'transcription' not in df.columns or 'medical_specialty' not in df.columns:
        print("Error: CSV must have 'transcription' and 'medical_specialty' columns.")
        return

    # Drop rows where our key data is missing
    df = df.dropna(subset=['transcription', 'medical_specialty'])
    
    if len(df) == 0:
        print("No valid data found in CSV.")
        return
        
    print(f"Found {len(df)} samples with transcriptions and specialties.")

    # Group all transcriptions by their specialty
    specialty_notes = defaultdict(list)
    for _, row in df.iterrows():
        specialty = row['medical_specialty'].strip()
        transcription = str(row['transcription'])
        specialty_notes[specialty].append(transcription)

    num_specialties = len(specialty_notes)
    print(f"Processing {num_specialties} medical specialties...")
    
    created_files = []
    
    # For each specialty, aggregate all its sections
    for specialty, notes in specialty_notes.items():
        all_sections_content = []
        
        for note_text in notes:
            sections = extract_sections(note_text)
            for header, content in sections:
                # We format it nicely for the markdown file
                all_sections_content.append(f"### Example: {header}\n{content}\n")
        
        if not all_sections_content:
            # Skip specialties where we found no matching sections
            continue
            
        # Sanitize the specialty name for a valid filename
        safe_filename = sanitize_filename(specialty)
        filename = f"{OUTPUT_PREFIX}{safe_filename}.md"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Knowledge Base: {specialty}\n\n")
                f.write(f"This document contains aggregated, anonymous sections for the '{specialty}' specialty, extracted from sample notes.\n\n")
                f.write("---\n\n")
                f.write("\n\n---\n\n".join(all_sections_content))
                
            created_files.append(filename)
        except Exception as e:
            print(f"Error writing file for {specialty}: {e}")

    print("\n--- Knowledge Base Build Complete ---")
    if created_files:
        print(f"Successfully created {len(created_files)} .md files:")
        for fname in created_files:
            print(f" - {fname}")
    else:
        print("No .md files were created. This may be because:")
        print("1. The CSV was empty or had no matching transcriptions.")
        print("2. The headers in 'TARGET_HEADERS' do not match *any* headers in your CSV.")
        print("   Please check your CSV and update the TARGET_HEADERS list if needed.")

if __name__ == "__main__":
    main()