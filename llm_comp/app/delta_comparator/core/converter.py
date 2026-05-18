# json_processor.py
import json
import re
import os
from collections import defaultdict
import zipfile
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from app.delta_comparator.utils.logger import log as logging
class ConverterProcessor:
    def __init__(self):
        self.header_patterns = [
            r"^SDV-6001\s+–\s+Diagnostic Infrastructure Specification",
            r"^Page:\s*\d+\s+of\s+\d+",
            r"^ECCN:\s*.*",
            r"^Release Date:\s*.*",
            r"^Cadence:\s*.*",
            r"^Group:\s*.*",
            r"^GM CONFIDENTIAL$"
        ]
        self.header_regex = re.compile("|".join(self.header_patterns), re.IGNORECASE)

    def json_To_csv(self, filepath: str, col_name: str) -> list:
        """Load and clean a JSON file, splitting lines and filtering headers."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_entries = defaultdict(list)
        for entry in data:
            page_entries[entry['page_idx']].append(entry)

        rows = []
        for page in sorted(page_entries):
            for entry in page_entries[page]:
                text = entry.get("text", "").strip()
                if not text or self.header_regex.match(text):
                    continue

                if "\n" in text:
                    for line in text.splitlines():
                        line = line.strip()
                        if line and not self.header_regex.match(line):
                            rows.append({col_name: line})
                else:
                    rows.append({col_name: text})
        return rows
    
    # This converts only the simple text, ignoring Tables and Images
    def json_To_csv_mercedes(self, filepath: str, col_name: str) -> list:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_entries = defaultdict(list)
        for entry in data:
            page_entries[entry['page_idx']].append(entry)

        rows = []
        for page in sorted(page_entries):
            for entry in page_entries[page]:
                text = entry.get("text", "").strip()
                entry_type = entry.get("type", "").lower()

                if not text or self.header_regex.match(text):
                    continue

                # If it's an image or table type, treat entire block as one row
                if entry_type in ["image", "table"]:
                    rows.append({col_name: text})
                    continue

                # Otherwise, split by lines and add non-header ones
                if "\n" in text:
                    for line in text.splitlines():
                        line = line.strip()
                        if line and not self.header_regex.match(line):
                            rows.append({col_name: line})
                else:
                    rows.append({col_name: text})

        return rows
    
    # This converts only the simple text, ignoring Tables and Images
    def json_To_csv_mercedes_no_tables_images(self, filepath: str, col_name: str) -> list:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_entries = defaultdict(list)
        for entry in data:
            page_entries[entry['page_idx']].append(entry)

        rows = []
        for page in sorted(page_entries):
            for entry in page_entries[page]:
                text = entry.get("text", "").strip()
                entry_type = entry.get("type", "").lower()

                # Skip if empty, header line, or unwanted type
                if (
                    not text
                    or self.header_regex.match(text)
                    or entry_type in ["image", "table"]
                ):
                    continue

                # If text contains newlines, split into individual lines
                if "\n" in text:
                    for line in text.splitlines():
                        line = line.strip()
                        if line and not self.header_regex.match(line):
                            rows.append({col_name: line})
                else:
                    rows.append({col_name: text})

        return rows
    
    def extract_reqifz_to_csv_general(self, reqifz_path, column_name, output_csv_path, skip_section_headers=True):
        """Extracts English requirements from a .reqifz file and writes them to a CSV."""

        ns = {
            'reqif': 'http://www.omg.org/spec/ReqIF/20110401/reqif.xsd',
            'xhtml': 'http://www.w3.org/1999/xhtml'
        }
        section_pattern = re.compile(r'^\d+(\.\d+)*$')

        def get_full_text(element):
            """Recursively extract plain text from an XHTML element."""
            text = element.text or ""
            for child in element:
                if child.tag.endswith('a'):
                    href = child.attrib.get('href', '')
                    inner_text = child.text or ""
                    text += f"[{inner_text}]({href})"
                else:
                    text += get_full_text(child)
                if child.tail:
                    text += child.tail
            return text.strip()

        # Prepare temp directory
        temp_dir = "temp_extracted_reqif"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Unzip the .reqifz file
        with zipfile.ZipFile(reqifz_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find .reqif file
        reqif_file = next((f for f in os.listdir(temp_dir) if f.endswith(".reqif")), None)
        if not reqif_file:
            raise FileNotFoundError("No .reqif file found in the .reqifz archive.")
        reqif_path = os.path.join(temp_dir, reqif_file)

        # Parse XML
        tree = ET.parse(reqif_path)
        root = tree.getroot()

        entries = []
        for xhtml_val in root.findall(".//reqif:ATTRIBUTE-VALUE-XHTML", ns):
            # Filter only English entries by UUID-style ID (5 hyphen-separated parts)
            def_elem = xhtml_val.find("reqif:DEFINITION/reqif:ATTRIBUTE-DEFINITION-XHTML-REF", ns)
            if def_elem is not None and def_elem.text:
                ref_id = def_elem.text.strip()
                if len(ref_id.split('-')) != 5:
                    continue

            value_elem = xhtml_val.find("reqif:THE-VALUE", ns)
            if value_elem is not None:
                # Try extracting <xhtml:p> blocks first
                paragraphs = value_elem.findall(".//xhtml:p", ns)
                if paragraphs:
                    full_text = "\n".join(get_full_text(p) for p in paragraphs if p is not None)
                else:
                    # Fallback to any XHTML structure
                    full_text = get_full_text(value_elem)

                cleaned_text = full_text.strip()
                if not cleaned_text:
                    continue
                if skip_section_headers and section_pattern.match(cleaned_text):
                    continue

                entries.append({column_name: cleaned_text})

        # Clean up
        shutil.rmtree(temp_dir)

        # Save to CSV
        df = pd.DataFrame(entries)
        df = df.drop_duplicates()
        df.to_csv(output_csv_path, index=False)
        #logging.debug(f"Reqifz Extracted and saved: {output_csv_path}")

    def extract_reqifz_to_csv(self, reqifz_path, column_name, output_csv_path):
        """Extract requirements from .reqifz to CSV/Excel"""
        ns = {
            'reqif': 'http://www.omg.org/spec/ReqIF/20110401/reqif.xsd',
            'xhtml': 'http://www.w3.org/1999/xhtml'
        }

        def get_full_text(element):
            text = element.text or ""
            for child in element:
                if child.tag.endswith('a'):
                    href = child.attrib.get('href', '')
                    inner_text = child.text or ""
                    text += f"[{inner_text}]({href})"
                else:
                    text += get_full_text(child)
                if child.tail:
                    text += child.tail
            return text.strip()

        temp_dir = "temp_extracted_reqif"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        with zipfile.ZipFile(reqifz_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        reqif_file = next((f for f in os.listdir(temp_dir) if f.endswith(".reqif")), None)
        if not reqif_file:
            raise FileNotFoundError("No .reqif file found in the .reqifz archive.")

        reqif_path = os.path.join(temp_dir, reqif_file)
        tree = ET.parse(reqif_path)
        root = tree.getroot()

        entries = []
        for xhtml_val in root.findall(".//reqif:ATTRIBUTE-VALUE-XHTML", ns):
            value_elem = xhtml_val.find("reqif:THE-VALUE", ns)
            if value_elem is not None:
                paragraphs = value_elem.findall(".//xhtml:p", ns)
                full_text = "\n".join(get_full_text(p) for p in paragraphs if p is not None)
                if full_text:
                    entries.append({column_name: full_text})

        shutil.rmtree(temp_dir)
        df = pd.DataFrame(entries)
        df = df.drop_duplicates()
        df.to_csv(output_csv_path, index=False)        
        #logging.debug(f"Reqifz Extracted and saved: {output_csv_path}")


    def extract_reqifz_to_csv_CAN(self, reqifz_path, column_name, output_csv_path):
        """Extract requirements from .reqifz to CSV/Excel"""
        ns = {
            'reqif': 'http://www.omg.org/spec/ReqIF/20110401/reqif.xsd',
            'xhtml': 'http://www.w3.org/1999/xhtml'
        }

        section_pattern = re.compile(r'^\d+(\.\d+)*$')

        def get_full_text(element):
            text = element.text or ""
            for child in element:
                if child.tag.endswith('a'):
                    href = child.attrib.get('href', '')
                    inner_text = child.text or ""
                    text += f"[{inner_text}]({href})"
                else:
                    text += get_full_text(child)
                if child.tail:
                    text += child.tail
            return text.strip()

        temp_dir = "temp_extracted_reqif"
    
        # Clear and create temp folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Unzip the reqifz file
        with zipfile.ZipFile(reqifz_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find .reqif file
        reqif_file = next((f for f in os.listdir(temp_dir) if f.endswith(".reqif")), None)
        if not reqif_file:
            raise FileNotFoundError("No .reqif file found in the .reqifz archive.")

        reqif_path = os.path.join(temp_dir, reqif_file)
        
        # Parse XML
        tree = ET.parse(reqif_path)
        root = tree.getroot()

        # Extract XHTML content
        entries = []
        # for xhtml_val in root.findall(".//reqif:ATTRIBUTE-VALUE-XHTML", ns):
        for xhtml_val in root.findall(".//reqif:ATTRIBUTE-VALUE-XHTML", ns):
        # Skip entries with DE-language definition reference
            definition_elem = xhtml_val.find("reqif:DEFINITION/reqif:ATTRIBUTE-DEFINITION-XHTML-REF", ns)
            if definition_elem is not None and definition_elem.text and definition_elem.text.endswith("_de-DE"):
                continue  # skip this entry
            value_elem = xhtml_val.find("reqif:THE-VALUE", ns)
            if value_elem is not None:
                # Try paragraphs first
                paragraphs = value_elem.findall(".//xhtml:p", ns)
                if paragraphs:
                    full_text = "\n".join(get_full_text(p) for p in paragraphs if p is not None)
                else:
                    # Fall back to the entire XHTML block (e.g., <div>, <span>, etc.)
                    full_text = get_full_text(value_elem)
                
                if full_text:
                    #entries.append({column_name: full_text.strip()})
                    cleaned_text = full_text.strip()
                    if not section_pattern.match(cleaned_text):  # Ignore section-like values
                        entries.append({column_name: cleaned_text})
        
        shutil.rmtree(temp_dir)        
        df = pd.DataFrame(entries)
        df = df.drop_duplicates()
        df.to_csv(output_csv_path, index=False)        
        #logging.debug(f"[converter.py] Extracted and saved: {output_csv_path}")

    
    ##<------------------UPDATED fUNCTION------------------------->
    def extract_excel_to_csv(self, excel_path, column_name, output_csv_path):
        """
        Extracts and cleans text data from the first column of an Excel file,
        removing NaN, empty strings, and duplicates, then writes it to a CSV.
        """
        try:
            
            df = pd.read_excel(excel_path)

            # Ensure column exists
            if column_name not in df.columns:
                raise ValueError(f"Column '{column_name}' not found in {excel_path}. Available columns: {list(df.columns)}")

            # Select only the column, drop NaN/empty, reset index
            selected = df[[column_name]].dropna().reset_index(drop=True)

            # Save as CSV
            selected.to_csv(output_csv_path, index=False)

        except Exception as e:
            logging.error(f"[ERROR] Could not process Excel file: {e}")
            raise    
    
    def extract_excel_to_json(
            self, 
            excel_path, 
            column_name, 
            output_json_path, 
            key_name, 
            selected_column: str = None
        ):
        """
        Extracts one column from Excel and saves it as JSON with the given key name.
        Example:
        [
            {"source": "Requirement 1"},
            {"source": "Requirement 2"}
        ]
        """
        try:
            df = pd.read_excel(excel_path)

            # If user provided a specific column, use it
            if selected_column:
                if selected_column not in df.columns:
                    raise ValueError(
                        f"Selected column '{selected_column}' not found in {excel_path}. "
                        f"Available columns: {list(df.columns)}"
                    )
                selected = df[[selected_column]].dropna().reset_index(drop=True)
                selected = selected.rename(columns={selected_column: column_name})
            else:
                # fallback: use column_name directly
                if column_name not in df.columns:
                    raise ValueError(
                        f"Column '{column_name}' not found in {excel_path}. "
                        f"Available columns: {list(df.columns)}"
                    )
                selected = df[[column_name]].dropna().reset_index(drop=True)

            # Convert to JSON-friendly structure
            records = [{key_name: val} for val in selected[column_name].tolist()]

            # Save as JSON
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logging.error(f"[ERROR] Could not process Excel file: {e}")
            raise

    def extract_reqifz_to_json(self, reqifz_path, column_name, output_json_path, key_name):
        """
        Extracts English requirements from a .reqifz file and writes them to JSON.
        """
        ns = {
            'reqif': 'http://www.omg.org/spec/ReqIF/20110401/reqif.xsd',
            'xhtml': 'http://www.w3.org/1999/xhtml'
        }
        section_pattern = re.compile(r'^\d+(\.\d+)*$')

        def get_full_text(element):
            """Recursively extract plain text from an XHTML element."""
            text = element.text or ""
            for child in element:
                if child.tag.endswith('a'):
                    href = child.attrib.get('href', '')
                    inner_text = child.text or ""
                    text += f"[{inner_text}]({href})"
                else:
                    text += get_full_text(child)
                if child.tail:
                    text += child.tail
            return text.strip()

        # Prepare temp directory
        temp_dir = "temp_extracted_reqif"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Unzip the .reqifz file
        with zipfile.ZipFile(reqifz_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find .reqif file
        reqif_file = next((f for f in os.listdir(temp_dir) if f.endswith(".reqif")), None)
        if not reqif_file:
            raise FileNotFoundError("No .reqif file found in the .reqifz archive.")
        reqif_path = os.path.join(temp_dir, reqif_file)

        # Parse XML
        tree = ET.parse(reqif_path)
        root = tree.getroot()

        entries = []
        for xhtml_val in root.findall(".//reqif:ATTRIBUTE-VALUE-XHTML", ns):
            # Filter only English entries by UUID-style ID
            def_elem = xhtml_val.find("reqif:DEFINITION/reqif:ATTRIBUTE-DEFINITION-XHTML-REF", ns)
            if def_elem is not None and def_elem.text:
                ref_id = def_elem.text.strip()
                if len(ref_id.split('-')) != 5:
                    continue

            value_elem = xhtml_val.find("reqif:THE-VALUE", ns)
            if value_elem is not None:
                paragraphs = value_elem.findall(".//xhtml:p", ns)
                if paragraphs:
                    full_text = "\n".join(get_full_text(p) for p in paragraphs if p is not None)
                else:
                    full_text = get_full_text(value_elem)

                cleaned_text = full_text.strip()
                if not cleaned_text:
                    continue
                if section_pattern.match(cleaned_text):
                    continue

                entries.append({key_name: cleaned_text})

        # Clean up
        shutil.rmtree(temp_dir)

        # Save to JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        #logging.debug(f"Reqifz Extracted and saved: {output_json_path}")
