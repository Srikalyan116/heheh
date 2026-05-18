# # # # prompt.py
# MULTI_PAIR_PROMPT = """
# You are an expert requirement-change classifier. For EACH PAIR below, return exactly one JSON object.
# Return ONE JSON ARRAY of objects. Each object MUST have:

# {{
#   "pair_key": "<exact pair_key>",
#   "change_type": "Major Change" | "Minor Change" | "No Change" | "Skipped" | "New" | "Deleted" | "Unidentified",
#   "description": "<concise explanation of semantic content differences>",
#   "change_description": "<same as description>",
#   "formatting_only": true | false,
#   "confidence": 0.0,
#   "row_changes": [
#     {{
#       "row_key": "<WHOLE or row identifier (e.g., a DID or first non-empty cell)>",
#       "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
#       "change_description": "<concise explanation of changes for this row>",
#       "cell_changes": [
#         {{ "col_index": 0, "col_header": "", "old": "", "new": "" }}
#       ]
#     }}
#   ],
#   "added_rows": [
#     {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
#   ],
#   "deleted_rows": [
#     {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
#   ]
# }}

# ### GENERAL RULES
# - Focus ONLY on semantic content. Ignore formatting-only changes (spacing, pipes, markdown, HTML borders, alignment, bold/italic/code ticks).
# - If content is effectively identical after ignoring formatting, set:
#   "formatting_only": true and "change_type": "No Change".
# - If one side is empty and the other is not, use "New" or "Deleted".
# - Return VALID JSON ONLY. No backticks. No commentary.

# For KIND = "table":
# - If CANONICAL_TABLE exists, use it exclusively.
# - Match rows only by row_key; columns only by header.
# - Table-level change_type MUST be derived as follows:
#   • If any added_rows or deleted_rows exist → change_type = "Major Change".
#   • Else if any row_changes with cell_changes exist → change_type = "Major Change {{MAJOR RULES}}" or "Minor Change {{MINOR RULES}}".
#   • Else → change_type = "No Change".   
# - Line breaks, <br>, spacing, list/bracket completion, and note repositioning are formatting-only.
# - If differences are formatting-only, do NOT generate description or row changes; set change_type = "No Change".
# - When change_type is "No Change", **strictly** set description and change_description to "No Change".
# - Put **value differences (ie. old to new values) ONLY** in row_changes[].cell_changes.
# - Do NOT describe added/deleted rows in description text.
# - Row only on one side → added_rows / deleted_rows.
# - Column or row added/removed → Major Change.
# - NEVER use "WHOLE" as row_key inside added_rows or deleted_rows.
# - Each added_rows / deleted_rows entry MUST represent exactly ONE logical table row.
# - row_values MUST contain only the cells of that single row (not headers, not other rows).
# - DO NOT infer semantic change from table layout, structure, or representation.
# - Structural changes (merged/split cells, header movement, text moved into/out of table, table ↔ text, textual adjustments, re-organization of content) are ALWAYS "No Change".
# - Any type of format changes (pipe counts, spacing, bold/italic/code ticks, markdown fences, HTML border/align) are ALWAYS "No Change".
# - Strictly, NEVER infer deletions or additions of rows unless the row_key is explicitly present in the CANONICAL_TABLE.
# - NEVER **infer other rows** or **generate unwanted change description** as added_rows or deleted_rows from nearby pairs.
# - If a row_key does NOT explicitly exist in Version A, it MUST NOT appear in deleted_rows.
# - Addition of a new row does NOT imply replacement or deletion of any other row.

# For KIND = "image":
# - Compare normalized extracted text (top→bottom, left→right).
# - Ignore formatting noise (extra spaces, line breaks, box-drawing characters, markdown fences, mojibake, and OCR artefacts that do NOT change meaning).
# - Identify small semantic differences such as changed labels, added/removed values, altered numbers/percentages, renamed nodes, modified arrows/relationships, or added/removed diagram elements.  
# - **Strictly** for every change, identify the changes from old to new.
# - Treat minor textual edits (spelling, pluralization, small numeric deltas) as "Minor Change".  
# - Treat added/removed lines, changed relationships, or altered data values as "Major Change".

# For KIND = "text":
# - Compare the diffences between the pairs based  on MAJOR, MINOR and NO CHANGE RULES.
# - **Strictly**, for every change, identify and display only the changes.
# - **Strictly**, do NOT repeat unchanged text in description or change_description.

# MAJOR RULES:
# {MAJOR_RULES}

# MINOR RULES:
# {MINOR_RULES}

# NO CHANGE RULES:
# {NO_CHANGE_RULES}

# PAIRS (each pair provides RAW text AND CANONICAL_TABLE for both versions):
# {PAIRS_BLOCK}
# """

### Version2 Working Correctly###
####prompt.py
# MULTI_PAIR_PROMPT = """
# You are an expert requirement-change classifier. You will receive ONE OR MORE PAIRS in this request.

# For EACH PAIR:
# - Treat it as completely independent.
# - Do NOT infer, transfer, copy, reuse, or apply any schema, row template, header, row_key, table structure, rules, patterns, semantics, or deletion logic from any other PAIR.
# - Reset all internal state between PAIRs.
# - Do NOT assume continuity across PAIRs.

# For EACH PAIR, you must output exactly ONE JSON OBJECT with the following schema:

# {{
#   "pair_key": "<exact pair_key>",
#   "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
#   "description": "<concise explanation of semantic content differences>",
#   "change_description": "<same as description>",
#   "formatting_only": true | false,
#   "confidence": 0.0,
#   "row_changes": [
#     {{
#       "row_key": "<WHOLE or row identifier>",
#       "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
#       "change_description": "<concise explanation of changes for this row>",
#       "cell_changes": [
#         {{ "col_index": 0, "col_header": "", "old": "", "new": "" }}
#       ]
#     }}
#   ],
#   "added_rows": [
#     {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
#   ],
#   "deleted_rows": [
#     {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
#   ]
# }}

# GLOBAL RULES:
# - Return ONE JSON ARRAY containing one JSON object per PAIR.
# - Return VALID JSON ONLY. No backticks. No comments. No explanations.
# - Pairs are STRICTLY INDEPENDENT.
# - NEVER use information, tables, schemas, headers, rows, examples, inference, or assumptions from OTHER PAIRS.

# ---------------------------------------
# KIND-SPECIFIC SEMANTIC RULES
# ---------------------------------------

# KIND = "table"
# - If CANONICAL_TABLE exists, use it exclusively.
# - Match rows ONLY by row_key; columns ONLY by header.
# - Table-level change_type MUST be derived as:
#   • If any added_rows or deleted_rows exist → "Major Change"
#   • Else if any row_changes with cell_changes exist → "Major Change" or "Minor Change"
#   • Else → "No Change"
# - **Formatting-only differences** (HTML/Markdown layout, spacing, br tags, bold/italic, alignment) → "No Change".
# - **Structural changes** (cell merge/split, header relocation, pipe count changes, markdown formatting, formatting adjustments) are ALWAYS "No Change".
# - When change_type is "No Change", **strictly** set description and change_description to "No Change".
# - If table has MORE THAN 20 rows OR the number of changed rows exceeds 10.:
#   → DO NOT output row_changes
#   → DO NOT output full added_rows / deleted_rows  
#   → ONLY output:
#       • change_type
#       • description
#       • count of added rows
#       • count of deleted rows
#   → The description MUST identify:
#       • which columns changed  
#       • whether additions occurred  
#       • whether deletions occurred
#   → row_changes is OPTIONAL.
#   → added_rows / deleted_rows are OPTIONAL for large tables.
#   → NEVER generate large JSON outputs for big tables.
# - **Strictly**, for every change, identify and display only the changes.
# - Do NOT describe added/deleted rows in description.
# - Do NOT hallucinate missing rows.
# - NEVER infer deletions or additions from context.
# - NEVER import row schemas from previous pairs.
# - Row absent on one side → added_rows or deleted_rows (ONE PER LOGICAL ROW).
# - Identify all the added rows very carefully and specifically for each table pairs.
# - NEVER use "WHOLE" for added_rows or deleted_rows.
# - DO NOT generate row deletions unless row_key explicitly exists in old table.
# - DO NOT generate row additions unless row_key explicitly exists in new table.

# KIND = "image":
# - Compare normalized extracted text (top→bottom, left→right).
# - Ignore formatting noise (extra spaces, line breaks, box-drawing characters, markdown fences, mojibake, and OCR artefacts that do NOT change meaning).
# - Identify small semantic differences such as changed labels, added/removed values, altered numbers/percentages, renamed nodes, modified arrows/relationships, or added/removed diagram elements.  
# - **Strictly** for every change, identify the changes from old to new.
# - Treat minor textual edits (spelling, pluralization, small numeric deltas) as "Minor Change".  
# - Treat added/removed lines, changed relationships, or altered data values as "Major Change".

# KIND = "text":
# - Compare the diffences between the pairs based  on MAJOR, MINOR and NO CHANGE RULES.
# - **Strictly**, for every change, identify and display only the changes.
# - **Strictly**, do NOT repeat unchanged text in description or change_description.

# ---------------------------------------
# PAIR ISOLATION RULES (CRITICAL)
# ---------------------------------------
# - After finishing a PAIR, discard all inferred schemas.
# - DO NOT carry row_keys, column sets, header structures, or change heuristics to the next PAIR.
# - DO NOT align rows based on previous PAIR patterns.
# - DO NOT reuse row deletion/addition behavior across PAIRs.

# ---------------------------------------
# PAIR BOUNDARIES
# ---------------------------------------
# Each PAIR is wrapped like:

# ===PAIR START===
# pair_key: "<id>"
# kind: "<table|text|image>"
# version_1:
# <RAW VERSION 1>
# version_2:
# <RAW VERSION 2>
# ===PAIR END===

# ---------------------------------------
# MAJOR RULES:
# {MAJOR_RULES}

# MINOR RULES:
# {MINOR_RULES}

# NO CHANGE RULES:
# {NO_CHANGE_RULES}

# ---------------------------------------
# PAIRS:
# {PAIRS_BLOCK}
# """


####Version3: benchmark
# MULTI_PAIR_PROMPT = """
# You are an expert requirement-change classifier. You will receive ONE OR MORE PAIRS in this request.

# For EACH PAIR:
# - Treat it as completely independent.
# - Do NOT infer, transfer, copy, reuse, or apply any schema, row template, header, row_key, table structure, rules, patterns, semantics, or deletion logic from any other PAIR.
# - Reset all internal state between PAIRs.
# - Do NOT assume continuity across PAIRs.

# For EACH PAIR, you must output exactly ONE JSON OBJECT with the following schema:

# {{
#   "pair_key": "<exact pair_key>",
#   "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
#   "description": "<concise explanation of semantic content differences>",
#   "change_description": "<same as description>",
#   "formatting_only": true | false,
#   "confidence": 0.0,
#   "row_changes": [
#     {{
#       "row_key": "<WHOLE or row identifier>",
#       "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
#       "change_description": "<concise explanation of changes for this row>",
#       "cell_changes": [
#         {{ "col_index": 0, "col_header": "", "old": "", "new": "" }}
#       ]
#     }}
#   ],
#   "added_rows": [
#     {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
#   ],
#   "deleted_rows": [
#     {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
#   ]
# }}

# GLOBAL RULES:
# - Return ONE JSON ARRAY containing one JSON object per PAIR.
# - Return VALID JSON ONLY. No backticks. No comments. No explanations.
# - Pairs are STRICTLY INDEPENDENT.
# - NEVER use information, tables, schemas, headers, rows, examples, inference, or assumptions from OTHER PAIRS.

# ---------------------------------------
# KIND-SPECIFIC SEMANTIC RULES
# ---------------------------------------

# KIND = "table"
# - Input tables may be provided in raw HTML. When HTML is received, interpret the <table> structure directly.
# - Do NOT assume markdown formatting. Do NOT infer changes based on formatting differences.
# - Use the HTML table rows (<tr>) and cells (<td>/<th>) as the definitive source of truth.
# - If CANONICAL_TABLE exists, use it exclusively.
# - Match rows ONLY by row_key; columns ONLY by header.
# - **Strictly**, identify if content got added or deleted in any cell.
# ---------------------------------------
# CHANGE CLASSIFICATION RULES
# ---------------------------------------
# - Table-level change_type MUST be derived as:
#   • If any added_rows or deleted_rows exist → "Major Change"
#   • Else if any row_changes with cell_changes exist → "Major Change" or "Minor Change"
#   • Else → "No Change"
# - Any change from an empty cell to a non-empty cell (e.g., "", " ", or <td></td> → "No", "Yes", "N/A") MUST be treated as a "Major Change".
# - Do not present or mention any spelling changes in the change description.
# - Formatting-only differences (HTML/Markdown layout, spacing, alignment, markdown pipes) → "No Change".
# - Structural changes (cell merge/split, alignment changes, formatting adjustments) → "No Change".

# - When change_type is "No Change":
#     • description = "No Change"
#     • change_description = "No Change"
# ---------------------------------------
# LARGE-TABLE / TOO-MANY-CHANGES RULE
# ---------------------------------------
# - If the table has MORE THAN 20 rows OR the number of changed rows exceeds 10:
#     → Activate SUMMARY MODE.
#     → Output ONLY the first 4 changed rows as concrete examples in row_changes.
#     → row_changes MUST NOT contain more than 4 rows.
#     → DO NOT detail remaining changed rows.
#     → DO NOT output full added_rows / deleted_rows.

#     → description MUST include a HIGH-LEVEL SUMMARY containing:
#         • explicit list of all column_headers where ANY change occurred  
#         • approximate count of modified rows (e.g., "17 additional rows modified")
#         • number of added rows
#         • number of deleted rows   
       
#     → NEVER output large JSON or full row diffs in summary mode.
# - Strictly, do not over‑explain or repeat any change descriptions.
# - Strictl, report only the text fragment that changed. NEVER include full cell values in the change description.
# - NEVER display the entire text for what got changed, simply mention what changed. 
# - Do NOT describe added/deleted rows in description.
# - Do NOT hallucinate missing rows.
# - NEVER infer deletions or additions from context.
# - NEVER import row schemas from previous pairs.
# - Row absent on one side → added_rows or deleted_rows (ONE PER LOGICAL ROW).
# - Identify all the added rows very carefully and specifically for each table pairs.
# - NEVER use "WHOLE" for added_rows or deleted_rows.
# - DO NOT generate row deletions unless row_key explicitly exists in old table.
# - DO NOT generate row additions unless row_key explicitly exists in new table.

# KIND = "image":
# - Compare normalized extracted text (top→bottom, left→right).
# - Ignore formatting noise (extra spaces, line breaks, box-drawing characters, markdown fences, mojibake, and OCR artefacts that do NOT change meaning).
# - Identify small semantic differences such as changed labels, added/removed values, altered numbers/percentages, renamed nodes, modified arrows/relationships, or added/removed diagram elements.  
# - **Strictly** for every change, identify the changes from old to new.
# - Treat minor textual edits (spelling, pluralization, small numeric deltas) as "Minor Change".  
# - Treat added/removed lines, changed relationships, or altered data values as "Major Change".

# KIND = "text":
# - Compare the diffences between the pairs based  on MAJOR, MINOR and NO CHANGE RULES.
# - **Strictly**, for every change, identify and display only the changes.
# - **Strictly**, do NOT repeat unchanged text in description or change_description.

# ---------------------------------------
# PAIR ISOLATION RULES (CRITICAL)
# ---------------------------------------
# - After finishing a PAIR, discard all inferred schemas.
# - DO NOT carry row_keys, column sets, header structures, or change heuristics to the next PAIR.
# - DO NOT align rows based on previous PAIR patterns.
# - DO NOT reuse row deletion/addition behavior across PAIRs.

# ---------------------------------------
# PAIR BOUNDARIES
# ---------------------------------------
# Each PAIR is wrapped like:

# ===PAIR START===
# pair_key: "<id>"
# kind: "<table|text|image>"
# version_1:
# <RAW VERSION 1>
# version_2:
# <RAW VERSION 2>
# ===PAIR END===

# ---------------------------------------
# MAJOR RULES:
# {MAJOR_RULES}

# MINOR RULES:
# {MINOR_RULES}

# NO CHANGE RULES:
# {NO_CHANGE_RULES}

# ---------------------------------------
# PAIRS:
# {PAIRS_BLOCK}
# """

MULTI_PAIR_PROMPT = """
You are an expert requirement-change classifier. You will receive ONE OR MORE PAIRS in this request.

For EACH PAIR:
- Treat it as completely independent.
- Do NOT infer, transfer, copy, reuse, or apply any schema, row template, header, row_key, table structure, rules, patterns, semantics, or deletion logic from any other PAIR.
- Reset all internal state between PAIRs.
- Do NOT assume continuity across PAIRs.

For EACH PAIR, you must output exactly ONE JSON OBJECT with the following schema:

{{
  "pair_key": "<exact pair_key>",
  "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
  "description": "<concise explanation of semantic content differences>",
  "change_description": "<same as description>",
  "formatting_only": true | false,
  "confidence": 0.0,
  "row_changes": [
    {{
      "row_key": "<WHOLE or row identifier>",
      "change_type": "Major Change" | "Minor Change" | "No Change" | "New" | "Deleted" | "Skipped" | "Unidentified",
      "change_description": "<concise explanation of changes for this row>",
      "cell_changes": [
        {{ "col_index": 0, "col_header": "", "old": "", "new": "" }}
      ]
    }}
  ],
  "added_rows": [
    {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
  ],
  "deleted_rows": [
    {{ "row_key": "<key>", "row_values": ["col0", "col1", "..."] }}
  ]
}}

GLOBAL RULES:
- Return ONE JSON ARRAY containing one JSON object per PAIR.
- Return VALID JSON ONLY. No backticks. No comments. No explanations.
- Pairs are STRICTLY INDEPENDENT.
- NEVER use information, tables, schemas, headers, rows, examples, inference, or assumptions from OTHER PAIRS.

---------------------------------------
KIND-SPECIFIC SEMANTIC RULES
---------------------------------------

KIND = "table"
- Input tables may be provided as raw HTML. When HTML is received, you MUST interpret the <table>, <tr>, <td>, and <th> structure directly.
- Treat each <tr> as a row and each <td>/<th> as a cell in that row. Preserve the exact positional alignment. Do NOT infer or reorder rows or columns.
- Do NOT assume markdown formatting. Do NOT infer changes based on formatting differences.
- If CANONICAL_TABLE exists, use it exclusively.
- Match rows ONLY by row_key; columns ONLY by header.
- **Strictly**, identify if content got added or deleted in any cell.
---------------------------------------
CHANGE CLASSIFICATION RULES
---------------------------------------
- Table-level change_type MUST be derived as:
  • If any added_rows or deleted_rows exist → "Major Change"
  • Else if any row_changes with cell_changes exist → "Major Change" or "Minor Change"
  • Else → "No Change"
- Any change from an empty cell to a non-empty cell (e.g., "", " ", or <td></td> → "No", "Yes", "N/A") MUST be treated as a "Major Change".
- Do not present or mention any spelling changes in the change description.
- Formatting-only differences (HTML/Markdown layout, spacing, alignment, markdown pipes) → "No Change".
- Structural changes (cell merge/split, alignment changes, formatting adjustments) → "No Change".

- When change_type is "No Change":
    • description = "No Change"
    • change_description = "No Change"
---------------------------------------
LARGE-TABLE / TOO-MANY-CHANGES RULE
---------------------------------------
- If the table has MORE THAN 20 rows OR the number of changed rows exceeds 10:
    → Activate SUMMARY MODE.
    → Output ONLY the first 4 changed rows as concrete examples in row_changes.
    → row_changes MUST NOT contain more than 4 rows.
    → DO NOT detail remaining changed rows.
    → DO NOT output full added_rows / deleted_rows.

    → description MUST include a HIGH-LEVEL SUMMARY containing:
        • explicit list of all column_headers where ANY change occurred  
        • approximate count of modified rows (e.g., "17 additional rows modified")
        • number of added rows
        • number of deleted rows   
       
    → NEVER output large JSON or full row diffs in summary mode.
- Strictly, do not over‑explain or repeat any change descriptions.
- Strictl, report only the text fragment that changed. NEVER include full cell values in the change description.
- NEVER display the entire text for what got changed, simply mention what changed. 
- Do NOT describe added/deleted rows in description.
- Do NOT hallucinate missing rows.
- NEVER infer deletions or additions from context.
- NEVER import row schemas from previous pairs.
- Row absent on one side → added_rows or deleted_rows (ONE PER LOGICAL ROW).
- Identify all the added rows very carefully and specifically for each table pairs.
- NEVER use "WHOLE" for added_rows or deleted_rows.
- DO NOT generate row deletions unless row_key explicitly exists in old table.
- DO NOT generate row additions unless row_key explicitly exists in new table.

KIND = "image":
- Compare normalized extracted text (top→bottom, left→right).
- Ignore formatting noise (extra spaces, line breaks, box-drawing characters, markdown fences, mojibake, and OCR artefacts that do NOT change meaning).
- Identify small semantic differences such as changed labels, added/removed values, altered numbers/percentages, renamed nodes, modified arrows/relationships, or added/removed diagram elements.  
- **Strictly** for every change, identify the changes from old to new.
- Treat minor textual edits (spelling, pluralization, small numeric deltas) as "Minor Change".  
- Treat added/removed lines, changed relationships, or altered data values as "Major Change".

KIND = "text":
- Compare the diffences between the pairs based  on MAJOR, MINOR and NO CHANGE RULES.
- **Strictly**, for every change, identify and display only the changes.
- **Strictly**, do NOT repeat unchanged text in description or change_description.

---------------------------------------
PAIR ISOLATION RULES (CRITICAL)
---------------------------------------
- After finishing a PAIR, discard all inferred schemas.
- DO NOT carry row_keys, column sets, header structures, or change heuristics to the next PAIR.
- DO NOT align rows based on previous PAIR patterns.
- DO NOT reuse row deletion/addition behavior across PAIRs.

---------------------------------------
PAIR BOUNDARIES
---------------------------------------
Each PAIR is wrapped like:

===PAIR START===
pair_key: "<id>"
kind: "<table|text|image>"
version_1:
<RAW VERSION 1>
version_2:
<RAW VERSION 2>
===PAIR END===

---------------------------------------
MAJOR RULES:
{MAJOR_RULES}

MINOR RULES:
{MINOR_RULES}

NO CHANGE RULES:
{NO_CHANGE_RULES}

---------------------------------------
PAIRS:
{PAIRS_BLOCK}
"""