import os
import io
import json
import re
from collections import defaultdict
from typing import Any

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="JEE Main PYQ Deduplicator", page_icon="📚", layout="wide")

st.title("📚 JEE Main PYQ Deduplicator")
st.write(
    "Upload a JEE Main PYQ PDF. The app extracts questions, groups duplicates and "
    "near-equivalent questions, calculates occurrence frequency, and lets you inspect "
    "the original questions inside each group."
)
st.info(
    "This is a local prototype. Your PDF is processed by this app and sent to the Gemini API "
    "for question extraction/grouping. Add your own Gemini API key in the sidebar."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get an API key from Google AI Studio. Keep it private."
    )
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash")
    batch_size = st.slider(
        "Pages per extraction batch", min_value=1, max_value=8, value=4,
        help="Smaller batches can help with dense PDFs or PDFs with many diagrams."
    )
    st.caption("Tip: start with a chapter PDF of 20–100 questions.")

def extract_page_content(pdf_bytes: bytes) -> list[dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        # Render each page to an image as well, so visual equations/diagrams can be reviewed by the model.
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pages.append({
            "page_number": i + 1,
            "text": text,
            "image_bytes": pix.tobytes("png"),
        })
    return pages

def call_gemini(client, model_name: str, prompt: str, images: list[bytes] | None = None) -> str:
    parts: list[Any] = [types.Part.from_text(text=prompt)]
    for img in images or []:
        parts.append(types.Part.from_bytes(data=img, mime_type="image/png"))
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return response.text or ""

def parse_json_response(raw: str) -> Any:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)

def extract_questions(client, model_name: str, pages: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    all_questions = []
    for start in range(0, len(pages), batch_size):
        batch = pages[start:start + batch_size]
        page_notes = "\n\n".join(
            f"PAGE {p['page_number']} EXTRACTED TEXT:\n{p['text'] or '[No selectable text; inspect page image]'}"
            for p in batch
        )
        prompt = f"""
You are extracting JEE MAIN previous-year questions from a PDF.
Return ONLY valid JSON with this schema:
{{
  "questions": [
    {{
      "temporary_id": "Q1",
      "question_text": "faithful transcription of the full question",
      "options": ["A ...", "B ..."],
      "answer_or_solution_if_printed": "text or empty string",
      "subject": "Physics/Chemistry/Mathematics/Unknown",
      "topic": "best-effort topic",
      "source_page": 1,
      "printed_question_number": "number/label if visible, else empty string",
      "exam_year_or_session": "if visible, else empty string",
      "has_diagram": false,
      "diagram_description": "brief description if needed"
    }}
  ]
}}
Rules:
- Extract actual questions only, not headings, worked examples, answer keys by themselves, or instructions.
- Keep mathematical notation, values, units, conditions, and answer options faithful.
- If a question spans pages, join it into one question.
- Do not invent missing text or years.
- If a page has no questions, return no entry for it.
- Use source_page as the PDF page number, not a printed page number.
- Assign temporary IDs uniquely within this batch.
- Preserve diagram-dependent information in diagram_description.
PAGE TEXT:
{page_notes}
"""
        raw = call_gemini(client, model_name, prompt, [p["image_bytes"] for p in batch])
        data = parse_json_response(raw)
        for q in data.get("questions", []):
            q["source_page"] = q.get("source_page") or batch[0]["page_number"]
            q["global_id"] = f"Q{len(all_questions) + 1}"
            all_questions.append(q)
    return all_questions

def group_questions(client, model_name: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    # Split very large lists into manageable comparison batches; each batch includes all question text.
    compact = []
    for i, q in enumerate(questions):
        compact.append({
            "id": q["global_id"],
            "text": q.get("question_text", ""),
            "options": q.get("options", []),
            "subject": q.get("subject", ""),
            "topic": q.get("topic", ""),
            "diagram_description": q.get("diagram_description", ""),
            "page": q.get("source_page"),
            "printed_number": q.get("printed_question_number", ""),
            "year_session": q.get("exam_year_or_session", "")
        })
    prompt = f"""
You are a careful JEE MAIN question deduplication analyst. Group questions that are repetitions,
near-repetitions, or equivalent enough that practicing one representative question would largely
cover the same pattern. Do NOT merge merely related questions if their setup, reasoning, constraints,
or application differs meaningfully. Prefer keeping more questions over incorrectly merging distinct
ones.

Return ONLY valid JSON in this schema:
{{
  "groups": [
    {{
      "group_id": "G1",
      "representative_question_id": "Q1",
      "member_question_ids": ["Q1", "Q5"],
      "relationship": "exact_duplicate / near_duplicate / equivalent_pattern / distinct",
      "reason": "short explanation",
      "confidence": 0.0
    }}
  ],
  "unassigned_question_ids": []
}}
Rules:
- Every question ID must appear in exactly one group or unassigned list.
- Each group must have at least one member.
- representative_question_id must be a member.
- A singleton is allowed and should be represented as a group with one member.
- Do not merge questions solely because they share a chapter or formula.
- Check numerical values, diagrams, answer options, boundary conditions, and the actual solution method.
- confidence is a number from 0 to 1.
- Group members should include the representative ID.
- Do not create groups with relationship "distinct"; use singleton groups instead.
QUESTIONS:
{json.dumps(compact, ensure_ascii=False)}
"""
    raw = call_gemini(client, model_name, prompt)
    data = parse_json_response(raw)

    # Validate and repair missing IDs conservatively as singletons.
    known = {q["global_id"] for q in questions}
    seen = set()
    clean_groups = []
    for idx, g in enumerate(data.get("groups", []), 1):
        members = [m for m in g.get("member_question_ids", []) if m in known and m not in seen]
        rep = g.get("representative_question_id")
        if rep not in members:
            rep = members[0] if members else None
        if not members:
            continue
        seen.update(members)
        clean_groups.append({
            "group_id": g.get("group_id", f"G{idx}"),
            "representative_question_id": rep,
            "member_question_ids": members,
            "relationship": g.get("relationship", "equivalent_pattern"),
            "reason": g.get("reason", ""),
            "confidence": float(g.get("confidence", 0.5) or 0.5),
        })
    for qid in sorted(known - seen, key=lambda x: int(x[1:])):
        clean_groups.append({
            "group_id": f"G{len(clean_groups)+1}",
            "representative_question_id": qid,
            "member_question_ids": [qid],
            "relationship": "singleton",
            "reason": "No safe match was assigned; retained separately.",
            "confidence": 1.0,
        })
    data["groups"] = clean_groups
    return data

def question_label(q: dict[str, Any]) -> str:
    bits = [q.get("global_id", "")]
    if q.get("printed_question_number"):
        bits.append(f"printed #{q['printed_question_number']}")
    bits.append(f"PDF p.{q.get('source_page', '?')}")
    if q.get("exam_year_or_session"):
        bits.append(str(q["exam_year_or_session"]))
    return " · ".join(bits)

uploaded = st.file_uploader("Upload a PYQ PDF", type=["pdf"])
if uploaded:
    st.caption(f"File: {uploaded.name} · {uploaded.size / 1024:.0f} KB")
    if not api_key:
        st.warning("Enter your Gemini API key in the sidebar to process the PDF.")
    if st.button("Extract, deduplicate and build report", type="primary", disabled=not api_key):
        try:
            client = genai.Client(api_key=api_key)
            pdf_bytes = uploaded.getvalue()
            with st.spinner("Reading PDF pages and diagrams..."):
                pages = extract_page_content(pdf_bytes)
            st.write(f"Read {len(pages)} PDF pages.")
            with st.spinner("Extracting individual questions..."):
                questions = extract_questions(client, model_name, pages, batch_size)
            if not questions:
                st.error("No questions were extracted. Try a clearer PDF or fewer pages per batch.")
                st.stop()
            with st.spinner(f"Comparing {len(questions)} extracted questions..."):
                grouping_data = group_questions(client, model_name, questions)
            st.session_state["pyq_questions"] = questions
            st.session_state["pyq_groups"] = grouping_data["groups"]
            st.session_state["pyq_filename"] = uploaded.name
            st.success(f"Done: extracted {len(questions)} questions and found {len(grouping_data['groups'])} representative groups.")
        except json.JSONDecodeError:
            st.error("The AI returned invalid JSON. Try again, use a smaller page batch, or choose a supported Gemini model.")
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.caption("Check your API key, internet connection, model name, and Gemini API quota.")

if "pyq_groups" in st.session_state:
    questions = st.session_state["pyq_questions"]
    groups = st.session_state["pyq_groups"]
    q_by_id = {q["global_id"]: q for q in questions}
    st.divider()
    st.header("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Questions extracted", len(questions))
    c2.metric("Unique groups", len(groups))
    c3.metric("Repeated instances", len(questions) - len(groups))
    st.write("Each group's frequency is the total number of original questions in that group, including the representative.")

    rows = []
    for g in groups:
        rep = q_by_id[g["representative_question_id"]]
        rows.append({
            "Representative": rep["global_id"],
            "Frequency": len(g["member_question_ids"]),
            "Relationship": g.get("relationship", ""),
            "Confidence": round(float(g.get("confidence", 0)), 2),
            "Topic": rep.get("topic", ""),
            "Source": question_label(rep),
            "Reason": g.get("reason", ""),
        })
    df = pd.DataFrame(rows).sort_values(["Frequency", "Representative"], ascending=[False, True])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Inspect a question and its repeats")
    rep_options = [
        f"{q_by_id[g['representative_question_id']]['global_id']} — "
        f"{q_by_id[g['representative_question_id']].get('topic','Unknown topic')} "
        f"({len(g['member_question_ids'])} occurrence(s))"
        for g in groups
    ]
    selected_label = st.selectbox("Choose a representative question", rep_options)
    selected_index = rep_options.index(selected_label)
    selected_group = groups[selected_index]
    representative = q_by_id[selected_group["representative_question_id"]]

    st.markdown(f"### {question_label(representative)}")
    st.write(representative.get("question_text", ""))
    if representative.get("options"):
        for option in representative["options"]:
            st.write(f"- {option}")
    if representative.get("diagram_description"):
        st.caption("Diagram: " + representative["diagram_description"])
    st.caption(
        f"Topic: {representative.get('topic','Unknown')} · "
        f"Subject: {representative.get('subject','Unknown')} · "
        f"Group confidence: {selected_group.get('confidence', 0):.2f}"
    )
    st.write("**Why these were grouped:** " + selected_group.get("reason", "No explanation returned."))

    with st.expander(f"View repeats ({len(selected_group['member_question_ids']) - 1} other question(s))", expanded=True):
        if len(selected_group["member_question_ids"]) == 1:
            st.write("No repeats were detected for this question.")
        else:
            for qid in selected_group["member_question_ids"]:
                if qid == representative["global_id"]:
                    continue
                q = q_by_id[qid]
                st.markdown(f"**{question_label(q)}**")
                st.write(q.get("question_text", ""))
                if q.get("options"):
                    for option in q["options"]:
                        st.write(f"- {option}")
                if q.get("diagram_description"):
                    st.caption("Diagram: " + q["diagram_description"])
                st.divider()

    # Export detailed report
    detail_rows = []
    for g in groups:
        rep = q_by_id[g["representative_question_id"]]
        for qid in g["member_question_ids"]:
            q = q_by_id[qid]
            detail_rows.append({
                "group_id": g["group_id"],
                "representative_id": rep["global_id"],
                "frequency_in_group": len(g["member_question_ids"]),
                "relationship": g.get("relationship", ""),
                "confidence": g.get("confidence", ""),
                "grouping_reason": g.get("reason", ""),
                "question_id": q["global_id"],
                "is_representative": qid == rep["global_id"],
                "question_text": q.get("question_text", ""),
                "options": "\n".join(q.get("options", [])),
                "subject": q.get("subject", ""),
                "topic": q.get("topic", ""),
                "pdf_page": q.get("source_page", ""),
                "printed_question_number": q.get("printed_question_number", ""),
                "exam_year_or_session": q.get("exam_year_or_session", ""),
                "diagram_description": q.get("diagram_description", ""),
            })
    report_df = pd.DataFrame(detail_rows)
    csv_bytes = report_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download full frequency report (CSV)", csv_bytes, "pyq_frequency_report.csv", "text/csv")

    representative_rows = []
    for g in groups:
        q = q_by_id[g["representative_question_id"]]
        representative_rows.append({
            "Unique question ID": q["global_id"],
            "Topic": q.get("topic", ""),
            "Frequency": len(g["member_question_ids"]),
            "Question": q.get("question_text", ""),
            "Options": "\n".join(q.get("options", [])),
            "Original reference": question_label(q),
            "Grouped IDs": ", ".join(g["member_question_ids"]),
            "Grouping reason": g.get("reason", ""),
        })
    unique_df = pd.DataFrame(representative_rows)
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        unique_df.to_excel(writer, sheet_name="Unique questions", index=False)
        report_df.to_excel(writer, sheet_name="All grouped questions", index=False)
    st.download_button(
        "Download Excel workbook", xlsx_buffer.getvalue(),
        "jee_main_pyq_deduplicated.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.warning(
        "Review low-confidence groups and diagram-heavy questions before relying on the set. "
        "AI grouping is a useful assistant, not a guaranteed mathematical proof of equivalence."
    )
