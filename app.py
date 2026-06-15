import streamlit as st
import pandas as pd
import PyPDF2
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

st.set_page_config(
    page_title="AI ATS Resume Screener",
    layout="wide"
)

st.title("AI ATS Resume Screener")


def extract_text(pdf_file):

    text = ""

    try:

        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

    except Exception as e:

        st.error(
            f"PDF Reading Error: {e}"
        )

    return text


def analyze_resume(
    resume_text,
    job_description
):

    prompt = f"""
You are an ATS Resume Analyzer.

Compare Resume and Job Description.

Return ONLY valid JSON.

Format:

{{
    "match_percentage": 0,
    "matching_skills": [],
    "missing_skills": [],
    "strengths": [],
    "improvements": [],
    "recommendation": ""
}}

Resume:

{resume_text}

Job Description:

{job_description}
"""

    try:

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            response_format={
                "type": "json_object"
            }

        )

        return response.choices[0].message.content

    except Exception as e:

        st.error(
            f"Groq API Error: {e}"
        )

        return None


uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area(
    "Paste Your Job Description"
)

if st.button("Analyze Resumes"):

    # to avoid empty resume check

    if not uploaded_files:

        st.warning(
            "Please upload at least one resume."
        )

        st.stop()

    if not job_description:

        st.warning(
            "Please enter Job Description."
        )

        st.stop()

    results = []

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    for file in uploaded_files:

        resume_text = extract_text(
            file
        )

        ai_response = analyze_resume(
            resume_text,
            job_description
        )

        if not ai_response:

            continue

        try:

            data = json.loads(
                ai_response
            )

        except Exception as e:

            st.error(
                f"JSON Error in {file.name}: {e}"
            )

            continue

        results.append({

            "Resume":
            file.name,

            "Match %":
            int(
                data.get(
                    "match_percentage",
                    0
                )
            ),

            "Matching Skills":
            ", ".join(
                data.get(
                    "matching_skills",
                    []
                )
            ),

            "Missing Skills":
            ", ".join(
                data.get(
                    "missing_skills",
                    []
                )
            ),

            "Recommendation":
            data.get(
                "recommendation",
                ""
            )

        })

    if len(results) == 0:

        st.error(
            "No valid results found."
        )

        st.stop()

    df = pd.DataFrame(
        results
    )

    # for sorting(high score first)

    df = df.sort_values(
        by="Match %",
        ascending=False
    )

    threshold = 70

    matched = df[
        df["Match %"] >= threshold
    ]

    unmatched = df[
        df["Match %"] < threshold
    ]

    matched.to_csv(
        "outputs/matched.csv",
        index=False
    )

    unmatched.to_csv(
        "outputs/unmatched.csv",
        index=False
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Matched Candidates"
        )

        st.dataframe(
            matched,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Unmatched Candidates"
        )

        st.dataframe(
            unmatched,
            use_container_width=True
        )

    # for Download Resume

    st.download_button(
        "Download Matched",
        matched.to_csv(index=False),
        "matched.csv",
        "text/csv"
    )

    st.download_button(
        "Download Unmatched",
        unmatched.to_csv(index=False),
        "unmatched.csv",
        "text/csv"
    )