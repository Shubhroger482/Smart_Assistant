# genai_assistant_app/main.py

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables (if needed later)
load_dotenv()

from utils.pdf_parser import extract_text_from_pdf
from utils.chunker import chunk_text
from backend.summarizer import generate_summary
from backend.local_qa import answer_query
from backend.local_evaluator import generate_challenges, evaluate_answer

st.set_page_config(page_title="GenAI Research Assistant", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>📄 GenAI Research Assistant</h1>
    <p style='text-align: center; font-size: 18px;'>Upload a document to summarize, ask questions, or take a challenge quiz</p>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)

# Session state
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
    st.session_state.chunks = []
    st.session_state.mode = None
    st.session_state.chat_history = []

# File uploader with progress message
st.sidebar.header("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    with st.spinner("Reading and summarizing your document..."):
        if uploaded_file.name.endswith(".pdf"):
            st.session_state.doc_text = extract_text_from_pdf(uploaded_file)
        else:
            st.session_state.doc_text = uploaded_file.read().decode("utf-8")

        # Chunk and summarize
        st.session_state.chunks = chunk_text(st.session_state.doc_text)
        summary = generate_summary(st.session_state.doc_text[:3000])

    st.success("✅ Document processed successfully!")
    st.subheader("🔍 Auto Summary (100–150 Words)")
    st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-radius:6px;border-left:4px solid #00BFFF;color:#000'>{summary}</div>", unsafe_allow_html=True)

    st.subheader("🧭 Select Interaction Mode")
    st.session_state.mode = st.radio("Choose a mode:", ["Ask Anything", "Challenge Me"], horizontal=True)

    if st.session_state.mode == "Ask Anything":
        st.markdown("---")
        st.markdown("### 🤖 Ask Questions About Your Document")
        query = st.text_input("Type your question below")
        if query:
            history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history[-3:]])
            full_query = f"{history_context}\nQ: {query}" if history_context else query
            response, reference = answer_query(full_query, st.session_state.chunks)
            st.session_state.chat_history.append((query, response))
            st.markdown(f"**Answer:** {response}")
            st.markdown(f"<span style='background-color:#ffeaa7;border-radius:4px;padding:4px;'>📌 Justification: {reference}</span>", unsafe_allow_html=True)

    elif st.session_state.mode == "Challenge Me":
        st.markdown("---")
        st.markdown("### 🧠 Take the Quiz Challenge")
        st.info("3 logic-based questions generated from your document")
        questions = generate_challenges(st.session_state.chunks)

        user_answers = []
        for i, q in enumerate(questions):
            if any(opt in q for opt in ["A.", "B.", "C.", "D."]):
                st.markdown(f"**Q{i+1}:**")
                st.markdown(q)
                user_input = st.radio(f"Your choice for Q{i+1}", ["A", "B", "C", "D"], key=f"mcq_{i}")
            else:
                user_input = st.text_input(f"Q{i+1}: {q}", key=f"text_{i}")
            user_answers.append((q, user_input))

        if st.button("Submit Answers"):
            import pandas as pd
            import datetime
            score = 0
            total = len(user_answers)
            results = []
            for i, (q, ans) in enumerate(user_answers):
                feedback, reference = evaluate_answer(q, ans, st.session_state.chunks)
                results.append({"Question": q, "Answer": ans, "Feedback": feedback, "Justification": reference})
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**Your Answer:** {ans}")
                st.markdown(f"**Feedback:** {feedback}")
                st.markdown(f"<span style='background-color:#ffeaa7;border-radius:4px;padding:4px;'>📌 Justification: {reference}</span>", unsafe_allow_html=True)
                if "✅" in feedback:
                    score += 1

            st.success(f"🎯 Your Score: {score} / {total}")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_df = pd.DataFrame(results)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Quiz Results (CSV)",
                data=csv,
                file_name=f"quiz_results_{timestamp}.csv",
                mime="text/csv"
            )

