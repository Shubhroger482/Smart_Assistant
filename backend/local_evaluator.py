from sentence_transformers import SentenceTransformer, util
import numpy as np
import random

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_challenges(chunks: list, n=3):
    top_chunks = sorted(chunks, key=lambda x: len(x), reverse=True)[:10]
    available = min(n, len(top_chunks))

    if available == 0:
        return ["Not enough content to generate questions. Please upload a longer document."]

    selected = random.sample(top_chunks, available)
    questions = []

    question_types = [
        "Summarize the main idea of the following:",
        "What issue is being addressed here:",
        "Why is this topic significant:",
        "Explain the reasoning behind this section:",
        "What implications does this have:",
        "Which of the following best describes the intent of this section:",
        "What conclusion can be drawn from the following text:"
    ]

    mcq_options = [
        ["A. It discusses background context", "B. It presents a solution", "C. It raises a question", "D. It offers no useful info"],
        ["A. Innovation", "B. Policy", "C. Research", "D. Unclear"],
        ["A. Technical analysis", "B. Summary", "C. Case study", "D. Opinion"]
    ]

    for chunk in selected:
        prefix = random.choice(question_types)
        formatted_chunk = chunk[:200].strip()
        if "Which of the following" in prefix or "conclusion" in prefix:
            options = random.choice(mcq_options)
            q = f"{prefix}\n\"{formatted_chunk}...\"\n\n" + "\n".join(options)
        else:
            q = f"{prefix}\n\"{formatted_chunk}...\""
        questions.append(q)

    return questions


from sentence_transformers import SentenceTransformer, util
import numpy as np
import random
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def has_reasoning_structure(answer: str) -> bool:
    reasoning_patterns = [
        r"\bbecause\b", r"\btherefore\b", r"\bas a result\b", r"\bdue to\b", r"\bthis implies\b", r"\bconsequently\b"
    ]
    return any(re.search(pattern, answer.lower()) for pattern in reasoning_patterns)

def contains_keywords(answer: str, reference: str) -> bool:
    answer_words = set(answer.lower().split())
    ref_words = set(reference.lower().split())
    overlap = answer_words.intersection(ref_words)
    return len(overlap) >= 3

def evaluate_answer(question: str, user_answer: str, chunks: list):
    # Extract reference context
    question_context = question.split('\"')[1] if '\"' in question else ""
    reference_embedding = model.encode([question_context])[0]
    answer_embedding = model.encode([user_answer])[0]

    similarity = util.cos_sim(reference_embedding, answer_embedding).item()

    score = 0
    if contains_keywords(user_answer, question_context):
        score += 1
    if has_reasoning_structure(user_answer):
        score += 1
    if similarity > 0.5:
        score += 1

    if score == 3:
        feedback = "✅ Excellent reasoning and content alignment!"
    elif score == 2:
        feedback = "✅ Good answer, but could be clearer or more structured."
    elif score == 1:
        feedback = "❌ Limited relevance or logic in the response."
    else:
        feedback = "❌ Needs significant improvement in reasoning or content."

    justification = f"Semantic Similarity: {similarity:.2f}, Reasoning: {'✓' if has_reasoning_structure(user_answer) else '✗'}, Keyword Match: {'✓' if contains_keywords(user_answer, question_context) else '✗'}"

    return feedback, justification
