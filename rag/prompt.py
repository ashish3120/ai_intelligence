from langchain_core.prompts import PromptTemplate

# PROMPT TEMPLATES

STANDARD_TEMPLATE = """You are a helpful personal knowledge assistant.
Your task is to answer the QUESTION based ONLY on the provided CONTEXT.

RULES:
1. Use ONLY the information in the CONTEXT below.
2. If the answer is not in the context, say "I cannot find the answer in the provided documents."
3. Do NOT make up information.
4. Be concise and direct.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

SUMMARIZE_TEMPLATE = """You are an expert summarizer.
Your task is to provide a comprehensive summary of the validation CONTEXT related to the QUESTION.
Focus on key points, timelines, and outcomes.

CONTEXT:
{context}

TOPIC TO SUMMARIZE:
{question}

SUMMARY:
"""

ELI5_TEMPLATE = """You are a helpful teacher explaining things to a 5-year-old.
Use simple words and analogies. Explain the CONTEXT clearly.

CONTEXT:
{context}

QUESTION:
{question}

EXPLANATION:
"""

EXAM_TEMPLATE = """You are a strict examiner.
Based on the CONTEXT, generate 3 validation quiz questions (with potential answers) that test knowledge of the content.

CONTEXT:
{context}

TOPIC:
{question}

QUIZ:
"""

PROMPTS = {
    "standard": STANDARD_TEMPLATE,
    "summarize": SUMMARIZE_TEMPLATE,
    "explain_simple": ELI5_TEMPLATE,
    "exam": EXAM_TEMPLATE
}

def get_prompt_template(mode: str = "standard") -> PromptTemplate:
    """Returns the prompt template for the RAG chain."""
    template_str = PROMPTS.get(mode, STANDARD_TEMPLATE)
    return PromptTemplate(
        template=template_str,
        input_variables=["context", "question"]
    )
