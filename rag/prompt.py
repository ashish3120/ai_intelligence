from langchain_core.prompts import PromptTemplate

# Explicit system instruction to use provided context only and avoid hallucination.
SYSTEM_PROMPT_TEMPLATE = """You are a helpful personal knowledge assistant.
Your task is to answer the QUESTION based ONLY on the provided CONTEXT.

RULES:
1. Use ONLY the information in the CONTEXT below.
2. If the answer is not in the context, say "I cannot find the answer in the provided documents."
3. Do NOT make up information.
4. Do NOT use outside knowledge unless it helps explain the context (but do not add new facts).
5. Be concise and direct.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def get_prompt_template() -> PromptTemplate:
    """Returns the prompt template for the RAG chain."""
    return PromptTemplate(
        template=SYSTEM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
