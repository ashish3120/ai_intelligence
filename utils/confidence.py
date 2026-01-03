from typing import List, Tuple

def calculate_confidence(docs_with_score: List[Tuple[object, float]]) -> Tuple[str, float, str]:
    """
    Calculates a confidence score based on similarity scores and count of retrieved docs.
    
    Args:
        docs_with_score: List of (Document, score) tuples. 
                         Chroma usually returns distance (lower is better) or similarity (higher is better).
                         langchain-chroma default similarity_search_with_score returns L2 distance by default for default collection?
                         Actually, let's assume we use standard similarity which often confusingly returns distance for chroma.
                         However, sentence-transformers usually uses cosine similarity.
                         
                         Let's handle standard Chroma usage: default space is usually L2 (Squared L2) or Cosine Distance.
                         Lower score = more similar.
                         
                         If we assume Cosine Distance: 0.0 is identical, 1.0 is opposite.
                         
    Returns:
        Tuple[str, float, str]: (Label, Score, Explanation)
    """
    if not docs_with_score:
        return "Low", 0.0, "No documents found."

    # Inspect scores to guess interpretation
    scores = [score for doc, score in docs_with_score]
    avg_score = sum(scores) / len(scores)
    
    # Heuristic for Chroma L2/Cosine distance (Lower is better)
    # A generic cutoff for "good" matches in MiniLM/Chroma context:
    # < 0.3 or 0.4 is usually quite relevant.
    # > 1.0 is usually irrelevant.
    
    # Inverting logic for user display: Confidence %
    # Let's approximate: 0.0 distance -> 100% confidence. 1.0 distance -> 0%.
    # This is a rough heuristic.
    
    confidence_value = max(0.0, 1.0 - avg_score) * 100
    
    label = "Low"
    if confidence_value > 70:
        label = "High"
    elif confidence_value > 40:
        label = "Medium"
        
    explanation = (
        f"Based on {len(docs_with_score)} sources with average distance score of {avg_score:.4f}. "
        f"(Lower distance is better)"
    )
    
    return label, confidence_value, explanation
