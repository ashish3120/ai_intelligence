from typing import List, Tuple

def calculate_confidence(docs_with_score: List[Tuple[object, float]]) -> dict:
    """
    Calculates detailed confidence metrics.
    
    Returns a dict with:
    - label: "High", "Medium", "Low", "No Answer"
    - score: 0-100 float
    - explanation: str
    - safe_to_answer: bool (True if we should proceed with generation)
    - details: dict (raw stats)
    """
    if not docs_with_score:
        return {
            "label": "No Answer",
            "score": 0.0,
            "explanation": "No matching documents found.",
            "safe_to_answer": False,
            "details": {}
        }

    scores = [score for doc, score in docs_with_score]
    # FAISS L2 Distance: Lower is better.
    # 0.0 = exact match. 
    # > 1.0 = poor match.
    
    top_1_score = min(scores)
    avg_score = sum(scores) / len(scores)
    
    # Heuristics for L2 Distance
    # Thresholds need tuning based on embeddings (MiniLM-L6-v2)
    SAFE_THRESHOLD = 1.1  # If top match is > 1.1 distance, it's likely irrelevant
    HIGH_CONFIDENCE_THRESHOLD = 0.5
    
    safe_to_answer = top_1_score < SAFE_THRESHOLD
    
    # Convert distance to a 0-100 score for display
    # This is an approximation
    display_score = max(0.0, 1.2 - avg_score) / 1.2 * 100
    
    label = "Low"
    if not safe_to_answer:
        label = "Unsafe / No Answer"
    elif display_score > 75:
        label = "High"
    elif display_score > 50:
        label = "Medium"
        
    explanation = (
        f"Top match distance: {top_1_score:.4f} (Threshold: {SAFE_THRESHOLD}). "
        f"Avg distance: {avg_score:.4f} across {len(scores)} chunks."
    )
    
    return {
        "label": label,
        "score": display_score,
        "explanation": explanation,
        "safe_to_answer": safe_to_answer,
        "details": {
            "top_1": top_1_score,
            "average": avg_score,
            "count": len(scores)
        }
    }
