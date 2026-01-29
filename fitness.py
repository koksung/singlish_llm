def singlish_score(text: str) -> float:
    markers = ["lah", "leh", "lor", "meh", "can lah", "steady"]
    return sum(m in text.lower() for m in markers) / len(markers)

def fluency_score(text: str) -> float:
    # Placeholder: penalize obvious repetition or gibberish
    return 1.0 - min(text.count(".."), 5) / 5.0

def evaluate_outputs(outputs):
    s1 = sum(singlish_score(o) for o in outputs) / len(outputs)
    s2 = sum(fluency_score(o) for o in outputs) / len(outputs)
    return s1, s2
