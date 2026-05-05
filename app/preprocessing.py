import re
import string

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "has", "have", "he", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "to", "was", "were", "will", "with", "this", "these", "those",
    "their", "they", "them", "you", "your", "we", "our", "but", "if", "than",
    "then", "there", "here", "after", "before", "into", "about", "over", "under",
    "also", "can", "could", "should", "would", "may", "might", "do", "does", "did",
}


def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    filtered_tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    return " ".join(filtered_tokens)
