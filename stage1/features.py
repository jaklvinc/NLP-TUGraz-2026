"""
Feature factories for Stage 1.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        strip_accents="unicode",
    )
