"""Stage 1 knowledge base: opinion-marker lexicon.

Planned shape: smth like a dict[str, list[str]] mapping marker
categories (hedges, modals, subjective adjectives, first-person pronouns, ...)
to wordlists, loaded from a YAML/JSON resource. Used by stage1.features to compute features.
"""
