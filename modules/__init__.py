"""
modules package
---------------
Each file in this package is one stage of the palm-reading pipeline:

    validation -> preprocessing -> hand_detection -> feature_extraction
    -> knowledge_base -> llm_engine -> chatbot

Keeping stages in separate modules means each one can be tested and
explained on its own, and app.py only orchestrates them.
"""
