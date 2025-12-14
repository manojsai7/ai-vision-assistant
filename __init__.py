"""
AI Vision Assistant Package
============================

An AI sidekick that understands images—classifies, detects, and segments—ready to plug into your apps.

Example usage:
    >>> from vision_assistant import VisionAssistant
    >>> assistant = VisionAssistant()
    >>> results = assistant.classify("image.jpg")
"""

from vision_assistant import VisionAssistant, create_assistant

__version__ = "0.1.0"
__all__ = ["VisionAssistant", "create_assistant"]
