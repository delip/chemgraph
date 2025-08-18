"""DSPy-powered agent for ClaimSpy.

This module provides a minimal ClaimSpyAgent that demonstrates how the
ChemGraph agent can be reimagined using the DSPy framework.
"""

from __future__ import annotations

import dspy


class ClaimSpyAgent:
    """Simple ClaimSpy agent that answers questions using DSPy.

    Parameters
    ----------
    model_name:
        Name of the language model backend understood by DSPy.
    **kwargs:
        Additional keyword arguments forwarded to the DSPy OpenAI wrapper.
    """

    class ChatSignature(dspy.Signature):
        """question -> answer"""

        question: str
        answer: str

    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs) -> None:
        self.lm = dspy.OpenAI(model=model_name, **kwargs)
        dspy.settings.configure(lm=self.lm)
        self._program = dspy.Predict(self.ChatSignature)

    def run(self, question: str) -> str:
        """Run the ClaimSpy agent on a question and return its answer."""
        result = self._program(question=question)
        return result.answer
