from abc import ABC, abstractmethod

class BaseGuardrail(ABC):
    @abstractmethod
    def validate(self, prompt: str) -> tuple[bool, str]:
        """
        Validate the prompt.
        Returns (True, "") if the prompt is acceptable.
        Returns (False, reason) if the prompt should be rejected.
        Raise GuardrailError for hard failures.
        """
        ...
