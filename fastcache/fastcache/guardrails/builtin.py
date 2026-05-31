from fastcache.guardrails.base import BaseGuardrail

class BuiltinGuardrail(BaseGuardrail):
    def __init__(self, min_length: int = 3, max_length: int = 32_000, block_injections: bool = True):
        self.min_length = min_length
        self.max_length = max_length
        self.block_injections = block_injections
        
        self._injection_patterns = [
            "ignore previous instructions",
            "ignore all previous instructions",
            "system:",
            "<|im_start|>",
            "you are now",
        ]

    def validate(self, prompt: str) -> tuple[bool, str]:
        if len(prompt) < self.min_length:
            return False, f"Prompt length ({len(prompt)}) is below minimum allowed ({self.min_length})."
            
        if len(prompt) > self.max_length:
            return False, f"Prompt length ({len(prompt)}) exceeds maximum allowed ({self.max_length})."
            
        if self.block_injections:
            prompt_lower = prompt.lower()
            for pattern in self._injection_patterns:
                if pattern in prompt_lower:
                    return False, f"Prompt contains blocked injection pattern: '{pattern}'"
                    
        return True, ""
