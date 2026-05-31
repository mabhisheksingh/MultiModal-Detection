import re
from typing import Tuple
from triton_server.app import settings


class OCRUtils:
    PLATE_PATTERN = re.compile(settings.ocr_cleanup_regex)

    _PLATE_ARTIFACT_TO_ALNUM = {
        "(": "C",
        "{": "C",
        "$": "S",
        "|": "I",
        "!": "1",
        "@": "A",
        "#": "H",
        '"': "",
        "\u201c": "",
        "\u201d": "",
        "'": "",
        "\u2018": "",
        "\u2019": "",
    }

    @staticmethod
    def parse_result(ocr_result) -> Tuple[str, float]:
        if not isinstance(ocr_result, list):
            return (ocr_result or "").strip(), 0.0
        text_parts, confidences = [], []
        for item in ocr_result:
            text_value = item.get("text", "").strip()
            if text_value:
                text_parts.append(text_value)
            conf_value = item.get("confidence")
            if conf_value is not None:
                try:
                    confidences.append(float(conf_value))
                except (TypeError, ValueError):
                    continue
        text = " ".join(text_parts).strip()
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return text, avg_conf

    @classmethod
    def normalize_plate_text(cls, text: str) -> str:
        out = []
        for ch in str(text or "").upper():
            mapped = cls._PLATE_ARTIFACT_TO_ALNUM.get(ch)
            if mapped is not None:
                if mapped:
                    out.append(mapped)
            elif ch.isalnum():
                out.append(ch)
        return "".join(out)

    @classmethod
    def validate_plate_text(cls, normalized_text: str, mode: str = "strict") -> bool:
        """
        Validate normalized plate text based on mode.
        
        Args:
            normalized_text: Already normalized plate text
            mode: 'strict' for regex validation, 'balanced' for no validation
            
        Returns:
            True if valid, False otherwise
        """
        if not normalized_text:
            return False
        
        if mode.lower() == "strict":
            return bool(cls.PLATE_PATTERN.fullmatch(normalized_text))
        
        return True
