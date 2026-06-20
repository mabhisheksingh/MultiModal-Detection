import re

from triton_server.app import settings


def test_ocr_cleanup_regex_accepts_valid_formats():
    valid_plates = [
        # Private vehicle
        "UP70AB1234",
        # Commercial vehicle
        "MH01CD5678",
        # Commercial rental (self-drive)
        "GA06R7890",
        # Private EV (green plates often include EV code)
        "KA03EV9012",
        # Commercial EV (yellow with EC code)
        "DL1CEC3456",
        # General multi-letter series variants
        "DL3CAY9324",
        "DL32ABC3456",
        "UP70H1234",
        "DL2AB1234",
        # Bharat series
        "21BH1234AA",
        "23BH5678AB",
        # Diplomatic plates (CD/CC/UN)
        "77CD12",
        "11CC123",
        "09UN7",
        # Temporary registration
        "UP32TR1456A",
    ]

    pattern = re.compile(settings.ocr_cleanup_regex)

    assert all(pattern.fullmatch(plate) for plate in valid_plates)


def test_ocr_cleanup_regex_rejects_invalid_formats():
    invalid_plates = [
        "BH1234AA",  # missing state/district prefix
        "DL3CA932",  # too few digits at the end
        "UP700H1234",  # extra digit in district code
        "22BH12345A",  # incorrect Bharat series suffix
        "77CD",  # diplomatic with no numeric identifier
        "DL1CEC34567",  # too many digits at end
        "KA03EV901",  # too few digits at end
        "7703D153874H",  # military with arrow stripped loses structure
    ]

    pattern = re.compile(settings.ocr_cleanup_regex)

    assert not any(pattern.fullmatch(plate) for plate in invalid_plates)
