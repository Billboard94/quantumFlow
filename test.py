import re

def convert_to_float(value):
    """Attempts to convert given value to float."""
    try:
        result = float(value)
    except ValueError:
        result = None
    return result

def _parse_complex(expression):
    complex_number = complex(expression)
    return complex_number

def convert_to_complex(input_str):
    complex_number = _parse_complex(input_str)
    real_part = complex_number.real
    imag_part = complex_number.imag
    sign = 1
    if real_part < 0:
        real_part *= -1
        sign *= -1
    imag_part *= sign
    return complex(real_part, imag_part)

# Test cases
testcases = [
    ("1+3i", "(1+3j)"),
    ("3-4i", "(3-4j)"),
    ("5", "(5)"),
    ("a", "Invalid input 'a' provided."),
    ("1+3ji", "Invalid input '1+3ji' provided."),
    ("1+3I", "Invalid input '1+3I' provided."),
    ("1+3J", "Invalid input '1+3J' provided."),
    ("1+3k", "Invalid input '1+3k' provided."),
    ("1+3l", "Invalid input '1+3l' provided."),
    ("1+3m", "Invalid input '1+3m' provided."),
    ("1+3n", "Invalid input '1+3n' provided."),
    ("1+3o", "Invalid input '1+3o' provided."),
    ("1+3p", "Invalid input '1+3p' provided."),
    ("1+3q", "Invalid input '1+3q' provided."),
    ("1+3r", "Invalid input '1+3r' provided."),
    ("1+3s", "Invalid input '1+3s' provided."),
    ("1+3t", "Invalid input '1+3t' provided."),
    ("1+3u", "Invalid input '1+3u' provided."),
    ("1+3v", "Invalid input '1+3v' provided."),
    ("1+3w", "Invalid input '1+3w' provided."),
    ("1+3x", "Invalid input '1+3x' provided."),
    ("1+3y", "Invalid input '1+3y' provided."),
    ("1+3z", "Invalid input '1+3z' provided."),
    ("1+3A", "Invalid input '1+3A' provided."),
    ("1+3B", "Invalid input '1+3B' provided."),
    ("1+3C", "Invalid input '1+3C' provided."),
    ("1+3D", "Invalid input '1+3D' provided."),
    ("1+3E", "Invalid input '1+3E' provided."),
    ("1+3F", "Invalid input '1+3F' provided."),
    ("1+3G", "Invalid input '1+3G' provided."),
    ("1+3H", "Invalid input '1+3H' provided."),
    ("1+3K", "Invalid input '1+3K' provided."),
    ("1+3L", "Invalid input '1+3L' provided."),
    ("1+3M", "Invalid input '1+3M' provided."),
    ("1+3N", "Invalid input '1+3N' provided."),
    ("1+3O", "Invalid input '1+3O' provided."),
    ("1+3P", "Invalid input '1+3P' provided."),
    ("1+3Q", "Invalid input '1+3Q' provided."),
    ("1+3R", "Invalid input '1+3R' provided."),
    ("1+3S", "Invalid input '1+3S' provided."),
    ("1+3T", "Invalid input '1+3T' provided."),
    ("1+3U", "Invalid input '1+3U' provided."),
    ("1+3V", "Invalid input '1+3V' provided."),
    ("1+3W", "Invalid input '1+3W' provided."),
    ("1+3X", "Invalid input '1+3X' provided."),
    ("1+3Y", "Invalid input '1+3Y' provided."),
    ("1+3Z", "Invalid input '1+3Z' provided."),
]

for tc in testcases:
    try:
        result = convert_to_complex(tc)
        if result != tc:
            raise AssertionError(f"Expected {tc} got {result}")
    except Exception as e:
        if str(e) != tc:
            raise AssertionError(f"Expected {tc} got {repr(e)}")

print("\nAll tests passed!")