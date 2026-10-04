import unittest

from cake_bo.kernels import KernelExpressionError, parse_expression


class KernelExpressionTests(unittest.TestCase):
    def test_valid_expressions_are_normalized(self) -> None:
        examples = {
            "SE": "SE",
            "SE + PER": "(SE + PER)",
            "SE+PER*LIN": "(SE + (PER * LIN))",
            "(SE + PER) * M3": "((SE + PER) * M3)",
        }
        for expression, normalized in examples.items():
            with self.subTest(expression=expression):
                self.assertEqual(str(parse_expression(expression)), normalized)

    def test_invalid_expressions_are_rejected(self) -> None:
        invalid = [
            "",
            "UNKNOWN",
            "M1",
            "SE - PER",
            "SE +",
            "(SE + PER",
            "SE; import os",
        ]
        for expression in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(KernelExpressionError):
                    parse_expression(expression)


if __name__ == "__main__":
    unittest.main()
