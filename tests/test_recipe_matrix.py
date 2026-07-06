import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import recipe_matrix  # noqa: E402


class ParseTests(unittest.TestCase):
    def test_parse_value(self):
        self.assertEqual(recipe_matrix.parse_value("0.42"), 0.42)
        self.assertEqual(recipe_matrix.parse_value("4"), 4)
        self.assertIs(recipe_matrix.parse_value("true"), True)
        self.assertIs(recipe_matrix.parse_value("False"), False)
        self.assertEqual(recipe_matrix.parse_value("linear_quadratic"),
                         "linear_quadratic")

    def test_parse_set(self):
        title, input_name, value = recipe_matrix.parse_set(
            "BasicScheduler:denoise=0.42")
        self.assertEqual((title, input_name, value),
                         ("BasicScheduler", "denoise", 0.42))

    def test_parse_set_title_with_colon(self):
        title, input_name, value = recipe_matrix.parse_set(
            "LoRA: distilled:strength_model=0.5")
        self.assertEqual(title, "LoRA: distilled")
        self.assertEqual(input_name, "strength_model")
        self.assertEqual(value, 0.5)

    def test_parse_axis(self):
        name, title, input_name, values = recipe_matrix.parse_axis(
            "denoise=BasicScheduler:denoise=0.3,0.42,0.55")
        self.assertEqual(name, "denoise")
        self.assertEqual(title, "BasicScheduler")
        self.assertEqual(input_name, "denoise")
        self.assertEqual(values, [0.3, 0.42, 0.55])

    def test_invalid_specs_raise(self):
        for spec in ("noequals", "title=novalue", "=x:y=1"):
            with self.assertRaises(ValueError):
                recipe_matrix.parse_axis(spec)
        with self.assertRaises(ValueError):
            recipe_matrix.parse_set("no-equals-here")


class OverrideTests(unittest.TestCase):
    def workflow(self):
        return {
            "10": {
                "class_type": "BasicScheduler",
                "inputs": {"scheduler": "normal", "steps": 4, "denoise": 1.0},
                "_meta": {"title": "BasicScheduler"},
            }
        }

    def test_apply_override(self):
        wf = self.workflow()
        recipe_matrix.apply_overrides(
            wf, [("BasicScheduler", "denoise", 0.42)])
        self.assertEqual(wf["10"]["inputs"]["denoise"], 0.42)

    def test_unknown_title_raises(self):
        with self.assertRaises(ValueError):
            recipe_matrix.apply_overrides(
                self.workflow(), [("Nope", "denoise", 0.42)])

    def test_unknown_input_raises(self):
        with self.assertRaises(ValueError):
            recipe_matrix.apply_overrides(
                self.workflow(), [("BasicScheduler", "nope", 0.42)])


class LabelTests(unittest.TestCase):
    def test_combo_label(self):
        self.assertEqual(
            recipe_matrix.combo_label(["denoise", "lora"], [0.42, 0.5]),
            "denoise-0.42__lora-0.5",
        )
        self.assertEqual(recipe_matrix.combo_label([], []), "single")

    def test_label_sanitizes(self):
        self.assertEqual(
            recipe_matrix.combo_label(["s"], ["linear quadratic/x"]),
            "s-linear_quadratic_x",
        )


if __name__ == "__main__":
    unittest.main()
