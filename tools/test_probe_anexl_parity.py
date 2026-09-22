"""Pure input/output contract checks; no model load, CoreML, MLX or network."""
import unittest

import numpy as np

from probe_anexl_parity import build_inputs, compare, softmax, validate_outputs


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.item = {"ids": [1, 99, 5, 99, 2], "markers": [1, 3], "qtype": 0}

    def test_padding_and_marker_orientation(self):
        inputs = build_inputs(self.item, 128, 7, 99)
        self.assertEqual(inputs["input_ids"].dtype, np.int32)
        self.assertEqual(inputs["attention_mask"].dtype, np.int32)
        self.assertEqual(inputs["marker_map"].dtype, np.float32)
        self.assertEqual(inputs["marker_map"].shape, (1, 32, 128))
        np.testing.assert_array_equal(inputs["input_ids"][0, :5], self.item["ids"])
        self.assertTrue(np.all(inputs["input_ids"][0, 5:] == 7))
        np.testing.assert_array_equal(inputs["attention_mask"][0, :6], [1, 1, 1, 1, 1, 0])
        self.assertEqual(inputs["attention_mask"].sum(), 5)
        self.assertEqual(inputs["marker_map"].sum(), 2)
        self.assertEqual(inputs["marker_map"][0, 0, 1], 1)
        self.assertEqual(inputs["marker_map"][0, 1, 3], 1)
        self.assertFalse(inputs["marker_map"][0, 2:].any())

    def test_all_question_type_onehots(self):
        for qtype in range(3):
            inputs = build_inputs({**self.item, "qtype": qtype}, 128, 0, 99)
            np.testing.assert_array_equal(inputs["question_type"], np.eye(3, dtype=np.float32)[qtype:qtype+1])

    def test_exact_bucket_and_max_options_fit(self):
        for length in (128, 256, 512, 1024):
            item = {"ids": [99] * length, "markers": list(range(32)), "qtype": 2}
            inputs = build_inputs(item, length, 0, 99)
            self.assertEqual(inputs["attention_mask"].sum(), length)
            self.assertEqual(inputs["marker_map"].sum(), 32)

    def test_rejects_overflow_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            build_inputs({**self.item, "ids": [99] * 129}, 128, 0, 99)

    def test_invalid_inputs_fail_closed(self):
        for change in ({"ids": []}, {"ids": [2**31, 99, 5, 99, 2]},
                       {"ids": [-1, 99, 5, 99, 2]}, {"markers": []},
                       {"markers": [1, 1]}, {"markers": list(range(33))},
                       {"markers": [0]}, {"markers": [-1]}, {"markers": [5]}, {"qtype": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_inputs({**self.item, **change}, 128, 0, 99)


class OutputContractTests(unittest.TestCase):
    def outputs(self):
        return {"logits": np.zeros((1, 32), dtype=np.float32),
                "probabilities": np.full((1, 32), 1 / 32, dtype=np.float32),
                "action_probabilities": np.array([[.25, .75]], dtype=np.float32)}

    def test_valid_outputs(self):
        validate_outputs(self.outputs())

    def test_rejects_bad_shape_dtype_names_and_probabilities(self):
        for key, value in (("logits", np.zeros((32,), dtype=np.float32)),
                           ("logits", np.zeros((1, 32), dtype=np.float16)),
                           ("logits", np.full((1, 32), np.nan, dtype=np.float32)),
                           ("action_probabilities", np.array([[1, 1]], dtype=np.float32)),
                           ("action_probabilities", np.array([[-1, 2]], dtype=np.float32))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_outputs({**self.outputs(), key: value})
        with self.assertRaises(ValueError):
            validate_outputs({"logits": self.outputs()["logits"]})

    def test_softmax_is_stable_and_calibrated(self):
        np.testing.assert_allclose(softmax([10000, 10002], 2), softmax([0, 1]))
        np.testing.assert_allclose(softmax([0, 1], 1e-5), softmax([0, 1], 1e-3))
        for values, temperature in (([], 1), ([np.inf], 1), ([1], 0), ([1], np.nan)):
            with self.assertRaises(ValueError):
                softmax(values, temperature)

    def test_comparison_uses_only_real_option_slots(self):
        result = compare(np.array([.5, .5]), np.array([.25, .75]),
                         np.array([[0., 0., 100.]]), np.log([[.25, .75]]), 1)
        self.assertEqual(result["option_max_abs_diff"], 0)
        self.assertAlmostEqual(result["action_max_abs_diff"], 0)
        self.assertTrue(result["option_argmax_agreement"])
        self.assertTrue(result["action_argmax_agreement"])


if __name__ == "__main__":
    unittest.main()
