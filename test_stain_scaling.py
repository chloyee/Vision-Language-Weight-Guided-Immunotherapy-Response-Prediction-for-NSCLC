"""Synthetic checks for raw means, training-only scaling and cache guards."""
import unittest
import numpy as np
from stain_scaling import CHANNELS, SCALAR_SCHEMA, fit_scaler, scale_values, scale_case, prepare_stain_scaler


class StainScalingTests(unittest.TestCase):
    def fitted(self):
        return fit_scaler(((c, str(i), x) for c in CHANNELS
                           for i, x in enumerate([1., 3., 5.])), "train-only")

    def test_population_moments_and_deduplication(self):
        records = [(c, str(i), x) for c in CHANNELS
                   for i, x in enumerate([1., 3., 5.])]
        s = fit_scaler(records + records, "train-only")
        for c in CHANNELS:
            self.assertEqual(s["channels"][c]["count"], 3)
            self.assertAlmostEqual(s["channels"][c]["mean"], 3.)
            self.assertAlmostEqual(s["channels"][c]["std"], np.std([1., 3., 5.]))

    def test_unseen_extremes_do_not_refit_or_clip(self):
        s = self.fitted()
        a = scale_values(np.array([0., 100.]), "he_h", s)
        self.assertLess(a[0], 0.)
        self.assertGreater(a[1], 1.)
        self.assertEqual(s["channels"]["he_h"]["mean"], 3.)

    def test_amplitude_information_is_retained(self):
        patch = np.array([1., 2., 3.])
        np.testing.assert_allclose((patch-patch.min())/np.ptp(patch),
                                   (2*patch-2*patch.min())/np.ptp(2*patch))
        a = scale_values(np.array([patch.mean(), (2*patch).mean()]), "he_h", self.fitted())
        self.assertGreater(a[1], a[0])

    def test_scaling_does_not_change_maps_or_raw_cache(self):
        maps = np.ones((2, 32, 32))
        case = dict(stain_scalar_schema=SCALAR_SCHEMA, he_h_map=maps,
                    **{c: np.array([1., 5.]) for c in CHANNELS})
        out = scale_case(case, self.fitted())
        self.assertIs(out["he_h_map"], maps)
        np.testing.assert_array_equal(case["he_h"], [1., 5.])
        self.assertFalse(np.array_equal(out["he_h"], case["he_h"]))

    def test_invalid_inputs_and_legacy_cache_rejected(self):
        for bad in [-1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                fit_scaler([("he_h", "tile", bad)], "train")
        with self.assertRaises(ValueError):
            scale_case({}, self.fitted())

    def test_constant_channel_and_serialization(self):
        import json
        s = fit_scaler([(c, "one", 2.) for c in CHANNELS], "train")
        s = json.loads(json.dumps(s, allow_nan=False))
        for c in CHANNELS:
            self.assertEqual(s["channels"][c]["scale"], 1.)
            np.testing.assert_array_equal(scale_values(np.array([2., 4.]), c, s), [0., 2.])

    def test_fit_pipeline_uses_training_rows_only(self):
        import tempfile
        import json
        import os
        from types import SimpleNamespace
        class Frame:
            empty = False
            def iterrows(self):
                yield 0, dict(project_name="train", case_id="p1", pair_case_name="a")
                yield 1, dict(project_name="train", case_id="p1", pair_case_name="b")
            def sort_values(self, columns):
                return self
        calls = []
        def build(row, args, cache_dir):
            calls.append(row["project_name"])
            return dict(stain_scalar_schema=SCALAR_SCHEMA,
                        he_patch_paths=["/synthetic/he/0_0.png", "/synthetic/he/0_1.png"],
                        ihc_patch_paths=["/synthetic/ihc/0_0.png", "/synthetic/ihc/0_1.png"],
                        **{c: np.array([1., 3.]) for c in CHANNELS})
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(stain_scalar_mode="train_standardized", morphology_mode="macenko", results_dir=d)
            prepare_stain_scaler(args, Frame(), build, d)
            self.assertEqual(calls, ["train", "train"])
            with open(os.path.join(d, "stain_scaler.json")) as f:
                self.assertEqual(json.load(f), args.stain_scaler)
            for c in CHANNELS:
                self.assertEqual(args.stain_scaler["channels"][c]["count"], 2)
                self.assertEqual(args.stain_scaler["channels"][c]["mean"], 2.)


if __name__ == "__main__":
    unittest.main()
