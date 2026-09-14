import unittest
import numpy as np
from factrisk.eval.seed_stability import distribution, split_rows


class SeedStabilityTests(unittest.TestCase):
    def test_summary_uses_sample_sd_and_all_five(self):
        value = distribution([1, 2, 3, 4, 5])
        self.assertEqual(value['mean'], 3)
        self.assertAlmostEqual(value['sample_sd'], np.sqrt(2.5))
        with self.assertRaises(ValueError): distribution([1, 2])

    def test_split_keeps_unresolved_only_in_test(self):
        rows = [dict(id=f'{split}{label}', split=split, speaker_id=split,
                severe_fact_error=label, fact_type='number')
                for split in ('train','calibration','test') for label in (0,1,None)]
        result = split_rows(rows)
        self.assertEqual([len(result[s]) for s in ('train','calibration','test')], [2,2,3])
        rows[-1]['speaker_id'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Speaker leakage'): split_rows(rows)
