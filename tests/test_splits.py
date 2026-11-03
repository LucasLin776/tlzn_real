import unittest

from wind_audio.splits import assert_no_leakage, make_splits

from tests.helpers import ann, record


def _dataset():
    recs = [
        record(f"a{i}", f"a{i}.wav", 10.0, source=f"S{i}",
               device="D1" if i < 2 else ("D2" if i < 4 else "D3"))
        for i in range(6)
    ]
    anns = [ann(f"x{i}", f"a{i}", 2.0, 3.0) for i in (0, 1, 4)]
    return recs, anns


class SplitTests(unittest.TestCase):
    def test_no_group_leakage_and_determinism(self):
        recs, anns = _dataset()
        s1 = make_splits(recs, anns, group_by="source", seed=3)
        s2 = make_splits(recs, anns, group_by="source", seed=3)
        self.assertEqual(s1.train, s2.train)
        self.assertEqual(s1.val, s2.val)
        self.assertEqual(s1.test, s2.test)
        assert_no_leakage(s1)
        tr, va, te = set(s1.train), set(s1.val), set(s1.test)
        self.assertFalse(tr & va or tr & te or va & te)

    def test_device_group_keeps_devices_together(self):
        recs, anns = _dataset()
        s = make_splits(recs, anns, group_by="device", seed=1)
        for device, members in {"D1": {"a0", "a1"}, "D2": {"a2", "a3"}, "D3": {"a4", "a5"}}.items():
            sets = [name for name in ("train", "val", "test") if set(members) & set(s.ids(name))]
            self.assertEqual(len(sets), 1, f"{device} split across {sets}")

    def test_too_few_groups_raises(self):
        recs = [record("a", "a.wav", 10.0, source="S1"), record("b", "b.wav", 10.0, source="S1")]
        with self.assertRaises(ValueError):
            make_splits(recs, [], group_by="source")


if __name__ == "__main__":
    unittest.main()
