"""Regressions for full events and weak-noise rejection, not just edge peaks."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np

from tests.test_model import CFG, wave
from wind_audio.features import extract_features
from wind_audio.detectors import SpectralChangeDetector
from wind_audio import cli
from wind_audio.audio import write_wav


def run(x, **postprocess):
    bundle = extract_features(x, CFG.sr, CFG, "extent")
    return SpectralChangeDetector(CFG, postprocess=postprocess).analyze(bundle)


class EventWindowTests(unittest.TestCase):
    def test_config_and_cli_controls_match(self):
        from wind_audio.detectors import load_detector
        self.assertEqual(load_detector(Path(__file__).parents[1] / "config/l1.json").to_dict(),
                         SpectralChangeDetector().to_dict())
        args = cli.build_parser().parse_args(["run", "--input", "unused", "--relative-floor-db", "12",
                                             "--min-prominence", ".4", "--quiet-s", ".4"])
        detector = cli._detector_from_args(args)
        self.assertEqual(detector.postprocess["relative_floor_db"], 12)
        self.assertEqual(detector.postprocess["min_prominence"], .4)
        self.assertEqual(detector.postprocess["quiet_s"], .4)
        for pp in ({"quiet_s": 0}, {"release_db": float("nan")}, {"min_prominence": 1}, {"typo": 3}):
            with self.assertRaises(ValueError):
                SpectralChangeDetector(postprocess=pp)

    def test_batch_report_and_long_names_keep_unique_window_files(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            x = wave(7, sr=44100)
            rng = np.random.default_rng(5)
            for a in (2, 4):
                x[a*44100:a*44100+8820] += rng.normal(0, .2, 8820)
            source = root / "in"
            write_wav(source / ("a"*110 + ".WAV"), x, 44100)
            write_wav(source / "b.wav", x, 44100)
            out = root / "out"
            self.assertEqual(cli.main(["run", "--input", str(source), "--out", str(out)]), 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["files_succeeded"], 2)
            self.assertEqual(summary["total_events"], 4)
            self.assertTrue((out / "index.html").exists())
            for item in summary["files"]:
                folder = Path(item["out"])
                payload = json.loads((folder / "events.json").read_text())
                names = [e["window"]["path"] for e in payload["events"]]
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all((folder / n).exists() for n in names))
                self.assertTrue((folder / "activity.png").exists())
                self.assertIn("event_active", (folder / "scores.csv").read_text().splitlines()[0])
                self.assertIn("rejected_candidates", payload["diagnostics"])

    def test_long_broadband_plateau_is_not_split_into_two_edges(self):
        x = wave(8)
        x[16000:40000] += np.random.default_rng(11).normal(0, .2, 24000)
        events, _ = run(x)
        self.assertEqual(len(events), 1)
        self.assertLessEqual(events[0]["start_s"], 2.02)
        self.assertGreaterEqual(events[0]["end_s"], 5)
        self.assertLess(events[0]["end_s"], 5.5)
        self.assertEqual(events[0]["boundary_reason"], "background_return")

    def test_decay_tail_remains_inside_event(self):
        x = wave(7)
        t = np.arange(24000) / CFG.sr
        x[16000:40000] += .3 * np.exp(-t / .5) * np.random.default_rng(13).normal(size=len(t))
        events, _ = run(x)
        self.assertEqual(len(events), 1)
        # At 3.2 s the event still has 27 mV noise, versus 1 mV background.
        self.assertLessEqual(events[0]["start_s"], 2)
        self.assertGreaterEqual(events[0]["end_s"], 3.2)
        self.assertLess(events[0]["end_s"], 5.5)

    def test_brief_quiet_gap_is_one_event_but_long_gap_is_not(self):
        x = wave(8)
        rng = np.random.default_rng(2)
        for a,b in ((1,1.5), (1.65,2.2), (4,4.6)):
            x[int(a*8000):int(b*8000)] += rng.normal(0, .2, int(b*8000)-int(a*8000))
        events, _ = run(x)
        self.assertEqual(len(events), 2)
        self.assertLessEqual(events[0]["start_s"], 1)
        self.assertGreaterEqual(events[0]["end_s"], 2.2)
        self.assertLess(events[0]["end_s"], events[1]["start_s"])

    def test_weak_independent_noise_rejected_but_strong_event_kept(self):
        x = wave(8)
        rng = np.random.default_rng(8)
        x[16000:16400] += rng.normal(0, .03, 400)
        x[32000:34400] += rng.normal(0, .2, 2400)
        events, traces = run(x)
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]["start_s"], 3.5)
        self.assertLessEqual(events[0]["start_s"], 4)
        self.assertGreaterEqual(events[0]["end_s"], 4.3)
        self.assertTrue(traces["rejected_triggers"])

    def test_relative_level_floor_rejects_small_click_on_quiet_background(self):
        x = wave(8) * .01
        rng = np.random.default_rng(4)
        x[16000:16400] += rng.normal(0, .003, 400)
        x[32000:34400] += rng.normal(0, .2, 2400)
        events, traces = run(x)
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]["start_s"], 3.5)
        self.assertTrue(any(r["reason"] == "below_level_floor" for r in traces["rejected_triggers"]))
        relaxed, _ = run(x, relative_floor_db=50)
        self.assertEqual(len(relaxed), 2)

    def test_fixed_gain_preserves_event_windows_and_filtering(self):
        x = wave(8)
        x[16000:28000] += np.random.default_rng(3).normal(0, .2, 12000)
        a, _ = run(x)
        b, _ = run(x * .3)
        np.testing.assert_allclose([[e["start_s"], e["end_s"]] for e in a],
                                   [[e["start_s"], e["end_s"]] for e in b], atol=.04)

    def test_new_tonal_state_closes_without_matching_old_level(self):
        t = np.arange(8*8000) / 8000
        f = np.where(t<2, 500, np.where(t<3, 500+(t-2)*1300, 1800))
        amplitude = np.where(t<2, .03, .2)
        x = amplitude*np.sin(2*np.pi*np.cumsum(f)/8000)
        events, _ = run(x.astype(np.float32))
        self.assertEqual(len(events), 1)
        self.assertGreaterEqual(events[0]["end_s"], 3)
        self.assertLess(events[0]["end_s"], 3.6)
        self.assertEqual(events[0]["boundary_reason"], "new_stable_state")

    def test_event_at_eof_is_kept_and_flagged_open(self):
        x = wave(6)
        x[16000:] += np.random.default_rng(5).normal(0, .2, len(x)-16000)
        events, _ = run(x)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["end_s"], 6)
        self.assertTrue(events[0]["right_censored"])
        self.assertEqual(events[0]["boundary_reason"], "end_of_file")
