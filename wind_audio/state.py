"""Event state machine on top of the stability metrics.

States::

    initializing -> stable -> changing/active -> stable (new background)

Design rules (business definition, not fitted to any recording):

* A clear change beyond the local noise tolerance triggers an event; a change
  that is *stable* no longer does.
* An event ends only after the spectrum has stayed stable for ``confirm_s``.
  The new stable state is accepted as the new normal regardless of its shape,
  band position or loudness, so no comparison with the pre-event background is
  used for release.
* Offline analysis backdates the end to the start of the confirmed stable run,
  so the confirmation delay is not counted as abnormal time.
* Gaps shorter than the stable window cannot confirm stability, so bursts of
  short events stay in one event automatically -- no fixed maximum duration and
  no post-hoc merging across a recovered stable interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from .stability import StabilityFrames

STATE_INIT = 0
STATE_STABLE = 1
STATE_ACTIVE = 2
STATE_LABELS = {STATE_INIT: "initializing", STATE_STABLE: "stable", STATE_ACTIVE: "active"}


@dataclass(frozen=True)
class EventConfig:
    """Trigger, confirmation and boundary settings.

    * noise tolerance / sensitivity: ``trigger_change_z``, ``trigger_exceed``
    * observation scale: ``trigger_frames``, ``slow_frames``
    * stable confirmation: ``confirm_s``
    """

    trigger_change_z: float = 3.00      # weighted change, in noise sigma
    trigger_exceed: float = 0.15        # fraction of bands over band_z
    trigger_frames: int = 3             # consecutive frames for a fast trigger
    strong_factor: float = 1.5          # this x threshold may trigger in 1 frame
    wide_factor: float = 0.80           # lower magnitude accepted when the change
    wide_exceed_factor: float = 2.0     # ... covers this many bands at once
    slow_ratio_trigger: float = 2.50    # slow-ramp trigger, tolerance units
    slow_frames: int = 10               # consecutive frames of slow-ramp evidence
    confirm_s: float = 0.80             # settled time required to close an event
    confirm_ratio: float = 0.90         # share of that window that must be settled
    hold_s: float = 0.20                # no trigger evidence at all for this long
    slow_min_duration: float = 0.50     # a slow-ramp event must last at least this long
    release_mode: str = "settled"       # "settled" = stable|quiet (transition),
                                        # "stable" = wait for the stability criterion
    segment_mode: str = "trigger"       # "trigger" = change-driven events,
                                        # "state"  = events are the non-stable runs
    state_enter_s: float = 0.20         # unstable time needed to start a state event
    output_mode: str = "anomaly"        # "anomaly" = report non-stable windows,
                                        # "stable"  = report stable segments and
                                        #             derive anomalies as their complement
    stable_enter_s: float = 0.40        # stable time needed to open a stable segment
    stable_exit_s: float = 0.60         # unstable time needed to close it
    quiet_change_ratio: float = 0.50    # "nothing is changing" = this x trigger
    quiet_exceed_ratio: float = 0.50
    quiet_excess: float = 0.20          # and the level is back near the background
    min_duration: float = 0.06
    padding: float = 0.00
    init_s: float = 0.50
    backdate_s: float = 0.60

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, str):
                continue
            if not np.isfinite(value):
                raise ValueError(f"{key} 必须是有限数值")
        if not 0.5 <= self.trigger_change_z <= 20:
            raise ValueError("trigger_change_z 需在 0.5–20 个噪声单位")
        if not 0 < self.trigger_exceed < 1:
            raise ValueError("trigger_exceed 需在 (0,1)")
        if not 1 <= self.trigger_frames <= 500 or not 1 <= self.slow_frames <= 5000:
            raise ValueError("trigger_frames / slow_frames 需为正整数")
        if not 1.0 <= self.strong_factor <= 10:
            raise ValueError("strong_factor 需在 1–10")
        if not 0.3 <= self.wide_factor <= 2 or not 1.0 <= self.wide_exceed_factor <= 6:
            raise ValueError("wide_factor 需在 0.3–2，wide_exceed_factor 需在 1–6")
        if not 0.5 <= self.slow_ratio_trigger <= 20:
            raise ValueError("slow_ratio_trigger 需在 0.5–20")
        if not 0.1 <= self.confirm_s <= 10:
            raise ValueError("confirm_s 需在 0.1–10 秒")
        if not 0.5 <= self.confirm_ratio <= 1:
            raise ValueError("confirm_ratio 需在 0.5–1")
        if not 0 <= self.hold_s <= 5:
            raise ValueError("hold_s 需在 0–5 秒")
        if not 0 < self.slow_min_duration <= 10:
            raise ValueError("slow_min_duration 需在 (0,10] 秒")
        if self.release_mode not in {"settled", "stable"}:
            raise ValueError("release_mode 只支持 settled / stable")
        if self.segment_mode not in {"trigger", "state"}:
            raise ValueError("segment_mode 只支持 trigger / state")
        if not 0 <= self.state_enter_s <= 5:
            raise ValueError("state_enter_s 需在 0–5 秒")
        if self.output_mode not in {"anomaly", "stable"}:
            raise ValueError("output_mode 只支持 anomaly / stable")
        if not 0 <= self.stable_enter_s <= 5 or not 0 <= self.stable_exit_s <= 5:
            raise ValueError("stable_enter_s / stable_exit_s 需在 0–5 秒")
        if not 0.05 <= self.quiet_change_ratio <= 1 or not 0.05 <= self.quiet_exceed_ratio <= 1:
            raise ValueError("quiet_change_ratio / quiet_exceed_ratio 需在 0.05–1")
        if not 0.01 <= self.quiet_excess <= 0.9:
            raise ValueError("quiet_excess 需在 0.01–0.9")
        if not 0 <= self.min_duration <= 10 or not 0 <= self.padding <= 2:
            raise ValueError("min_duration / padding 需为非负秒数")
        if not 0 <= self.init_s <= 10:
            raise ValueError("init_s 需在 0–10 秒")
        if not 0 <= self.backdate_s <= 2:
            raise ValueError("backdate_s 需在 0–2 秒")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "EventConfig":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"不支持的事件参数 {sorted(unknown)}")
        return cls(**data)


def _sustained(mask: np.ndarray, frames: int) -> np.ndarray:
    """True where ``mask`` has been true for at least ``frames`` consecutive frames."""
    if frames <= 1 or len(mask) == 0:
        return mask.copy()
    run = np.zeros(len(mask), dtype=np.int64)
    count = 0
    for i, flag in enumerate(mask):
        count = count + 1 if flag else 0
        run[i] = count
    return run >= frames


def trigger_mask(frames: StabilityFrames, cfg: EventConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (fast, slow) trigger masks before the consecutive-frame filter."""
    fast = (frames.change_z >= cfg.trigger_change_z) & (frames.exceed >= cfg.trigger_exceed)
    fast_sustained = _sustained(fast, cfg.trigger_frames)
    # A very short transient may only last one frame: accept it when it is
    # either clearly larger than the trigger, or spreads over many more bands.
    strong = (frames.change_z >= cfg.trigger_change_z * cfg.strong_factor) & fast
    wide = ((frames.change_z >= cfg.trigger_change_z * cfg.wide_factor)
            & (frames.exceed >= cfg.trigger_exceed * cfg.wide_exceed_factor))
    fast = (fast_sustained | strong | wide) & frames.valid
    slow = _sustained(frames.slow_ratio >= cfg.slow_ratio_trigger, cfg.slow_frames) & frames.valid
    return fast, slow


def track_events(frames: StabilityFrames, cfg: EventConfig, duration_s: float,
                 ) -> tuple[list[dict], dict]:
    """Run the state machine and return (events, traces)."""
    times = frames.times
    n = len(times)
    step = float(np.median(np.diff(times))) if n > 1 else 0.02
    init_frames = int(round(cfg.init_s / step))
    confirm_frames = max(1, int(round(cfg.confirm_s / step)))
    hold_frames = max(0, int(round(cfg.hold_s / step)))
    backdate_frames = max(0, int(round(cfg.backdate_s / step)))
    context_frames = init_frames + confirm_frames + 1

    state = np.full(n, STATE_STABLE, dtype=np.int8)
    state[: min(init_frames, n)] = STATE_INIT
    hot_fast, hot_slow = trigger_mask(frames, cfg)
    hot = hot_fast | hot_slow
    state_mode = cfg.segment_mode == "state"
    if state_mode:
        # In state mode the event *is* the run where the learned stability
        # criterion fails; no change trigger is required.
        enter_frames = max(1, int(round(cfg.state_enter_s / step)))
        hot = _sustained((~frames.stable) & frames.valid, enter_frames)
    # Two ways to be "settled": either the spectrum is formally stable, or
    # nothing anywhere near the trigger level is happening.  The second path is
    # what releases events on recordings whose background never meets the strict
    # stability test (continuous wind noise), without ever comparing the new
    # state against the pre-event background.
    quiet = ((frames.change_z < cfg.trigger_change_z * cfg.quiet_change_ratio)
             & (frames.exceed < cfg.trigger_exceed * cfg.quiet_exceed_ratio)
             & (frames.excess_ratio < cfg.quiet_excess)
             & frames.valid)
    # A frame that is triggering cannot also count as "settled": a slow ramp
    # has a low fast-change score while still being a change.
    base = frames.stable if (cfg.release_mode == "stable" or cfg.segment_mode == "state") \
        else (frames.stable | quiet)
    settled = base & ~hot

    traces = {
        "state": state,
        "settled": settled,
        "active": np.zeros(n, dtype=bool),
        "rejected_triggers": [],
        "context_insufficient": bool(n < context_frames),
        "context_required_s": round(context_frames * step, 3),
        "available_s": round(n * step, 3),
        "trigger_frames": hot,
        "slow_trigger_frames": hot_slow,
    }
    if n < context_frames:
        return [], traces

    def backdate(index: int) -> int:
        """Walk back to the last settled frame before the trigger.

        The trigger fires only after the change has become unambiguous, so the
        real onset is earlier.  Because ``quiet`` includes the level-excess
        test, the walk stops at the last frame that was still normal and does
        not depend on the pre-event background level.
        """
        limit = max(init_frames, index - backdate_frames)
        k = index
        while k > limit and not quiet[k - 1]:
            k -= 1
        return k

    events: list[dict] = []
    i = init_frames
    while i < n:
        if not hot[i]:
            i += 1
            continue
        onset = (max(0, i - enter_frames + 1) if state_mode else backdate(i))
        end = None
        reason = "end_of_file"
        stable_count = 0
        last_hot = i
        j = i
        while j < n:
            stable_count += int(settled[j])
            if hot[j]:
                last_hot = j
            if j - i + 1 >= confirm_frames:
                # A few unstable frames must not restart the confirmation, but
                # (a) the window as a whole has to be settled and (b) no trigger
                # evidence at all may remain in the last hold_s seconds.
                if stable_count < confirm_frames:
                    stable_count -= int(settled[j - confirm_frames + 1])
                if (j - onset + 1 >= confirm_frames
                        and stable_count >= confirm_frames * cfg.confirm_ratio
                        and j - last_hot >= hold_frames):
                    end = j - confirm_frames + 1
                    while end < j and not settled[end]:
                        end += 1
                    # The event can never end before its last trigger evidence.
                    end = max(end, min(last_hot + 1, n - 1))
                    reason = "stable_confirmed"
                    break
            j += 1
        censored = end is None
        if end is None:
            end = n - 1
        end = max(end, onset)
        state[onset:end + 1] = STATE_ACTIVE
        start_s = max(0.0, float(times[onset] - step / 2))
        end_s = float(duration_s) if censored else min(float(duration_s), float(times[end] + step / 2))
        if cfg.padding:
            start_s = max(0.0, start_s - cfg.padding)
            end_s = min(float(duration_s), end_s + cfg.padding)
        event_duration = end_s - start_s
        triggers = []
        for k in range(onset, end + 1):
            if hot[k] and (k == onset or not hot[k - 1]):
                stop = k
                while stop + 1 <= end and hot[stop + 1]:
                    stop += 1
                triggers.append({"start_s": max(0.0, float(times[k] - step / 2)),
                                 "end_s": min(float(duration_s), float(times[stop] + step / 2))})
        slow_only = not hot_fast[onset:end + 1].any()
        if event_duration < cfg.min_duration or (slow_only and event_duration < cfg.slow_min_duration):
            traces["rejected_triggers"].append({
                "start_s": float(start_s), "end_s": float(end_s),
                "reason": "slow_too_short" if slow_only else "too_short",
                "duration_s": float(event_duration),
                "peak_change_z": float(frames.change_z[onset:end + 1].max()),
            })
            i = max(end, i + 1)
            continue
        events.append({
            "start_s": float(start_s),
            "end_s": float(end_s),
            "duration_s": float(event_duration),
            "i0": int(onset),
            "i1": int(end),
            "boundary_reason": reason,
            "right_censored": bool(censored),
            "at_edge": bool(onset <= 0 or censored),
            "peak_change_z": float(frames.change_z[onset:end + 1].max()),
            "mean_change_z": float(frames.change_z[onset:end + 1].mean()),
            "peak_exceed": float(frames.exceed[onset:end + 1].max()),
            "peak_slow_ratio": float(frames.slow_ratio[onset:end + 1].max()),
            "peak_time_s": float(times[onset + int(np.argmax(frames.change_z[onset:end + 1]))]),
            "triggers": triggers,
            "trigger_start_s": triggers[0]["start_s"] if triggers else float(start_s),
            "trigger_end_s": triggers[-1]["end_s"] if triggers else float(end_s),
            "stable_confirmed_s": float(confirm_frames * step) if not censored else 0.0,
        })
        i = max(end, i + 1)
    traces["active"] = state == STATE_ACTIVE
    traces["near_miss"] = _near_misses(frames, cfg, traces["active"], duration_s)
    return events, traces


def _near_misses(frames: StabilityFrames, cfg: EventConfig, active: np.ndarray,
                 duration_s: float, limit: int = 200) -> list[dict]:
    """Changes that came close to the trigger but were not admitted.

    Recorded so a missed event can be traced to "below trigger" instead of being
    invisible.  They never become events and never modify the audio.
    """
    times = frames.times
    n = len(times)
    step = float(np.median(np.diff(times))) if n > 1 else 0.02
    near = ((frames.change_z >= cfg.trigger_change_z * 0.7) & frames.valid
            & ~active & ~frames.stable)
    out: list[dict] = []
    i = 0
    while i < n and len(out) < limit:
        if not near[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and near[j + 1]:
            j += 1
        if j - i + 1 >= max(2, cfg.trigger_frames):
            peak = float(frames.change_z[i:j + 1].max())
            exceed = float(frames.exceed[i:j + 1].max())
            reason = ("below_exceed" if peak >= cfg.trigger_change_z else "below_trigger")
            out.append({"start_s": max(0.0, float(times[i] - step / 2)),
                        "end_s": min(float(duration_s), float(times[j] + step / 2)),
                        "reason": reason, "peak_change_z": peak,
                        "peak_exceed": exceed,
                        "trigger_change_z": cfg.trigger_change_z,
                        "trigger_exceed": cfg.trigger_exceed})
        i = j + 1
    return out


def track_stable_segments(frames, cfg: EventConfig, duration_s: float) -> list[dict]:
    """Stable segments with hysteresis, i.e. the primary output in stable mode.

    A segment opens after ``stable_enter_s`` of continuous stability and closes
    after ``stable_exit_s`` of continuous instability, so single-frame flicker
    neither starts nor ends a segment.
    """
    times = frames.times
    n = len(times)
    if n == 0:
        return []
    step = float(np.median(np.diff(times))) if n > 1 else 0.02
    enter = max(1, int(round(cfg.stable_enter_s / step)))
    exit_ = max(1, int(round(cfg.stable_exit_s / step)))
    core = frames.stable & frames.valid
    segments: list[list[int]] = []
    stable_run = unstable_run = 0
    start = 0
    inside = False
    for t in range(n):
        if core[t]:
            stable_run += 1
            unstable_run = 0
            if not inside and stable_run >= enter:
                inside = True
                start = t - stable_run + 1
        else:
            unstable_run += 1
            stable_run = 0
            if inside and unstable_run >= exit_:
                segments.append([start, t - unstable_run])
                inside = False
    if inside:
        segments.append([start, n - 1])
    out = []
    for a, b in segments:
        start_s = max(0.0, float(times[a] - step / 2))
        end_s = min(float(duration_s), float(times[b] + step / 2))
        if end_s - start_s < cfg.min_duration:
            continue
        out.append({"start_s": start_s, "end_s": end_s,
                    "duration_s": end_s - start_s,
                    "coverage": (end_s - start_s) / max(duration_s, 1e-9)})
    return out


def complement_events(segments: list[dict], duration_s: float, cfg: EventConfig) -> list[dict]:
    """Anomaly = whole recording minus the stable segments."""
    merged: list[list[float]] = []
    for seg in sorted(segments, key=lambda s: s["start_s"]):
        a, b = float(seg["start_s"]), float(seg["end_s"])
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps: list[list[float]] = []
    prev = 0.0
    for a, b in merged:
        if a - prev > cfg.min_duration + 1e-9:
            gaps.append([prev, a])
        prev = max(prev, b)
    if duration_s - prev > cfg.min_duration + 1e-9:
        gaps.append([prev, float(duration_s)])
    return [{"start_s": max(0.0, a), "end_s": min(float(duration_s), b),
             "duration_s": min(float(duration_s), b) - max(0.0, a)}
            for a, b in gaps]
