"""HTML report: waveform, spectrogram, score, annotations, audio, review links."""
from __future__ import annotations

import html
import math
from pathlib import Path


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    if digits == 0:
        return f"{v:.0f}{suffix}"
    return f"{v:.{digits}f}{suffix}"


def _nice_step(span: float, target: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _x_ticks(duration: float) -> list[float]:
    step = _nice_step(duration)
    vals = []
    t = 0.0
    while t <= duration + 1e-6:
        vals.append(round(t, 6))
        t += step
    return vals


def _x_axis(duration: float, left_px: int) -> str:
    ticks = _x_ticks(duration)
    labels = []
    for t in ticks:
        pct = 100.0 * t / duration if duration > 0 else 0.0
        labels.append(
            f'<span style="left:{pct:.3f}%;transform:translateX(-50%)">{_fmt(t, 1 if duration < 20 else 0)}</span>'
        )
    return f'<div class="xaxis" style="margin-left:{left_px}px">{"".join(labels)}</div>'


def _plot_block(
    image: str,
    title: str,
    caption: str,
    duration: float,
    *,
    y_labels: list[tuple[float, str]] | None = None,
    left_px: int = 54,
    height: int = 0,
) -> str:
    y_html = ""
    if y_labels:
        spans = []
        for pct, text in y_labels:
            spans.append(f'<span style="top:{pct:.3f}%;transform:translateY(-50%)">{text}</span>')
        y_html = f'<div class="yaxis" style="width:{left_px}px">{"".join(spans)}</div>'
    else:
        y_html = f'<div class="yaxis" style="width:{left_px}px"></div>'
    return f"""
    <div class="plot-title">{html.escape(title)}</div>
    <div class="plot">
      {y_html}
      <div class="canvas"><img src="{html.escape(image)}" alt="{html.escape(title)}"></div>
    </div>
    {_x_axis(duration, left_px)}
    <div class="cap">{caption}</div>
    """


def _match_label(ev: dict, gt: list[dict]) -> str:
    best = None
    best_iou = 0.0
    for i, g in enumerate(gt, 1):
        inter = max(0.0, min(ev["end_s"], g["end_s"]) - max(ev["start_s"], g["start_s"]))
        union = (ev["end_s"] - ev["start_s"]) + (g["end_s"] - g["start_s"]) - inter
        v = inter / union if union > 0 else 0.0
        if v > best_iou:
            best_iou, best = v, i
    if best is None or best_iou <= 0:
        return "未匹配"
    return f"GT #{best} (IoU {best_iou:.2f})"


def write_report(
    out_dir: str | Path,
    *,
    audio_id: str,
    file_name: str,
    duration: float,
    sr: int,
    analyzed_fmax: float,
    model_version: str,
    detector_type: str,
    score_type: str,
    threshold: float,
    threshold_source: str,
    feature_config: dict,
    postprocess: dict,
    events: list[dict],
    regions: list[dict],
    gt_events: list[dict],
    reviews: dict | None = None,
    audio_relpath: str,
    review_relpath: str | None,
    spectrogram_name: str = "spectrogram.png",
    waveform_name: str = "waveform.png",
    score_name: str = "score.png",
) -> Path:
    out = Path(out_dir)
    event_rows = []
    for i, ev in enumerate(events, 1):
        clip = ev.get("clip") or {}
        player = (
            f'<audio controls preload="none" src="{html.escape(clip["path"])}"></audio>'
            if clip.get("path")
            else "-"
        )
        window_path = (ev.get("window") or {}).get("path", "")
        window_link = f'<a href="{html.escape(window_path)}">检测窗口 WAV</a>' if window_path else "-"
        boundary = {"background_return": "背景恢复", "new_stable_state": "新工况稳定",
                    "end_of_file": "到文件末尾，结束未确认"}.get(ev.get("boundary_reason"), "-")
        event_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{_fmt(ev['start_s'])} – {_fmt(ev['end_s'])}</td>"
            f"<td>{_fmt(ev['duration_s'])}</td>"
            f"<td>{_fmt(ev['peak_score'], 3)}</td>"
            f"<td>{html.escape(boundary)}</td>"
            f"<td>{html.escape(_match_label(ev, gt_events))}</td>"
            f"<td>{html.escape(str((reviews or {}).get(ev.get('event_id'), {}).get('l1_label') or '待审核'))}</td>"
            f"<td>{html.escape(str((reviews or {}).get(ev.get('event_id'), {}).get('l2_label') or 'not_trained'))}</td>"
            f"<td>{window_link}<br>{player}</td>"
            "</tr>"
        )
    event_table = "\n".join(event_rows) if event_rows else '<tr><td colspan="9">未检测到事件</td></tr>'

    region_rows = []
    for r in regions:
        if r["label"] == "uncertain" and not r.get("reviewed", False):
            cls = "uncertain"
        else:
            cls = r["label"]
        region_rows.append(
            f'<tr><td><span class="dot {cls}"></span>{html.escape(r["label"])}</td>'
            f"<td>{_fmt(r['start_s'])} – {_fmt(r['end_s'])}</td>"
            f"<td>{_fmt(r['end_s'] - r['start_s'])}</td>"
            f"<td>{'已审核' if r.get('reviewed') else '未审核'}</td></tr>"
        )
    region_table = "\n".join(region_rows) if region_rows else '<tr><td colspan="4">无标注</td></tr>'

    banner = (
        '<div class="ok"><b>完整事件跟踪＋微弱噪声过滤（无需训练）</b>：变化触发后，'
        '继续跟踪持续段和衰减尾部，不以边缘分数回落作为结束。候选不等于已确认故障；'
        '稳定新工况可结束事件，无需恢复旧幅值。边界和过滤门槛仍需人工检验。</div>'
    )
    total_alarm = sum(ev["duration_s"] for ev in events)
    threshold_unit = "dB change"
    y_wave = [(0.0, "+1"), (50.0, "0"), (100.0, "−1")]
    y_spec = [(0.0, _fmt(analyzed_fmax / 1000, 1) + "k"), (25.0, _fmt(analyzed_fmax * 0.75 / 1000, 1) + "k"),
              (50.0, _fmt(analyzed_fmax / 2 / 1000, 1) + "k"), (75.0, _fmt(analyzed_fmax * 0.25 / 1000, 1) + "k"),
              (100.0, "0")]
    cfg_txt = (
        f"sr={feature_config.get('sr')} · n_fft={feature_config.get('n_fft')} · "
        f"hop={feature_config.get('hop')} · n_mels={feature_config.get('n_mels')} · "
        f"fmin={feature_config.get('fmin')} · fmax={feature_config.get('fmax')} · "
        f"target_step={feature_config.get('target_step')}s · compare={feature_config.get('compare_s')}s · "
        f"smooth={feature_config.get('smooth_s')}s"
    )
    post_txt = (
        f"min_duration={postprocess.get('min_duration')}s · merge_gap={postprocess.get('merge_gap')}s · "
        f"low_ratio={postprocess.get('low_ratio')} · padding={postprocess.get('padding')}s · "
        f"reference={postprocess.get('reference_s')}s · release={postprocess.get('release_db')}dB · "
        f"quiet={postprocess.get('quiet_s')}s · settle={postprocess.get('settle_s')}s · "
        f"stability={postprocess.get('stability_db')}dB · prominence={postprocess.get('min_prominence')} · "
        f"relative_floor={postprocess.get('relative_floor_db')}dB"
    )
    review_html = (
        f'<p class="note">审核模板：<a href="{html.escape(review_relpath)}">{html.escape(review_relpath)}</a>'
        "（填写后用 <code>review-import</code> 导入；L2 字段与 L1 字段独立）</p>"
        if review_relpath
        else ""
    )

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>L1 频谱波动窗口报告 · {html.escape(file_name)}</title>
<style>
  :root {{ --ink:#1b2430; --muted:#66707d; --line:#e2e6eb; --bg:#f5f7f9; --accent:#1f4e8c; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:26px; background:var(--bg); color:var(--ink);
    font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:21px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:16px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin-bottom:14px; }}
  .warn {{ background:#fff5f0; border:1px solid #f0c4b4; color:#8a321d; padding:9px 13px; border-radius:8px; margin-bottom:12px; }}
  .ok {{ background:#f1f8f3; border:1px solid #c3e2cb; color:#22653a; padding:9px 13px; border-radius:8px; margin-bottom:12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }}
  .kpi {{ background:var(--bg); border-radius:8px; padding:9px 11px; }}
  .kpi .v {{ font-size:18px; font-weight:600; color:var(--accent); }}
  .kpi .l {{ font-size:12px; color:var(--muted); }}
  .plot-title {{ font-size:13px; font-weight:600; margin:8px 0 4px; }}
  .plot {{ display:flex; align-items:stretch; }}
  .yaxis {{ position:relative; font-size:11px; color:var(--muted); }}
  .yaxis span {{ position:absolute; right:6px; white-space:nowrap; }}
  .canvas {{ flex:1; min-width:0; }}
  img {{ width:100%; display:block; border-radius:4px; }}
  .xaxis {{ position:relative; height:18px; margin-top:2px; font-size:11px; color:var(--muted); }}
  .xaxis span {{ position:absolute; white-space:nowrap; }}
  .xaxis b {{ position:absolute; right:0; font-weight:400; }}
  .cap {{ font-size:12px; color:var(--muted); margin:4px 0 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  th,td {{ padding:6px 7px; border-bottom:1px solid var(--line); text-align:right; vertical-align:middle; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:500; background:#fafbfc; }}
  .note {{ color:var(--muted); font-size:12.5px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:4px; }}
  audio {{ height:28px; max-width:220px; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }}
  .dot.normal {{ background:#7fc896; }} .dot.anomaly {{ background:#e28a7a; }} .dot.uncertain {{ background:#b9bec5; }}
  details {{ margin-top:6px; }}
  a {{ color:var(--accent); }}
</style></head><body><div class="wrap">
  <h1>L1 异常检测报告</h1>
  <div class="sub">{html.escape(file_name)} · 音频 ID <code>{html.escape(audio_id)}</code> ·
    真实时长 {duration:.3f}s · 采样率 {sr} Hz · 分析频段 0–{analyzed_fmax:.0f} Hz</div>
  {banner}
  <div class="card"><div class="grid">
    <div class="kpi"><div class="v">{len(events)}</div><div class="l">报警事件</div></div>
    <div class="kpi"><div class="v">{total_alarm:.2f}s</div><div class="l">报警总时长</div></div>
    <div class="kpi"><div class="v">{_fmt(threshold, 4)} {threshold_unit}</div><div class="l">阈值（{html.escape(threshold_source)}）</div></div>
    <div class="kpi"><div class="v">{html.escape(detector_type)}</div><div class="l">L1 检测器</div></div>
    <div class="kpi"><div class="v">待审核</div><div class="l">L2 故障类型</div></div>
  </div></div>

  <div class="card">
    <div class="plot-title">原始音频（可播放）</div>
    <audio controls preload="none" src="{html.escape(audio_relpath)}" style="width:100%;max-width:100%"></audio>
  </div>

  <div class="card">
    {_plot_block(waveform_name, "波形（时间轴：秒；含人工标注明暗）", "绿色=已审核正常，红色=已审核异常，灰色=未审核；橙色框=模型事件。", duration, y_labels=y_wave)}
    {_plot_block(spectrogram_name, "时频图（时间轴：秒；线性频率，低频在下）", "橙色完整矩形=候选事件窗口（持续段＋尾部）；顶部色带为人工标注。颜色越亮能量越强。", duration, y_labels=y_spec)}
    {_plot_block(score_name, "边缘触发分数（不是事件持续分数）", f"红色虚线=触发阈值 {_fmt(threshold, 4)}；分数回落不代表事件结束。", duration)}
    {_plot_block("activity.png", "相对事件前背景的持续活动分数", f"红线=释放阈值 {_fmt(postprocess.get('release_db'), 2)} dB；需连续安静或新工况稳定才结束。过滤掉的候选可能仍有曲线，但没有事件框。", duration)}
  </div>

  <div class="card">
    <div class="plot-title">候选异常窗口</div>
    <table><thead><tr>
      <th>#</th><th>起止 (s)</th><th>时长 (s)</th><th>触发峰值</th><th>结束依据</th>
      <th>与人工标注</th><th>L1 人工</th><th>L2 人工</th><th>试听（含上下文）</th>
    </tr></thead><tbody>{event_table}</tbody></table>
    <p class="note">触发分数与持续活动分数不同，均不是概率。检测窗口 WAV 不含额外试听上下文。
    事件特征同时写入 <code>events.csv</code> / <code>events.json</code>，供 L2 使用。</p>
    {review_html}
  </div>

  <div class="card">
    <div class="plot-title">人工标注区间</div>
    <table><thead><tr><th>标签</th><th>起止 (s)</th><th>时长 (s)</th><th>状态</th></tr></thead>
    <tbody>{region_table}</tbody></table>
    <p class="note">L1 不需要训练。标注仅用于检验窗口；未审核区域不当正常，整段粗标不展开成精确真值。</p>
  </div>

  <div class="card"><details><summary>模型、特征与后处理参数</summary>
    <p class="note">模型 <code>{html.escape(model_version)}</code> · 检测器 {html.escape(detector_type)} ·
      分数类型 <code>{html.escape(score_type)}</code> · 阈值 {_fmt(threshold, 4)}（来源 {html.escape(threshold_source)}）</p>
    <p class="note">特征：{html.escape(cfg_txt)}</p>
    <p class="note">后处理：{html.escape(post_txt)}</p>
    <p class="note">逐时刻触发、活动、显著性、能量和窗口状态见 <a href="scores.csv">scores.csv</a>。
    <a href="events.json">events.json</a> 保留过滤原因；相对能量过滤可能漏掉较弱的真实异常。</p>
    <p class="note">本报告只定位“明显频谱波动”，不确认异常或机械故障类型；L2 未训练时统一为“待审核/未训练”。</p>
  </details></div>
</div></body></html>
"""
    path = out / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def write_batch_report(out_dir, summary):
    """Portable overview; all report links are relative to the batch directory."""
    from urllib.parse import quote
    out = Path(out_dir)
    rows = []
    for item in summary["files"]:
        relative = Path(item["report"]).resolve().relative_to(out.resolve()).as_posix()
        windows = "；".join(f'{w["start_s"]:.2f}–{w["end_s"]:.2f}'
                            + ("（末尾未结束）" if w["right_censored"] else "")
                            for w in item["windows"]) or "无"
        rows.append(
            f'<tr><td><a href="{quote(relative)}">{html.escape(item["file"])}</a></td>'
            f'<td>{item["duration_s"]:.2f}</td><td>{item["events"]}</td>'
            f'<td>{item["alarm_s"]:.2f}</td><td>{item["filtered_candidates"]}</td>'
            f'<td>{html.escape(windows)}</td></tr>')
    failures = "".join(f'<li>{html.escape(f["file"])}：{html.escape(f["error"])}</li>'
                       for f in summary["failures"])
    cfg = html.escape(__import__("json").dumps(summary["detector_config"], ensure_ascii=False, indent=2))
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>L1 完整事件窗口 · 批量测试</title>
<style>body{{max-width:1400px;margin:30px auto;padding:0 20px;color:#243347;
font:15px/1.65 system-ui,sans-serif;background:#f6f8fb}}table{{border-collapse:collapse;width:100%;background:white}}
td,th{{padding:10px;border-bottom:1px solid #dce2eb;text-align:left}}td:first-child{{max-width:360px;overflow-wrap:anywhere}}
a{{color:#2056a3}}.note{{background:#fff4da;padding:14px;border-radius:8px}}pre{{overflow:auto}}</style>
<h1>L1 完整事件窗口 · 批量测试</h1>
<p>成功 {summary["files_succeeded"]}/{summary["files_requested"]} 条；
录音共 {summary["total_duration_s"]:.2f} 秒；
候选 {summary["total_events"]} 个，窗口共 {summary["total_alarm_s"]:.2f} 秒。</p>
<p class="note">{html.escape(summary["note"])} 文件夹名“异常”不是逐段真值。
微弱候选已过滤，但过滤结果不代表确定是噪声；仍可能漏掉弱异常。
点击录音名称可查看频谱窗口、播放音频并审核。</p>
<table><thead><tr><th>录音 / 详细报告</th><th>时长 s</th><th>窗口数</th>
<th>窗口时长 s</th><th>过滤候选数</th><th>窗口范围 s</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
{"<h2>未成功处理</h2><ul>"+failures+"</ul>" if failures else ""}
<p><a href="summary.json">完整汇总 JSON</a> · 各录音子目录含 events.csv、windows/、review.csv。</p>
<details><summary>本批统一配置</summary><pre>{cfg}</pre></details></html>"""
    target = out / "index.html"
    target.write_text(body, encoding="utf-8")
    return target
