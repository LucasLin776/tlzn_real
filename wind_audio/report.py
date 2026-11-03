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
    calibrated: bool,
    calibration: dict | None,
    demo: bool,
    backend: dict | None,
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
        event_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{_fmt(ev['start_s'])} – {_fmt(ev['end_s'])}</td>"
            f"<td>{_fmt(ev['duration_s'])}</td>"
            f"<td>{_fmt(ev['peak_score'], 3)}</td>"
            f"<td>{html.escape(str(ev.get('score_type', score_type)))}</td>"
            f"<td>{html.escape(_match_label(ev, gt_events))}</td>"
            f"<td>{html.escape(str((reviews or {}).get(ev.get('event_id'), {}).get('l1_label') or '待审核'))}</td>"
            f"<td>{html.escape(str((reviews or {}).get(ev.get('event_id'), {}).get('l2_label') or 'not_trained'))}</td>"
            f"<td>{player}</td>"
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

    if demo:
        banner = (
            '<div class="warn"><b>演示模型</b>：该模型用待检测录音的安静段拟合，'
            "这些帧不一定是正常样本。结果只用于验证链路，<b>不能当作真值、召回率或生产结论</b>。</div>"
        )
    elif not calibrated:
        banner = (
            '<div class="warn"><b>未校准</b>：阈值没有在验证集上按目标召回选取，'
            "报告中的报警数量不代表生产性能。</div>"
        )
    else:
        cal = calibration or {}
        banner = (
            f'<div class="ok"><b>已校准</b>：验证集目标召回 {_fmt(cal.get("target_recall"), 2)}，'
            f'目标{"达到" if cal.get("target_reached") else "未达到"}；'
            f'验证集实际召回 {_fmt((cal.get("validation") or {}).get("event_recall"), 3)}。</div>'
        )
    backend_note = ""
    if backend and backend.get("used") not in (None, "", "xgboost"):
        backend_note = (
            f'<div class="warn"><b>后端不是 XGBoost</b>：实际使用 <code>{html.escape(str(backend.get("used")))}</code>'
            f'（请求 {html.escape(str(backend.get("requested")))}）。'
            "该替代算法已写入模型元数据，不能按 XGBoost 结果解读。</div>"
        )

    total_alarm = sum(ev["duration_s"] for ev in events)
    threshold_unit = "z" if score_type == "baseline_z" else "score"
    y_wave = [(0.0, "+1"), (50.0, "0"), (100.0, "−1")]
    y_spec = [(0.0, _fmt(analyzed_fmax / 1000, 1) + "k"), (25.0, _fmt(analyzed_fmax * 0.75 / 1000, 1) + "k"),
              (50.0, _fmt(analyzed_fmax / 2 / 1000, 1) + "k"), (75.0, _fmt(analyzed_fmax * 0.25 / 1000, 1) + "k"),
              (100.0, "0")]
    cfg_txt = (
        f"sr={feature_config.get('sr')} · n_fft={feature_config.get('n_fft')} · "
        f"hop={feature_config.get('hop')} · n_mels={feature_config.get('n_mels')} · "
        f"fmin={feature_config.get('fmin')} · fmax={feature_config.get('fmax')} · "
        f"target_step={feature_config.get('target_step')}s · context=±{feature_config.get('context_s')}s · "
        f"normalize={feature_config.get('normalize')}"
    )
    post_txt = (
        f"min_duration={postprocess.get('min_duration')}s · merge_gap={postprocess.get('merge_gap')}s · "
        f"low_ratio={postprocess.get('low_ratio')}"
    )
    review_html = (
        f'<p class="note">审核模板：<a href="{html.escape(review_relpath)}">{html.escape(review_relpath)}</a>'
        "（填写后用 <code>review-import</code> 导入；L2 字段与 L1 字段独立）</p>"
        if review_relpath
        else ""
    )

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>L1 异常检测报告 · {html.escape(file_name)}</title>
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
  {banner}{backend_note}
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
    {_plot_block(spectrogram_name, "时频图（时间轴：秒；线性频率，低频在下）", "顶部 7px 色带为人工标注；橙色竖线=模型事件。颜色越亮能量越强。", duration, y_labels=y_spec)}
    {_plot_block(score_name, "模型分数与阈值（时间轴：秒）", f"红色虚线=阈值 {_fmt(threshold, 4)}；浅色底=人工标注区域；橙色框=报警事件。", duration)}
  </div>

  <div class="card">
    <div class="plot-title">报警事件</div>
    <table><thead><tr>
      <th>#</th><th>起止 (s)</th><th>时长 (s)</th><th>峰值分数</th><th>分数类型</th>
      <th>与人工标注</th><th>L1 人工</th><th>L2 人工</th><th>试听（含上下文）</th>
    </tr></thead><tbody>{event_table}</tbody></table>
    <p class="note">分数是检测分数，不是故障概率；{html.escape(score_type)} 未做概率校准。
    事件特征同时写入 <code>events.csv</code> / <code>events.json</code>，供 L2 使用。</p>
    {review_html}
  </div>

  <div class="card">
    <div class="plot-title">人工标注区间</div>
    <table><thead><tr><th>标签</th><th>起止 (s)</th><th>时长 (s)</th><th>状态</th></tr></thead>
    <tbody>{region_table}</tbody></table>
    <p class="note">未审核区域不参与训练、阈值校准和评估；文件级粗标不会展开成逐帧真值。</p>
  </div>

  <div class="card"><details><summary>模型、特征与后处理参数</summary>
    <p class="note">模型 <code>{html.escape(model_version)}</code> · 检测器 {html.escape(detector_type)} ·
      分数类型 <code>{html.escape(score_type)}</code> · 阈值 {_fmt(threshold, 4)}（来源 {html.escape(threshold_source)}）</p>
    <p class="note">特征：{html.escape(cfg_txt)}</p>
    <p class="note">后处理：{html.escape(post_txt)}</p>
    <p class="note">校准：{html.escape(str(calibrated))}；demo：{html.escape(str(demo))}；
      后端：{html.escape(str((backend or {}).get("used", "baseline")))}</p>
    <p class="note">本报告只说明“声音是否异常”，不判断机械故障类型；L2 未训练时统一为“待审核/未训练”。</p>
  </details></div>
</div></body></html>
"""
    path = out / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
