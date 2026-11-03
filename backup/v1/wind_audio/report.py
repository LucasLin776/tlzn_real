"""HTML report writer (self-contained except for the two PNGs)."""
from __future__ import annotations

import html
from pathlib import Path


def _fmt(value: float, digits: int = 2) -> str:
    if value is None:
        return "-"
    if value != value:  # NaN
        return "-"
    return f"{value:.{digits}f}"


def write_report(
    out_dir: str | Path,
    *,
    file_name: str,
    duration: float,
    sr: int,
    model_version: str,
    threshold: float,
    demo: bool,
    train_files: int,
    events: list[dict],
    spectrogram_name: str = "spectrogram.png",
    score_name: str = "score.png",
) -> Path:
    out = Path(out_dir)
    rows = []
    for i, ev in enumerate(events, 1):
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{_fmt(ev['start'])} – {_fmt(ev['end'])}</td>"
            f"<td>{_fmt(ev['duration'])}</td>"
            f"<td>{_fmt(ev['peak_z'], 1)}</td>"
            f"<td>{_fmt(ev['centroid_hz'], 0)}</td>"
            f"<td>{_fmt(ev['flatness'], 3)}</td>"
            f"<td>{_fmt(ev['share_4k_10k'] * 100, 1)}%</td>"
            f"<td>{_fmt(ev['share_0_200'] * 100, 1)}%</td>"
            f"<td>{_fmt(ev['peak_freq_hz'], 0)}</td>"
            f"<td>{_fmt(ev['onset_slope_z_per_s'], 0)}</td>"
            "</tr>"
        )
    table = "\n".join(rows) if rows else '<tr><td colspan="10">未检测到事件</td></tr>'

    if demo:
        banner = (
            '<div class="warn"><b>演示模式</b>：模型只用本文件的安静段拟合，'
            "阈值没有用正常样本校准。本报告仅用于验证链路，<b>不能作为故障判定依据</b>。</div>"
        )
    else:
        banner = (
            f'<div class="ok"><b>正常基线模型</b>：由 {train_files} 个正常文件训练，'
            f"阈值 = {threshold:.1f}σ。</div>"
        )

    total_alarm = sum(ev["duration"] for ev in events)
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>L1 异常检测报告 · {html.escape(file_name)}</title>
<style>
  :root {{ --ink:#1c2430; --muted:#66707d; --line:#e3e7ec; --bg:#f6f8fa; --accent:#1f4e8c; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px; background:var(--bg); color:var(--ink);
         font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:21px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px;
           padding:18px 20px; margin-bottom:16px; }}
  .warn {{ background:#fff4f2; border:1px solid #f2c3bb; color:#8a2c1e;
           padding:10px 14px; border-radius:8px; margin-bottom:14px; }}
  .ok {{ background:#f1f8f3; border:1px solid #c4e3cc; color:#22653a;
         padding:10px 14px; border-radius:8px; margin-bottom:14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:4px 0 2px; }}
  .kpi {{ background:var(--bg); border-radius:8px; padding:10px 12px; }}
  .kpi .v {{ font-size:20px; font-weight:600; color:var(--accent); }}
  .kpi .l {{ font-size:12px; color:var(--muted); }}
  img {{ width:100%; display:block; border-radius:6px; }}
  .cap {{ font-size:12px; color:var(--muted); margin:6px 0 14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:500; background:#fafbfc; }}
  .note {{ color:var(--muted); font-size:12.5px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:4px; }}
  details {{ margin-top:8px; }}
</style></head><body><div class="wrap">
  <h1>L1 异常检测报告</h1>
  <div class="sub">{html.escape(file_name)} · 时长 {duration:.2f}s · 采样率 {sr} Hz · 模型 {html.escape(model_version)}</div>
  {banner}
  <div class="card">
    <div class="grid">
      <div class="kpi"><div class="v">{len(events)}</div><div class="l">报警事件</div></div>
      <div class="kpi"><div class="v">{total_alarm:.2f}s</div><div class="l">报警总时长</div></div>
      <div class="kpi"><div class="v">{threshold:.1f}σ</div><div class="l">当前阈值</div></div>
      <div class="kpi"><div class="v">L2 未训练</div><div class="l">故障 / 工况变化判别</div></div>
    </div>
  </div>
  <div class="card">
    <img src="{spectrogram_name}" alt="spectrogram">
    <div class="cap">时频图（线性频率 0–20 kHz）。绿框 = L1 报警事件。颜色越亮能量越强。</div>
    <img src="{score_name}" alt="score curve">
    <div class="cap">帧级异常分数（10 ms 一帧，已做 0.15 s 中值平滑）。红色虚线 = 阈值；浅红底 = 报警区间。</div>
  </div>
  <div class="card">
    <table>
      <thead><tr>
        <th>#</th><th>起止 (s)</th><th>时长 (s)</th><th>峰值 z</th>
        <th>质心 (Hz)</th><th>平坦度</th><th>4–10k 占比</th><th>&lt;200Hz 占比</th>
        <th>峰值频率 (Hz)</th><th>起始斜率 (z/s)</th>
      </tr></thead>
      <tbody>{table}</tbody>
    </table>
    <p class="note">事件特征已写入 <code>events.csv</code> / <code>events.json</code>，这是后续 L2 的训练输入。
    平坦度接近 0 → 窄带/谐波；接近 1 → 宽带噪声。起始斜率大 → 冲击型；小 → 缓变型。</p>
  </div>
  <div class="card">
    <details><summary>模型与参数</summary>
      <p class="note">版本 <code>{html.escape(model_version)}</code> · 阈值 {threshold:.2f}
      · 正常训练文件 {train_files} 个 · demo={str(demo).lower()}</p>
      <p class="note">L1 只学“正常长什么样”。L2 尚未训练，按设计在服务器上用人工复核数据迭代；
      当前报告只把事件和特征交给人审。</p>
    </details>
  </div>
</div></body></html>
"""
    path = out / "report.html"
    path.write_text(doc, encoding="utf-8")
    return path
