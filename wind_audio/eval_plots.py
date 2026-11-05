"""Per-recording L1 comparison figures and an ordered PDF (test mode only)."""
from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from .evaluate import _percent, _reviewed_regions, match_events

GREEN, RED, BLUE, ORANGE, GRAY = "#25805b", "#c53e4e", "#347eb8", "#ec8c20", "#dfe3e8"


def natural_key(value):
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", str(value))]


def merge_intervals(events, gap):
    """Return copies; clustering is diagnostic and must never modify truth."""
    merged = []
    for event in sorted(events, key=lambda e: e["start_s"]):
        a, b = event["start_s"], event["end_s"]
        if merged and a - merged[-1]["end_s"] <= gap + 1e-9:
            merged[-1]["end_s"] = max(merged[-1]["end_s"], b)
        else:
            merged.append({"start_s": a, "end_s": b})
    return merged


def _font_properties(font_path=None):
    from matplotlib import font_manager
    if font_path:
        if not Path(font_path).is_file():
            raise ValueError(f"图表字体不存在: {font_path}")
        return font_manager.FontProperties(fname=str(font_path))
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    candidates += [f.fname for f in font_manager.fontManager.ttflist
                   if f.name in {"Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"}]
    for candidate in candidates:
        if Path(candidate).is_file():
            return font_manager.FontProperties(fname=candidate)
    raise ValueError("未找到中文字体；请安装 Noto Sans CJK，或使用 --plot-font 指定中文 TTF/TTC 字体")


def _envelope(waveform, sr, width=2400):
    """Min/max pooling preserves short impulses without normalising volume."""
    edges = np.linspace(0, len(waveform), min(width, len(waveform)) + 1, dtype=int)
    times = (edges[:-1] + edges[1:]) / (2 * sr)
    return times, np.minimum.reduceat(waveform, edges[:-1]), np.maximum.reduceat(waveform, edges[:-1])


def _timeline(ax, gt, pred, matches, regions, duration, font, *, no_truth, normal=False):
    from matplotlib.font_manager import FontProperties
    text_font = FontProperties(size=7)
    renderer = ax.figure.canvas.get_renderer()
    for region in regions:
        if region["label"] == "uncertain":
            ax.axvspan(region["start_s"], region["end_s"], color=GRAY, alpha=.5, lw=0, zorder=0)
    gt_matched = {m["gt_index"] for m in matches} if matches is not None else None
    pred_matched = {m["pred_index"] for m in matches} if matches is not None else None
    for events, y, known, truth in ((gt, 1., gt_matched, True), (pred, 0., pred_matched, False)):
        # Place short-window labels on alternating levels to avoid collisions.
        text_ends = [-float("inf")] * 3
        if not events:
            ax.text(.012, y, no_truth if truth else ("无预测正常区域" if normal else "无预测事件"), transform=ax.get_yaxis_transform(),
                    fontproperties=font, fontsize=9, va="center", color="#7b8491")
        for index, event in enumerate(events):
            a, b = event["start_s"], event["end_s"]
            reviewed = any(r["label"] != "uncertain" and min(b, r["end_s"]) > max(a, r["start_s"])
                           for r in regions)
            matched = known is None or index in known
            face = (GRAY if known is None else GREEN if matched else RED) if truth else (BLUE if reviewed else GRAY)
            edge = ORANGE if not truth and reviewed and not matched else face
            ax.broken_barh([(a, b - a)], (y - .115, .23), facecolors=face, edgecolors=edge,
                          linewidth=1.7 if edge == ORANGE else .5, zorder=3)
            text = f"{index + 1}: {a:.2f}-{b:.2f}"
            text_width = renderer.get_text_width_height_descent(text, text_font, ismath=False)[0]
            half_width = duration * (text_width / ax.bbox.width / 2 + .004)
            middle = min(max((a + b) / 2, half_width), max(half_width, duration - half_width))
            lane = next((i for i, end in enumerate(text_ends) if middle - half_width > end),
                        int(np.argmin(text_ends)))
            text_ends[lane] = middle + half_width
            ax.text(middle, y + .18 + lane * .14, text, fontproperties=text_font,
                    ha="center", va="bottom", color="#344052")
    ax.set_ylim(-.35, 1.72)
    ax.set_yticks([1, 0], ["人工正常区域", "模型正常区域"] if normal else ["标签时间段", "预测时间段"],
                  fontproperties=font, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color="#e0e5eb", linewidth=.6, zorder=0)


class ComparisonReport:
    """Stream one image/page at a time; no whole-dataset audio kept in RAM."""

    def __init__(self, out_dir, *, cluster_gap=.3, font_path=None):
        if not np.isfinite(cluster_gap) or cluster_gap < 0:
            raise ValueError("--plot-merge-gap 必须是有限非负数")
        # Use a writable cache even on a headless or restricted machine.
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wind_audio_matplotlib"))
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from reportlab.pdfgen.canvas import Canvas
            from reportlab.lib.pagesizes import A4, landscape
        except ImportError as exc:
            raise ValueError("图表/PDF 依赖缺失，请运行 python -m pip install 'matplotlib>=3.8,<4' 'reportlab>=4,<5'") from exc
        self.Figure, self.FigureCanvasAgg, self.Canvas = Figure, FigureCanvasAgg, Canvas
        self.page_size = landscape(A4)
        self.font = _font_properties(font_path)
        self.out = Path(out_dir)
        self.gap = cluster_gap
        self.canvas = None
        self.pages = []
        self.final_pdf = self.out / "comparison.pdf"
        self.partial_pdf = self.out / "comparison.partial.pdf"
        self.index_path = self.out / "plots/index.csv"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.canvas is not None:
            self.canvas.save()
            if exc_type is None:
                self.partial_pdf.replace(self.final_pdf)
                with self.index_path.open("w", encoding="utf-8-sig", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=["page", "audio_id", "audio_path", "image"])
                    writer.writeheader()
                    writer.writerows(self.pages)
            else:
                self.partial_pdf.unlink(missing_ok=True)

    def add(self, record, anns, waveform, sr, bundle, traces, row, detector, index, total):
        from matplotlib.patches import Patch
        if self.canvas is None:
            (self.out / "plots").mkdir(parents=True, exist_ok=True)
            self.canvas = self.Canvas(str(self.partial_pdf), pagesize=self.page_size, pageCompression=1)
            self.canvas.setTitle("L1 test: labels and predictions")
        duration = record.duration_s
        normal = row.get("target_label") == "normal"
        gt, pred = ((row["ground_truth_normal_regions"], row["predicted_normal_regions"]) if normal else
                    (row["ground_truth_events"], row["predicted_events"]))
        regions = _reviewed_regions(record, anns)
        clustered_gt, clustered_pred = merge_intervals(gt, self.gap), merge_intervals(pred, self.gap)
        cluster_matches = (match_events(clustered_pred, clustered_gt, row["iou_threshold"])["matches"]
                           if row["unknown_s"] < 1e-8 else None)
        fig = self.Figure(figsize=(11.69, 8.27), facecolor="white")
        self.FigureCanvasAgg(fig)
        axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.4, 1.4]})
        fig.subplots_adjust(left=.095, right=.975, top=.835, bottom=.075, hspace=.48)
        fp = self.font
        title = "正常区域测试" if normal else "L1 测试"
        fig.text(.035, .965, f"{title} | {record.audio_id}", fontproperties=fp, fontsize=16, weight="bold", color="#142b42")
        filename = Path(row["audio_path"]).name
        fig.text(.035, .939, filename, fontproperties=fp, fontsize=7.5, color="#667487")
        result_names = {"TP": "含事件 / 已报警", "TN": "正常 / 无报警", "FP": "正常 / 误报警", "FN": "含事件 / 未报警", "excluded": "文件标签未知"}
        outcome = (f"正常精确率 {_percent(row['time_metrics']['precision'])} / 正常召回率 {_percent(row['time_metrics']['recall'])}"
                   if normal else result_names[row['file_outcome']])
        fig.text(.035, .913,
                 f"{duration:.2f} 秒  |  标签 {len(gt)} 段  |  预测 {len(pred)} 段  |  "
                 f"匹配 {row['n_matched']}/{len(gt)}  |  {outcome}",
                 fontproperties=fp, fontsize=10, color="#344052")
        legend = [Patch(facecolor=GREEN, label="匹配标签"), Patch(facecolor=RED, label="未匹配标签"),
                  Patch(facecolor=BLUE, label="模型预测"), Patch(facecolor=BLUE, edgecolor=ORANGE, linewidth=2, label="未匹配预测"),
                  Patch(facecolor=GRAY, label="未知区域 / 不评分")]
        legend_font = fp.copy()
        legend_font.set_size(8)
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.03, .893), ncol=5,
                   prop=legend_font, frameon=False)

        t, low, high = _envelope(waveform, sr)
        axes[0].fill_between(t, low, high, color="#66717a", linewidth=.25)
        axes[0].set_title("音频波形（最小 / 最大包络，未做增益归一化）", loc="left", fontproperties=fp, fontsize=10)
        axes[0].set_ylabel("幅值", fontproperties=fp, fontsize=9)
        axes[0].grid(axis="x", color="#e0e5eb", linewidth=.6)

        curves = traces.get("curves") or []
        primary = curves[0] if curves else None
        if primary is None:
            raise ValueError("检测器没有提供可绘制的诊断曲线")
        same_unit = [c for c in curves if c.get("unit") == primary.get("unit")]
        line_colors = ["#283d52", "#159987", "#7a5cc0"]
        dash = ["--", ":", "-."]
        for i, curve in enumerate(same_unit):
            axes[1].plot(bundle.times, curve["values"], color=line_colors[i % 3], lw=.85, label=curve["label"])
            axes[1].axhline(curve["threshold"], color=[RED, ORANGE, BLUE][i % 3], ls=dash[i % 3], lw=.85,
                            label=f"{curve['label']}阈值 {curve['threshold']:g}")
        axes[1].set_title(primary["title"], loc="left", fontproperties=fp, fontsize=10)
        unit_label = {"dB": "分数 / dB", "noise": "变化 / 噪声σ", "ratio": "比例 (0–1)"}.get(
            primary.get("unit"), "正常偏离倍数" if primary.get("unit") == "deviation_ratio" else primary.get("unit", ""))
        axes[1].set_ylabel(unit_label, fontproperties=fp, fontsize=9)
        axes[1].set_ylim(bottom=0)
        small_font = fp.copy()
        small_font.set_size(7)
        axes[1].legend(loc="upper right", prop=small_font, framealpha=.9, ncol=4)
        axes[1].grid(color="#e0e5eb", lw=.6)

        no_truth = ("无人工正常区域" if normal else "无异常区间标签（已审核正常）"
                    if row["file_true_label"] == "normal" and row["unknown_s"] < 1e-8 else "无精确异常区间标签")
        _timeline(axes[2], gt, pred, row["matches"], regions, duration, fp, no_truth=no_truth, normal=normal)
        region_name = "正常区域" if normal else "原始事件"
        axes[2].set_title(f"{region_name}对比 | 一对一匹配 IoU >= {row['iou_threshold']:g}",
                          loc="left", fontproperties=fp, fontsize=10)
        _timeline(axes[3], clustered_gt, clustered_pred, cluster_matches, regions, duration, fp, no_truth=no_truth, normal=normal)
        suffix = f"匹配 {len(cluster_matches)}/{len(clustered_gt)}" if cluster_matches is not None else "未完整审核，仅显示范围"
        cluster_title = "正常区域合并辅助对比" if normal else "事件簇辅助对比"
        axes[3].set_title(f"{cluster_title} | 两侧间隔 <= {self.gap:g} 秒合并 | {suffix}",
                          loc="left", fontproperties=fp, fontsize=10)
        axes[3].set_xlabel("相对原始音频起点的时间（秒）", fontproperties=fp, fontsize=9)
        for ax in axes:
            ax.set_xlim(0, duration)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#8c98a4")
                spine.set_linewidth(.55)
        footer = ("上行：人工正常区域；下行：模型正常区域（非异常窗口）。辅助合并不修改正式指标。" if normal else
                  "上下两行使用同一时间轴。事件簇仅供观察，不修改标签、预测或正式指标。")
        fig.text(.035, .025, footer,
                 fontproperties=fp, fontsize=8, color="#667487")
        fig.text(.97, .025, f"{index} / {total}", ha="right", fontsize=8, color="#667487")
        name = re.sub(r"[^\w.-]+", "_", record.audio_id)[:80] or "audio"
        relative = f"plots/{index:04d}_{name}.png"
        image_path = self.out / relative
        fig.savefig(image_path, dpi=180, facecolor="white")
        fig.clear()
        self.canvas.drawImage(str(image_path), 0, 0, width=self.page_size[0], height=self.page_size[1])
        self.canvas.bookmarkPage(f"page_{index}")
        self.canvas.addOutlineEntry(record.audio_id, f"page_{index}", level=0)
        self.canvas.showPage()
        self.pages.append({"page": index, "audio_id": record.audio_id, "audio_path": row["audio_path"], "image": relative})
        row["comparison_image"] = relative
        row["pdf_page"] = index
        if index == 1 or index % 10 == 0 or index == total:
            print(f"对照图 {index}/{total}: {record.audio_id}", flush=True)

    def metadata(self):
        return {"pdf": self.final_pdf.name, "image_index": "plots/index.csv", "pages": len(self.pages),
                "cluster_gap_s": self.gap, "cluster_changes_metrics": False,
                "order": [p["audio_id"] for p in self.pages]}
