"""PANNs-only comparison reports; rendering never changes predictions or labels."""
from __future__ import annotations

import csv
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from .data import assert_disjoint, native_audio, partition, write_json

GREEN, BLUE, RED, GRAY = "#25805b", "#347eb8", "#c53e4e", "#e3e7ed"


def report_dependencies(font_path=None):
    """Fail before training if required PDF libraries/font are unavailable."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wind_audio_matplotlib"))
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.pagesizes import A4, landscape
        from matplotlib.ft2font import FT2Font
        # Read-only reuse of plotting utilities, NOT the old evaluator/detector.
        from ..eval_plots import _font_properties
    except (ImportError, OSError) as exc:
        raise ValueError("报告需要本地 matplotlib、reportlab 和中文字体；不会自动安装") from exc
    font = _font_properties(font_path)
    try:
        loaded = FT2Font(font.get_file())
        if any(ord(c) not in loaded.get_charmap() for c in "正常验证测试未知"):
            raise ValueError("报告字体缺少中文字符，请通过 --plot-font 指定中文字体")
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"报告字体无法加载: {font.get_file()}") from exc
    return Figure, FigureCanvasAgg, Canvas, landscape(A4), font


def write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_inputs(data, out):
    """Audit trail, no complement-generated annotations and no inferred splits."""
    rows = data["records"]
    write_csv(out / "manifest.csv", ("audio_id", "audio_path", "source_id", "split", "duration_s",
                                    "audio_sha256", "pcm_sha256"), rows)
    write_csv(out / "pairing.csv", ("audio_id", "audio_path", "label_path", "source_id", "split"), rows)
    write_csv(out / "annotations.csv", ("audio_id", "start_s", "end_s", "label"),
              ({"audio_id": r["audio_id"], **a} for r in rows for a in r["intervals"]))


def percent(value):
    return "不可计算" if value is None else f"{value:.2%}"


def normal_metrics(metrics):
    """Invert only confusion-matrix roles, NEVER invert unknown intervals."""
    tp = metrics["tn_s"]
    precision_den = tp + metrics["fn_s"]
    return {"precision_on_reviewed_predictions": tp / precision_den if precision_den else None,
            "recall_including_abstentions": tp / metrics["reviewed_normal_s"]
            if metrics["reviewed_normal_s"] else None}


def unknown_truth(intervals, duration):
    result, start = [], 0.0
    for r in intervals:
        if r["label"] == "unknown":
            continue
        if r["start_s"] > start:
            result.append({"start_s": start, "end_s": r["start_s"]})
        start = r["end_s"]
    if start < duration:
        result.append({"start_s": start, "end_s": duration})
    return result


def _timeline(ax, truth, predicted, truth_unknown, predicted_unknown, duration, font, label):
    """Two independent unknown lanes; no clustering across confirmed normals."""
    for regions, y in ((truth_unknown, 1), (predicted_unknown, 0)):
        ax.broken_barh([(r["start_s"], r["end_s"] - r["start_s"]) for r in regions],
                      (y - .18, .36), facecolors=GRAY, edgecolors="#c2cad5", hatch="///", lw=.3)
    for regions, y, color in ((truth, 1, GREEN), (predicted, 0, BLUE)):
        ends = [-np.inf] * 2
        for i, r in enumerate(regions, 1):
            a, b = r["start_s"], r["end_s"]
            ax.broken_barh([(a, b - a)], (y - .12, .24), facecolors=color)
            text = f"{i}: {a:.2f}-{b:.2f}"
            # Conservative text width; suppress crowded labels (full ranges in CSV).
            width = duration * len(text) * .0065
            middle = min(max((a + b) / 2, width / 2), max(width / 2, duration - width / 2))
            lane = next((j for j, end in enumerate(ends) if middle - width / 2 > end), None)
            if lane is not None:
                ax.text(middle, y + .17 + lane * .19, text, ha="center", va="bottom",
                        fontsize=6.4, color="#344052")
                ends[lane] = middle + width / 2
    names = ["人工异常标注", "模型异常候选"] if label == "异常候选" else [f"人工{label}", f"模型{label}"]
    ax.set_yticks([1, 0], names, fontproperties=font, fontsize=8)
    ax.set_ylim(-.3, 1.65)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#dce3eb", linewidth=.6)


class ComparisonPages:
    """One landscape A4 page per recording; same format in val and test."""
    def __init__(self, out, split, font_path=None):
        self.Figure, self.Agg, self.Canvas, self.page_size, self.font = report_dependencies(font_path)
        self.out, self.split, self.pages, self.canvas = Path(out), split, [], None
        self.partial = self.out / "comparison.partial.pdf"
        self.final = self.out / "comparison.pdf"

    def __enter__(self):
        if self.final.exists() or self.partial.exists():
            raise ValueError("PDF 已存在，拒绝覆盖")
        (self.out / "plots").mkdir()
        self.canvas = self.Canvas(str(self.partial), pagesize=self.page_size, pageCompression=1)
        self.canvas.setTitle(f"L1 PANNs {self.split}: reviewed labels and predictions")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.canvas.save()
        if exc_type is None:
            if not self.pages:
                raise ValueError("没有报告页面，不能生成空验证/测试报告")
            self.partial.rename(self.final)
            write_csv(self.out / "plots/index.csv", ("page", "audio_id", "audio_path", "image"), self.pages)
        # On failure retain the clearly named partial artifact for diagnosis, never publish as complete.

    def add(self, record, evaluated, index, total):
        from matplotlib.patches import Patch
        from ..eval_plots import _envelope
        pred, metrics = evaluated["prediction"], evaluated["metrics"]
        wave, info = native_audio(record["audio_path"])
        duration, fp = info.duration_s, self.font
        fig = self.Figure(figsize=(11.69, 8.27), facecolor="white")
        self.Agg(fig)
        axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.05, 1.3, 1.3]})
        fig.subplots_adjust(left=.12, right=.975, top=.81, bottom=.10, hspace=.53)
        name = record["audio_id"]
        title_name = name if len(name) <= 62 else name[:59] + "..."
        title = "验证集" if self.split == "val" else "测试集"
        fig.text(.035, .958, f"L1 PANNs {title} | {title_name}", fontproperties=fp, fontsize=15, color="#142b42")
        filename = Path(record["audio_path"]).name
        fig.text(.035, .929, filename if len(filename) < 125 else filename[:122] + "...",
                 fontproperties=fp, fontsize=7, color="#667487")
        nm = normal_metrics(metrics)
        fig.text(.035, .902,
                 f"{duration:.2f} 秒 | 正常精确率 {percent(nm['precision_on_reviewed_predictions'])} | "
                 f"正常召回率（含拒判）{percent(nm['recall_including_abstentions'])} | "
                 f"已审核区域拒判 {metrics['abstained_reviewed_s']:.2f} 秒",
                 fontproperties=fp, fontsize=9, color="#344052")
        font_small = fp.copy()
        font_small.set_size(8)
        fig.legend(handles=[Patch(facecolor=GREEN, label="人工明确标注"), Patch(facecolor=BLUE, label="模型预测"),
                            Patch(facecolor=GRAY, hatch="///", label="人工未审核 / 模型拒判（分别显示）")],
                   loc="upper left", bbox_to_anchor=(.028, .883), ncol=3, prop=font_small, frameon=False)
        t, low, high = _envelope(wave, info.sample_rate)
        axes[0].fill_between(t, low, high, color="#66717a", linewidth=.2)
        axes[0].set_title("音频波形（原始幅值，最小 / 最大包络）", loc="left", fontproperties=fp, fontsize=9)
        axes[0].set_ylabel("幅值", fontproperties=fp, fontsize=8)
        scores = pred["scores"]
        axes[1].plot(scores["time_s"], np.asarray(scores["values"], dtype=float), color="#283d52", lw=.85)
        for key, color, style in (("low", GREEN, ":"), ("high", RED, "--"), ("impulse", "#b87a16", "-.")):
            value = pred["model"]["thresholds"][key]
            axes[1].axhline(value, color=color, ls=style, lw=.7, label=f"{key}: {value:.4g}")
        for r in pred["unknown_regions"]:
            axes[1].axvspan(r["start_s"], r["end_s"], color=GRAY, alpha=.35, lw=0)
        axes[1].set_title("网络重建残差 RMS（不是概率；阈值固定）", loc="left", fontproperties=fp, fontsize=9)
        axes[1].set_ylabel("残差 RMS", fontproperties=fp, fontsize=8)
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=7, ncol=3, loc="upper right")
        truth_unknown = unknown_truth(record["intervals"], duration)
        for ax, label, predicted in ((axes[2], "normal", pred["normal_regions"]),
                                     (axes[3], "anomaly", pred["events"])):
            word = "正常区域" if label == "normal" else "异常候选"
            truth = [r for r in record["intervals"] if r["label"] == label]
            _timeline(ax, truth, predicted, truth_unknown, pred["unknown_regions"], duration, fp, word)
            ax.set_title(f"{word}对比 | 人工 {len(truth)} 段 / 模型 {len(predicted)} 段（不跨间隔合并）",
                         loc="left", fontproperties=fp, fontsize=9)
        axes[-1].set_xlabel("相对原始音频起点的时间（秒）", fontproperties=fp, fontsize=8)
        for ax in axes:
            ax.set_xlim(0, duration)
            ax.grid(axis="x", color="#dce3eb", linewidth=.6)
            ax.tick_params(labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#8c98a4")
                spine.set_linewidth(.5)
        diag = pred["diagnostics"]
        fig.text(.035, .042,
                 f"原生步长 {diag['native_step_s']:.3f} s，不等于定位精度；评分上下文约 "
                 f"{diag['score_receptive_field_bound_s']:.3f} s。未审核不评分；拒判单列，不强行归类。",
                 fontproperties=fp, fontsize=7, color="#667487")
        fig.text(.035, .023, "上下行独立保留 unknown；拥挤区间标注省略，完整起止时间见 CSV。",
                 fontproperties=fp, fontsize=7, color="#667487")
        fig.text(.97, .023, f"{index} / {total}", ha="right", fontsize=8, color="#667487")
        safe_name = re.sub(r"[^\w.-]+", "_", name)[:65] or "audio"
        relative = f"plots/{index:04d}_{safe_name}.png"
        fig.savefig(self.out / relative, dpi=160, facecolor="white")
        fig.clear()
        self.canvas.drawImage(str(self.out / relative), 0, 0, width=self.page_size[0], height=self.page_size[1])
        self.canvas.bookmarkPage(f"page_{index}")
        self.canvas.addOutlineEntry(name, f"page_{index}")
        self.canvas.showPage()
        self.pages.append({"page": index, "audio_id": name, "audio_path": record["audio_path"], "image": relative})


def write_evaluation_report(model, data, split, directory, *, font_path=None):
    """Predict exactly once per audio through the shared evaluator; then export."""
    from .model import evaluate, new_directory
    selected = partition(data, split)
    model.require_ready()
    assert_disjoint(selected, model.meta["training"]["provenance"] + model.meta["calibration"]["provenance"])
    report_dependencies(font_path)
    out = new_directory(directory)
    with ComparisonPages(out, split, font_path) as pages:
        result = evaluate(model, data, split, on_file=pages.add)
    result["visual_report"] = {"pdf": "comparison.pdf", "pages": len(pages.pages),
                               "image_index": "plots/index.csv", "order": [p["audio_id"] for p in pages.pages],
                               "changes_predictions": False, "unknown_is_normal": False}
    result["normal_metrics"] = normal_metrics(result["duration_metrics"])
    write_json(out / "metrics.json", result)
    export_inputs({"records": selected}, out)
    fields = ["audio_id", *result["duration_metrics"], "normal_precision", "normal_recall"]
    per_file, normal_rows, event_rows, unknown_rows, labels = [], [], [], [], []
    by_id = {r["audio_id"]: r for r in selected}
    for item in result["per_file"]:
        aid, pred = item["audio_id"], item["prediction"]
        nm = normal_metrics(item["metrics"])
        per_file.append({"audio_id": aid, **item["metrics"],
                         "normal_precision": nm["precision_on_reviewed_predictions"],
                         "normal_recall": nm["recall_including_abstentions"]})
        for label, dest, key in (("normal", normal_rows, "normal_regions"), ("anomaly", event_rows, "events")):
            for origin, intervals in (("truth", [r for r in by_id[aid]["intervals"] if r["label"] == label]),
                                      ("prediction", pred[key])):
                dest.extend({"audio_id": aid, "origin": origin, **r} for r in intervals)
        unknown_rows.extend({"audio_id": aid, "origin": origin, **r}
                            for origin, intervals in (("truth", unknown_truth(by_id[aid]["intervals"],
                                                                             by_id[aid]["duration_s"])),
                                                       ("prediction", pred["unknown_regions"])) for r in intervals)
        labels.append({"audio_id": aid, "n_events": len(pred["events"]),
                       "anomaly_intervals": "; ".join(f"{r['start_s']:.6f}-{r['end_s']:.6f}" for r in pred["events"]),
                       "unknown_intervals": "; ".join(f"{r['start_s']:.6f}-{r['end_s']:.6f}" for r in pred["unknown_regions"])})
    write_csv(out / "per_file.csv", fields, per_file)
    for name, rows in (("normal_regions", normal_rows), ("events", event_rows), ("unknown_regions", unknown_rows)):
        write_csv(out / f"{name}.csv", ("audio_id", "origin", "start_s", "end_s"), rows)
    write_csv(out / "labels.csv", ("audio_id", "n_events", "anomaly_intervals", "unknown_intervals"), labels)
    with (out / "brief.md").open("x", encoding="utf-8") as f:
        f.write(f"# L1 PANNs {split} 报告\n\n"
                f"录音 {len(selected)} 条；comparison.pdf {len(pages.pages)} 页。\n\n"
                f"正常精确率（已审核预测）：{percent(result['normal_metrics']['precision_on_reviewed_predictions'])}\n\n"
                f"正常召回率（含拒判）：{percent(result['normal_metrics']['recall_including_abstentions'])}\n\n"
                f"异常召回率（含拒判）：{percent(result['anomaly_recall_including_abstentions'])}\n\n"
                f"已审核区域预测覆盖率：{percent(result['reviewed_coverage'])}\n\n"
                "阈值只由独立正常 calibration 集确定；val/test 不参与训练、阈值拟合或参数选择。\n\n"
                "未审核区域不评分；模型 unknown/边缘作为拒判另列。条件指标不代表整体准确率。\n\n"
                "normal_regions.csv / events.csv / unknown_regions.csv 保存完整范围；图表不合并或改写预测。\n")
    return result
