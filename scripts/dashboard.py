"""Tkinter dashboard for OFN-based DDoS experiments."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import tkinter as tk
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddos_ofn import BuilderConfig, DDoSDetector, DetectorConfig, GAConfig, SimulationConfig, evaluate_predictions, generate_scenario
from ddos_ofn.datasets import build_train_validation_sets
from ddos_ofn.ga_optimize import optimize_detector


ARTIFACTS_DIR = ROOT / "artifacts"
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.json"

DARK = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
PURPLE = "#bc8cff"

FONT_MONO = ("Consolas", 9)
FONT_UI = ("Segoe UI", 9)
FONT_H = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 12, "bold")

MPL_STYLE = {
    "figure.facecolor": DARK,
    "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": BORDER,
    "text.color": TEXT,
    "lines.linewidth": 1.8,
}


def _lbl(parent: tk.Widget, text: str, font: tuple[str, int] | tuple[str, int, str] = FONT_UI, color: str = TEXT, bg: str = PANEL, **kw: Any) -> tk.Label:
    return tk.Label(parent, text=text, bg=bg, fg=color, font=font, **kw)


def _sep(parent: tk.Widget, bg: str = BORDER) -> tk.Frame:
    return tk.Frame(parent, bg=bg, height=1)


def _simulation_config(routers: int, steps: int, seed: int) -> SimulationConfig:
    attack_duration = max(12, steps // 4)
    attack_start = max(16, steps // 2)
    return SimulationConfig(
        routers=routers,
        steps=steps,
        seed=seed,
        attack_start=min(attack_start, steps - attack_duration),
        attack_duration=attack_duration,
    )


def _load_saved_payload() -> dict[str, Any] | None:
    if not BEST_PARAMS_PATH.exists():
        return None
    return json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))


def _payload_to_detector(payload: dict[str, Any]) -> tuple[dict[str, float], DetectorConfig]:
    detector_cfg = DetectorConfig(
        alert_threshold=float(payload["alert_threshold"]),
        clear_threshold=float(payload["clear_threshold"]),
        alert_windows=int(payload["alert_windows"]),
        clear_windows=int(payload["clear_windows"]),
        min_positive_routers=int(payload["min_positive_routers"]),
    )
    return dict(payload["weights"]), detector_cfg


class StatusBar(tk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent, bg=DARK, height=24)
        self._items: dict[str, tk.Label] = {}

    def add(self, key: str, text: str = "", color: str = TEXT) -> None:
        lbl = tk.Label(self, text=text, bg=DARK, fg=color, font=FONT_MONO, padx=10, pady=2)
        lbl.pack(side="left")
        tk.Frame(self, bg=BORDER, width=1).pack(side="left", fill="y", pady=2)
        self._items[key] = lbl

    def upd(self, key: str, text: str, color: str = TEXT) -> None:
        if key in self._items:
            self._items[key].config(text=text, fg=color)


class LabeledSpin(tk.Frame):
    def __init__(self, parent: tk.Widget, label: str, variable: tk.IntVar, low: int, high: int) -> None:
        super().__init__(parent, bg=PANEL)
        row = tk.Frame(self, bg=PANEL)
        row.pack(fill="x")
        _lbl(row, label, anchor="w").pack(side="left")
        self.spin = tk.Spinbox(
            row,
            from_=low,
            to=high,
            textvariable=variable,
            bg=DARK,
            fg=ACCENT,
            insertbackground=ACCENT,
            buttonbackground=BORDER,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            justify="right",
            font=FONT_MONO,
            width=8,
        )
        self.spin.pack(side="right")


class DashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._setup_window()
        self._setup_style()

        self.builder_cfg = BuilderConfig()
        self.base_detector_cfg = DetectorConfig()
        self.saved_payload = _load_saved_payload()
        self.worker_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.scenario_var = tk.StringVar(value="ddos_ramp")
        self.routers_var = tk.IntVar(value=12)
        self.steps_var = tk.IntVar(value=120)
        self.seed_var = tk.IntVar(value=7)
        self.mode_var = tk.StringVar(value="baseline")
        self.header_mode_var = tk.StringVar(value="MODE: BASELINE")
        self.status_var = tk.StringVar(value="Ready")
        self.saved_info_var = tk.StringVar(value="Brak zapisanego modelu.")
        self.diagnostic_var = tk.StringVar(value="Uruchom scenariusz, aby zobaczyc diagnostyke i interpretacje wyniku.")
        self.summary_var = tk.StringVar(value="Pakietowy trend jest zamieniany na OFN per router, a potem agregowany do jednego score.")

        self.metric_vars = {
            "scenario": tk.StringVar(value="-"),
            "recall": tk.StringVar(value="-"),
            "precision": tk.StringVar(value="-"),
            "f1": tk.StringVar(value="-"),
            "fpr": tk.StringVar(value="-"),
            "delay": tk.StringVar(value="-"),
            "alarm": tk.StringVar(value="-"),
        }
        self.card_vars = {
            "mode": tk.StringVar(value="baseline"),
            "thresholds": tk.StringVar(value="alert 4.00 / clear 2.00"),
            "logic": tk.StringVar(value="2 okna / min 4 routery"),
            "model": tk.StringVar(value="brak zapisanego modelu"),
        }

        self._build_ui()
        self._refresh_saved_payload_view()
        self._set_status("Ready", GREEN)
        self._poll_worker()

    def _setup_window(self) -> None:
        self.root.title("OFN DDoS Detector - Dashboard")
        self.root.configure(bg=DARK)
        width, height = 1440, 950
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+{max((sw - width) // 2, 20)}+{max((sh - height) // 2, 20)}")
        self.root.minsize(1180, 780)
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=DARK,
            background=PANEL,
            foreground=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            arrowcolor=TEXT,
            relief="flat",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", DARK)],
            background=[("readonly", PANEL)],
            foreground=[("readonly", TEXT)],
            selectforeground=[("readonly", TEXT)],
            selectbackground=[("readonly", DARK)],
        )
        style.configure("Vertical.TScrollbar", background=BORDER, troughcolor=DARK, bordercolor=DARK, arrowcolor=TEXT)
        style.configure("Horizontal.TScrollbar", background=BORDER, troughcolor=DARK, bordercolor=DARK, arrowcolor=TEXT)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=DARK, pady=8)
        header.pack(fill="x")
        tk.Label(header, text="OFN DDoS Detector", bg=DARK, fg=ACCENT, font=FONT_TITLE).pack(side="left", padx=14)
        tk.Label(
            header,
            text="Directed fuzzy fusion of multi-router traffic",
            bg=DARK,
            fg=MUTED,
            font=FONT_MONO,
        ).pack(side="left")
        self.header_mode_label = tk.Label(header, textvariable=self.header_mode_var, bg=DARK, fg=MUTED, font=FONT_MONO)
        self.header_mode_label.pack(side="right", padx=(0, 12))
        self.header_state_label = tk.Label(
            header,
            textvariable=self.status_var,
            bg=PANEL,
            fg=GREEN,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            relief="flat",
        )
        self.header_state_label.pack(side="right", padx=(0, 10))
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        main = tk.Frame(self.root, bg=DARK)
        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=PANEL, width=340)
        left.pack(side="left", fill="y", padx=(6, 0), pady=6)
        left.pack_propagate(False)
        self._build_left(left)

        right = tk.Frame(main, bg=DARK)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self._build_right(right)

        self._sb = StatusBar(self.root)
        self._sb.pack(fill="x", side="bottom")
        self._sb.add("scenario", "SCENARIO: -", ACCENT)
        self._sb.add("mode", "MODE: baseline", MUTED)
        self._sb.add("thresholds", "THR: 4.00 / 2.00", YELLOW)
        self._sb.add("positives", "POS: -", GREEN)
        self._sb.add("peak", "PEAK: -", PURPLE)
        self._sb.add("state", "STATE: Ready", GREEN)

    def _build_left(self, parent: tk.Frame) -> None:
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _reflow(_: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(window_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _reflow)
        canvas.bind("<Configure>", _reflow)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))

        pad = dict(padx=10, pady=4, fill="x")
        def heading(text: str) -> tk.Label:
            return _lbl(inner, text, font=FONT_H, color=ACCENT)

        _sep(inner).pack(fill="x", padx=6, pady=3)
        heading("SCENARIUSZ").pack(**pad)
        _lbl(inner, "Typ ruchu", color=MUTED).pack(**pad)
        ttk.Combobox(
            inner,
            textvariable=self.scenario_var,
            values=("normal", "ddos_ramp", "ddos_pulse", "ddos_low_and_slow", "ddos_rotating", "flash_crowd", "flash_cascade"),
            state="readonly",
        ).pack(padx=10, pady=(0, 8), fill="x")
        LabeledSpin(inner, "Liczba routerow", self.routers_var, 4, 64).pack(**pad)
        LabeledSpin(inner, "Liczba krokow", self.steps_var, 48, 240).pack(**pad)
        LabeledSpin(inner, "Seed", self.seed_var, 0, 9999).pack(**pad)

        _sep(inner).pack(fill="x", padx=6, pady=3)
        heading("TRYB DETEKCJI").pack(**pad)
        _lbl(inner, "Tryb uruchomienia", color=MUTED).pack(**pad)
        ttk.Combobox(
            inner,
            textvariable=self.mode_var,
            values=("baseline", "saved_tuned"),
            state="readonly",
        ).pack(padx=10, pady=(0, 8), fill="x")
        _lbl(inner, "Zapisany model", color=MUTED).pack(**pad)
        self.saved_info_label = _lbl(inner, "", color=TEXT, justify="left", wraplength=290)
        self.saved_info_label.configure(textvariable=self.saved_info_var)
        self.saved_info_label.pack(padx=10, pady=(0, 8), fill="x")

        _sep(inner).pack(fill="x", padx=6, pady=3)
        heading("AKCJE").pack(**pad)
        self.run_button = tk.Button(
            inner,
            text="URUCHOM SCENARIUSZ",
            bg=GREEN,
            fg=DARK,
            activebackground=GREEN,
            activeforeground=DARK,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            height=2,
            command=self.run_scenario,
        )
        self.run_button.pack(padx=10, pady=(4, 6), fill="x")
        self.train_button = tk.Button(
            inner,
            text="DOSTROJ GA",
            bg=ACCENT,
            fg=DARK,
            activebackground=ACCENT,
            activeforeground=DARK,
            relief="flat",
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            height=2,
            command=self.train_tuned_model,
        )
        self.train_button.pack(padx=10, pady=2, fill="x")
        self.reload_button = tk.Button(
            inner,
            text="Wczytaj zapisany model",
            bg=BORDER,
            fg=TEXT,
            activebackground=BORDER,
            activeforeground=TEXT,
            relief="flat",
            cursor="hand2",
            font=FONT_UI,
            command=self.reload_saved_model,
        )
        self.reload_button.pack(padx=10, pady=(2, 8), fill="x")
        _lbl(
            inner,
            "Dostrajanie uzywa szybkiego zestawu syntetycznego: normal, ddos_ramp, ddos_pulse, flash_crowd.",
            color=MUTED,
            justify="left",
            wraplength=290,
        ).pack(padx=10, pady=(0, 8), fill="x")

        _sep(inner).pack(fill="x", padx=6, pady=3)
        heading("METRYKI").pack(**pad)
        for key, label in (
            ("scenario", "Scenariusz"),
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("f1", "F1"),
            ("fpr", "False Positive Rate"),
            ("delay", "Detection Delay"),
            ("alarm", "Kroki alarmu"),
        ):
            row = tk.Frame(inner, bg=PANEL)
            row.pack(fill="x", padx=10, pady=2)
            _lbl(row, f"{label}:", color=MUTED, width=18, anchor="w").pack(side="left")
            _lbl(row, "", color=TEXT, anchor="w", textvariable=self.metric_vars[key]).pack(side="left")

        _sep(inner).pack(fill="x", padx=6, pady=3)
        heading("DIAGNOSTYKA").pack(**pad)
        _lbl(inner, "", justify="left", wraplength=290, textvariable=self.summary_var).pack(padx=10, pady=(0, 8), fill="x")
        _lbl(inner, "", justify="left", wraplength=290, color=MUTED, textvariable=self.diagnostic_var).pack(padx=10, pady=(0, 8), fill="x")

    def _build_right(self, parent: tk.Frame) -> None:
        summary_outer = tk.Frame(parent, bg=PANEL, height=126)
        summary_outer.pack(fill="x")
        summary_outer.pack_propagate(False)
        top = tk.Frame(summary_outer, bg=PANEL)
        top.pack(fill="x", padx=10, pady=(8, 2))
        _lbl(top, "BIEZACY STAN", font=FONT_H, color=ACCENT).pack(side="left")
        _lbl(
            top,
            "Wykresy pokazuja score globalny, ruch, kierunki routerow i wagi.",
            font=FONT_MONO,
            color=MUTED,
        ).pack(side="right")

        cards = tk.Frame(summary_outer, bg=PANEL)
        cards.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        cards.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="card")
        for column, (title, key, color) in enumerate(
            (
                ("Tryb", "mode", ACCENT),
                ("Progi", "thresholds", YELLOW),
                ("Logika alarmu", "logic", GREEN),
                ("Model zapisany", "model", PURPLE),
            )
        ):
            card = tk.Frame(cards, bg=DARK, highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=column, padx=4, pady=2, sticky="nsew")
            _lbl(card, title, font=FONT_MONO, color=MUTED, bg=DARK).pack(anchor="w", padx=10, pady=(8, 2))
            _lbl(
                card,
                "",
                font=("Segoe UI", 10, "bold"),
                color=color,
                bg=DARK,
                justify="left",
                wraplength=180,
                textvariable=self.card_vars[key],
            ).pack(anchor="w", padx=10, pady=(0, 8))

        tk.Frame(parent, bg=BORDER, height=2).pack(fill="x", pady=(4, 0))

        main_outer = tk.Frame(parent, bg=PANEL)
        main_outer.pack(fill="both", expand=True)
        self._build_main_figure(main_outer)

        tk.Frame(parent, bg=BORDER, height=2).pack(fill="x", pady=(4, 0))

        aux_outer = tk.Frame(parent, bg=PANEL, height=238)
        aux_outer.pack(fill="x")
        aux_outer.pack_propagate(False)
        self._build_aux_panel(aux_outer)

    def _build_main_figure(self, parent: tk.Frame) -> None:
        matplotlib.rcParams.update(MPL_STYLE)
        self.figure = Figure(figsize=(10.4, 6.5), facecolor=DARK)
        grid = gridspec.GridSpec(
            2,
            3,
            figure=self.figure,
            height_ratios=[1.35, 1.0],
            left=0.055,
            right=0.985,
            top=0.95,
            bottom=0.09,
            hspace=0.34,
            wspace=0.24,
        )
        self.ax_score = self.figure.add_subplot(grid[0, :])
        self.ax_traffic = self.figure.add_subplot(grid[1, :2])
        self.ax_counts = self.figure.add_subplot(grid[1, 2])
        for axis in (self.ax_score, self.ax_traffic, self.ax_counts):
            self._style_axis(axis, PANEL)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_aux_panel(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=PANEL)
        top.pack(fill="x", padx=10, pady=(6, 2))
        _lbl(top, "WAGI I INTERPRETACJA", font=FONT_H, color=ACCENT).pack(side="left")
        _lbl(top, "Prawy panel podsumowuje co model wlasnie zrobil.", font=FONT_MONO, color=MUTED).pack(side="right")

        body = tk.Frame(parent, bg=PANEL)
        body.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        chart_box = tk.Frame(body, bg=PANEL)
        chart_box.pack(side="left", fill="both", expand=True)
        self.weights_figure = Figure(figsize=(4.6, 2.3), facecolor=PANEL)
        self.ax_weights = self.weights_figure.add_subplot(111)
        self._style_axis(self.ax_weights, DARK)
        self.weights_canvas = FigureCanvasTkAgg(self.weights_figure, master=chart_box)
        self.weights_canvas.get_tk_widget().pack(fill="both", expand=True)

        info_box = tk.Frame(body, bg=DARK, highlightbackground=BORDER, highlightthickness=1, width=360)
        info_box.pack(side="left", fill="both", padx=(8, 0))
        info_box.pack_propagate(False)
        _lbl(info_box, "Diagnoza ostatniego przebiegu", font=FONT_H, color=TEXT, bg=DARK).pack(anchor="w", padx=12, pady=(10, 4))
        _lbl(
            info_box,
            "",
            justify="left",
            wraplength=320,
            bg=DARK,
            color=TEXT,
            textvariable=self.diagnostic_var,
        ).pack(anchor="w", padx=12, pady=(0, 8))
        _lbl(info_box, "Jak czytac dashboard", font=FONT_H, color=ACCENT, bg=DARK).pack(anchor="w", padx=12, pady=(6, 4))
        _lbl(
            info_box,
            "- Score: laczna sila anomalii OFN.\n"
            "- Traffic matrix: ruch per router i krok.\n"
            "- Router direction counts: ile wezlow wzmacnia lub oslabia alarm.\n"
            "- Top weights: routery o najwiekszym udziale w decyzji.",
            justify="left",
            wraplength=320,
            bg=DARK,
            color=MUTED,
        ).pack(anchor="w", padx=12, pady=(0, 10))

    def _style_axis(self, axis: Any, facecolor: str) -> None:
        axis.set_facecolor(facecolor)
        for spine in axis.spines.values():
            spine.set_edgecolor(BORDER)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.grid(True, alpha=0.22, color=BORDER)

    def _set_status(self, message: str, color: str) -> None:
        self.status_var.set(message)
        self.header_state_label.config(fg=color)
        self._sb.upd("state", f"STATE: {message}", color)

    def _set_busy(self, busy: bool, message: str, color: str = YELLOW) -> None:
        state = "disabled" if busy else "normal"
        self.run_button.config(state=state)
        self.train_button.config(state=state)
        self.reload_button.config(state=state)
        self._set_status(message, color)

    def _refresh_saved_payload_view(self) -> None:
        payload = self.saved_payload
        if payload is None:
            self.saved_info_var.set("Brak zapisanego modelu. Tryb saved_tuned przejdzie na baseline, dopoki nie wykonasz strojenia.")
            self.card_vars["model"].set("brak")
            return

        validation = payload.get("validation", {})
        validation_text = "walidacja brak"
        if validation:
            mean_f1 = float(np.mean([item["f1"] for item in validation.values()]))
            mean_recall = float(np.mean([item["recall"] for item in validation.values()]))
            validation_text = f"srednie F1 {mean_f1:.3f}, recall {mean_recall:.3f}"
        self.saved_info_var.set(
            f"Fitness {float(payload['best_fitness']):.4f}\n"
            f"Alert {float(payload['alert_threshold']):.2f} / clear {float(payload['clear_threshold']):.2f}\n"
            f"{validation_text}"
        )
        self.card_vars["model"].set(f"fitness {float(payload['best_fitness']):.4f}\n{validation_text}")

    def reload_saved_model(self) -> None:
        self.saved_payload = _load_saved_payload()
        if self.mode_var.get() == "saved_tuned" and self.saved_payload is None:
            self.mode_var.set("baseline")
        self._refresh_saved_payload_view()
        status = "Wczytano zapisany model" if self.saved_payload else "Brak pliku best_params.json"
        color = ACCENT if self.saved_payload else RED
        self._set_status(status, color)

    def _scenario_input(self) -> tuple[str, SimulationConfig]:
        scenario_name = self.scenario_var.get()
        sim_cfg = _simulation_config(
            routers=int(self.routers_var.get()),
            steps=int(self.steps_var.get()),
            seed=int(self.seed_var.get()),
        )
        return scenario_name, sim_cfg

    def run_scenario(self) -> None:
        scenario_name, sim_cfg = self._scenario_input()
        scenario = generate_scenario(scenario_name, sim_cfg)
        detector_cfg = self.base_detector_cfg
        weights: dict[str, float] | None = None
        mode = self.mode_var.get()

        if mode == "saved_tuned" and self.saved_payload is not None:
            saved_weights, detector_cfg = _payload_to_detector(self.saved_payload)
            weights = {router: saved_weights.get(router, 1.0) for router in scenario.router_ids}
        elif mode == "saved_tuned" and self.saved_payload is None:
            self.mode_var.set("baseline")
            mode = "baseline"
            self._set_status("Brak zapisanego modelu - uruchamiam baseline", YELLOW)

        detector = DDoSDetector(self.builder_cfg, detector_cfg, weights=weights)
        trace = detector.run(scenario.traffic, scenario.router_ids, scenario.labels, scenario.name)
        metrics = evaluate_predictions(trace.labels, trace.predictions)
        self._render_result(scenario, trace, metrics, weights or {router: 1.0 for router in scenario.router_ids}, detector_cfg, mode)

    def train_tuned_model(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return

        _, sim_cfg = self._scenario_input()
        self._set_busy(True, "Training GA...", YELLOW)

        def worker() -> None:
            try:
                train_set, valid_set = build_train_validation_sets(sim_cfg)
                result = optimize_detector(
                    train_set,
                    self.builder_cfg,
                    self.base_detector_cfg,
                    GAConfig(population_size=8, generations=3, elite_count=2, seed=int(self.seed_var.get()) + 6),
                )
                tuned_detector_cfg = DetectorConfig(
                    alert_threshold=result.alert_threshold,
                    clear_threshold=result.clear_threshold,
                    alert_windows=result.alert_windows,
                    clear_windows=result.clear_windows,
                    min_positive_routers=result.min_positive_routers,
                )
                validation: dict[str, dict[str, float]] = {}
                for scenario in valid_set:
                    detector = DDoSDetector(self.builder_cfg, tuned_detector_cfg, result.weights)
                    trace = detector.run(scenario.traffic, scenario.router_ids, scenario.labels, scenario.name)
                    metrics = evaluate_predictions(trace.labels, trace.predictions)
                    validation[scenario.name] = {
                        "recall": metrics.recall,
                        "precision": metrics.precision,
                        "f1": metrics.f1,
                        "false_positive_rate": metrics.false_positive_rate,
                        "detection_delay": metrics.detection_delay,
                    }

                payload = {
                    "best_fitness": result.best_fitness,
                    "alert_threshold": result.alert_threshold,
                    "clear_threshold": result.clear_threshold,
                    "min_positive_routers": result.min_positive_routers,
                    "alert_windows": result.alert_windows,
                    "clear_windows": result.clear_windows,
                    "weights": result.weights,
                    "validation": validation,
                }
                ARTIFACTS_DIR.mkdir(exist_ok=True)
                BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self.worker_queue.put(("trained", {"payload": payload}))
            except Exception as exc:
                self.worker_queue.put(("error", {"message": str(exc)}))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_worker(self) -> None:
        try:
            kind, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll_worker)
            return

        if kind == "trained":
            self.saved_payload = payload["payload"]
            self.mode_var.set("saved_tuned")
            self._refresh_saved_payload_view()
            self._set_busy(False, "Training finished", ACCENT)
            self.run_scenario()
        elif kind == "error":
            self._set_busy(False, f"Training failed: {payload['message']}", RED)
        else:
            self._set_busy(False, "Ready", GREEN)

        self.root.after(120, self._poll_worker)

    def _validation_summary(self) -> str:
        if not self.saved_payload or "validation" not in self.saved_payload:
            return "Brak walidacji zapisanej w modelu."
        validation = self.saved_payload["validation"]
        if not validation:
            return "Brak walidacji zapisanej w modelu."
        lines = []
        for scenario, metrics in validation.items():
            lines.append(
                f"{scenario}: F1 {metrics['f1']:.3f}, recall {metrics['recall']:.3f}, FPR {metrics['false_positive_rate']:.3f}"
            )
        return "\n".join(lines)

    def _render_result(
        self,
        scenario: Any,
        trace: Any,
        metrics: Any,
        weights: dict[str, float],
        detector_cfg: DetectorConfig,
        mode: str,
    ) -> None:
        self.metric_vars["scenario"].set(scenario.name)
        self.metric_vars["recall"].set(f"{metrics.recall:.3f}")
        self.metric_vars["precision"].set(f"{metrics.precision:.3f}")
        self.metric_vars["f1"].set(f"{metrics.f1:.3f}")
        self.metric_vars["fpr"].set(f"{metrics.false_positive_rate:.3f}")
        self.metric_vars["delay"].set(f"{metrics.detection_delay:.1f}")
        self.metric_vars["alarm"].set(str(int(np.sum(trace.predictions))))

        peak_score = float(np.max(trace.scores)) if trace.scores.size else 0.0
        peak_positive = max((snapshot.positive_routers for snapshot in trace.snapshots), default=0)
        top_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:6]
        top_weight_text = ", ".join(f"{router}={value:.2f}" for router, value in top_weights[:4]) or "-"
        attack_slice = scenario.attack_slice if scenario.attack_slice is not None else "brak"

        self.header_mode_var.set(f"MODE: {mode.upper()}")
        self.card_vars["mode"].set(mode)
        self.card_vars["thresholds"].set(f"alert {detector_cfg.alert_threshold:.2f}\nclear {detector_cfg.clear_threshold:.2f}")
        self.card_vars["logic"].set(
            f"{detector_cfg.alert_windows}/{detector_cfg.clear_windows} okna\nmin {detector_cfg.min_positive_routers} routery"
        )
        self.summary_var.set(
            f"Scenariusz {scenario.name}, attack slice: {attack_slice}, peak score: {peak_score:.2f}, peak dodatnich routerow: {peak_positive}."
        )
        self.diagnostic_var.set(
            f"Tryb: {mode}\n"
            f"Threshold alert/clear: {detector_cfg.alert_threshold:.2f} / {detector_cfg.clear_threshold:.2f}\n"
            f"Histereza: alert {detector_cfg.alert_windows} okna, clear {detector_cfg.clear_windows} okna\n"
            f"Min dodatnich routerow: {detector_cfg.min_positive_routers}\n"
            f"Recall {metrics.recall:.3f}, precision {metrics.precision:.3f}, F1 {metrics.f1:.3f}, FPR {metrics.false_positive_rate:.3f}, delay {metrics.detection_delay:.1f}\n"
            f"Top wagi: {top_weight_text}\n"
            f"Walidacja zapisanego modelu:\n{self._validation_summary()}"
        )
        self._set_status("Run complete", ACCENT)
        self._sb.upd("scenario", f"SCENARIO: {scenario.name}", ACCENT)
        self._sb.upd("mode", f"MODE: {mode}", MUTED if mode == "baseline" else ACCENT)
        self._sb.upd("thresholds", f"THR: {detector_cfg.alert_threshold:.2f} / {detector_cfg.clear_threshold:.2f}", YELLOW)
        self._sb.upd("positives", f"POS: {peak_positive}", GREEN)
        self._sb.upd("peak", f"PEAK: {peak_score:.2f}", PURPLE)

        score_max = max(peak_score, 1.0)
        steps = np.arange(len(trace.scores))
        self.ax_score.clear()
        self._style_axis(self.ax_score, PANEL)
        if scenario.attack_slice is not None:
            start, stop = scenario.attack_slice
            self.ax_score.axvspan(start, stop - 1, color=RED, alpha=0.09)
        self.ax_score.plot(steps, trace.scores, label="score", color=ACCENT, linewidth=2.0)
        self.ax_score.plot(steps, trace.labels * score_max, label="attack label", color=RED, linestyle="--", alpha=0.9)
        self.ax_score.plot(steps, trace.predictions * score_max, label="alarm", color=GREEN, linestyle=":")
        self.ax_score.axhline(detector_cfg.alert_threshold, color=YELLOW, linestyle="-.", label="alert threshold")
        self.ax_score.set_title("Globalny score OFN i alarmy", color=TEXT, fontsize=10, pad=6)
        self.ax_score.set_xlabel("Krok", color=MUTED, fontsize=8)
        self.ax_score.set_ylabel("Score", color=MUTED, fontsize=8)
        legend = self.ax_score.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc="upper left")
        for text in legend.get_texts():
            text.set_color(TEXT)

        self.ax_traffic.clear()
        self._style_axis(self.ax_traffic, PANEL)
        self.ax_traffic.grid(False)
        self.ax_traffic.imshow(scenario.traffic.T, aspect="auto", origin="lower", cmap="magma")
        if scenario.attack_slice is not None:
            start, stop = scenario.attack_slice
            self.ax_traffic.axvspan(start, stop - 1, color="white", alpha=0.08)
        self.ax_traffic.set_title("Macierz ruchu router x czas", color=TEXT, fontsize=9, pad=5)
        self.ax_traffic.set_xlabel("Krok", color=MUTED, fontsize=8)
        self.ax_traffic.set_ylabel("Router", color=MUTED, fontsize=8)

        snap_steps = np.array([snapshot.step for snapshot in trace.snapshots], dtype=np.int32)
        positives = np.array([snapshot.positive_routers for snapshot in trace.snapshots], dtype=np.int32)
        negatives = np.array([snapshot.negative_routers for snapshot in trace.snapshots], dtype=np.int32)
        neutrals = np.array([snapshot.neutral_routers for snapshot in trace.snapshots], dtype=np.int32)
        self.ax_counts.clear()
        self._style_axis(self.ax_counts, PANEL)
        self.ax_counts.plot(snap_steps, positives, label="dodatnie", color=GREEN)
        self.ax_counts.plot(snap_steps, negatives, label="ujemne", color=RED)
        self.ax_counts.plot(snap_steps, neutrals, label="neutralne", color=MUTED)
        self.ax_counts.set_title("Kierunki routerow", color=TEXT, fontsize=9, pad=5)
        self.ax_counts.set_xlabel("Krok", color=MUTED, fontsize=8)
        self.ax_counts.set_ylabel("Liczba", color=MUTED, fontsize=8)
        legend = self.ax_counts.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, loc="upper right")
        for text in legend.get_texts():
            text.set_color(TEXT)

        self.ax_weights.clear()
        self._style_axis(self.ax_weights, DARK)
        weight_subset = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:10]
        labels = [item[0] for item in weight_subset][::-1]
        values = [item[1] for item in weight_subset][::-1]
        colors = [ACCENT if value >= 1.0 else PURPLE for value in values]
        self.ax_weights.barh(labels, values, color=colors, alpha=0.95)
        self.ax_weights.set_title("Top wagi routerow", color=TEXT, fontsize=9, pad=5)
        self.ax_weights.set_xlabel("Waga", color=MUTED, fontsize=8)
        self.ax_weights.set_ylabel("", color=MUTED, fontsize=8)

        self.canvas.draw_idle()
        self.weights_canvas.draw_idle()

    def _on_exit(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    root = tk.Tk()
    app = DashboardApp(root)

    if args.smoke_test:
        app.run_scenario()
        root.update_idletasks()
        root.update()
        root.destroy()
        print("dashboard_ok")
        return

    root.mainloop()


if __name__ == "__main__":
    main()
