from __future__ import annotations

import multiprocessing
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backtest import run_ma_cross_backtest
from data_provider import DataRequest, StockProfile, load_a_share_daily, load_csv, resolve_stock
from indicators import add_indicators
from quant_metrics import QuantMetrics, calculate_quant_metrics
from strategy_engine import StrategySignal, evaluate_signal


def resource_path(relative_path: str) -> str:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base_path / relative_path)


class DataWorker(QThread):
    succeeded = Signal(object, object, object, object, object, object)
    failed = Signal(str)

    def __init__(self, params: dict[str, object]) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            start = date.fromisoformat(str(self.params["start_date"]))
            end = date.fromisoformat(str(self.params["end_date"]))
            fast = int(str(self.params["fast_window"]))
            slow = int(str(self.params["slow_window"]))
            if start >= end:
                raise ValueError("开始日期必须早于结束日期")

            if self.params["data_source"] == "本地 CSV":
                csv_path = str(self.params["csv_path"])
                if not csv_path:
                    raise ValueError("请先选择 CSV 文件")
                df = load_csv(Path(csv_path))
                profile = StockProfile(symbol=Path(csv_path).stem, name=Path(csv_path).stem, source="CSV 文件名")
            else:
                resolved = resolve_stock(str(self.params["symbol"]))
                df = load_a_share_daily(DataRequest(symbol=resolved.symbol, start_date=start, end_date=end, adjust=str(self.params["adjust"])))
                profile = StockProfile(symbol=resolved.symbol, name=resolved.name, source=resolved.source)

            if df.empty:
                raise ValueError("没有读取到行情数据")

            market_df = add_indicators(df)
            backtest_df, metrics = run_ma_cross_backtest(market_df, fast, slow)
            quant_metrics = calculate_quant_metrics(market_df)
            signal = evaluate_signal(market_df)
            self.succeeded.emit(market_df, backtest_df, metrics, signal, profile, quant_metrics)
        except Exception as exc:
            self.failed.emit(str(exc))


class LineChart(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.series: list[tuple[pd.Series, str, str]] = []
        self.setMinimumHeight(260)
        self.setObjectName("chart")

    def set_series(self, series: list[tuple[pd.Series, str, str]]) -> None:
        self.series = series
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#07111f"))
        painter.setPen(QPen(QColor("#e6edf3")))
        painter.drawText(16, 24, self.title)

        left, top, right, bottom = 62, 42, self.width() - 20, self.height() - 34
        painter.setPen(QPen(QColor("#233247"), 1))
        painter.drawLine(left, bottom, right, bottom)
        painter.drawLine(left, top, left, bottom)
        for ratio in (0.25, 0.5, 0.75):
            y = int(top + (bottom - top) * ratio)
            painter.drawLine(left, y, right, y)

        if not self.series:
            painter.setPen(QPen(QColor("#8b949e")))
            painter.drawText(self.rect(), Qt.AlignCenter, "读取行情后显示图表")
            return

        values = pd.concat([item[0] for item in self.series]).dropna()
        if values.empty:
            return
        min_value = float(values.min())
        max_value = float(values.max())
        if min_value == max_value:
            min_value -= 1
            max_value += 1

        painter.setPen(QPen(QColor("#8b949e")))
        painter.drawText(8, top + 4, f"{max_value:.2f}")
        painter.drawText(8, bottom, f"{min_value:.2f}")

        colors = {
            "blue": QColor("#38bdf8"),
            "red": QColor("#ff5a6b"),
            "green": QColor("#2ee59d"),
            "orange": QColor("#f6c85f"),
            "purple": QColor("#a78bfa"),
            "gray": QColor("#94a3b8"),
        }
        legend_x = max(left + 20, right - 260)
        legend_y = top - 20
        for offset, (_data, color_name, label) in enumerate(self.series):
            color = colors.get(color_name, QColor("#38bdf8"))
            x = legend_x + offset * 88
            painter.setPen(QPen(color, 3))
            painter.drawLine(x, legend_y, x + 20, legend_y)
            painter.setPen(QPen(QColor("#c9d1d9")))
            painter.drawText(x + 26, legend_y + 4, label)

        for data, color_name, _label in self.series:
            clean = data.reset_index(drop=True)
            if len(clean) < 2:
                continue
            painter.setPen(QPen(colors.get(color_name, QColor("#38bdf8")), 2))
            previous = None
            for index, value in clean.items():
                if pd.isna(value):
                    previous = None
                    continue
                x = int(left + (right - left) * index / (len(clean) - 1))
                y = int(bottom - (bottom - top) * (float(value) - min_value) / (max_value - min_value))
                if previous is not None:
                    painter.drawLine(previous[0], previous[1], x, y)
                previous = (x, y)


def analysis_method_text(profile: StockProfile | None = None) -> str:
    stock_line = ""
    if profile is not None:
        stock_line = f"当前标的：{profile.name} ({profile.symbol})\n\n"

    return (
        stock_line
        + "为什么使用这些数据\n\n"
        "1. 收盘价：多数策略用收盘价确认趋势，因为它代表当天多空博弈后的最终价格。\n\n"
        "2. MA20 / MA60：20 日均线近似一个月交易节奏，60 日均线近似一个季度趋势，用来判断短中期趋势是否一致。\n\n"
        "3. MACD：用快慢均线差观察动量变化。金叉、死叉和多空排列可以帮助判断上涨或下跌动能是否增强。\n\n"
        "4. RSI：衡量近期涨跌强弱。过热时追高风险上升，超卖时可能有反弹，但必须结合趋势确认。\n\n"
        "5. 成交量：价格上涨如果伴随放量，说明资金参与度更高；放量下跌说明抛压需要警惕。\n\n"
        "6. 波动率：同样的买入信号，在高波动环境下风险更大，所以仓位需要降低。\n\n"
        "7. 最大回撤：回撤反映从阶段高点下跌的幅度，用来判断趋势损伤和持仓风险。\n\n"
        "8. 近 20 日突破/跌破：突破近期高点说明趋势可能延续，跌破近期低点说明风险可能仍在释放。\n\n"
        "综合评分方式\n\n"
        "系统不是只看一个指标，而是把趋势、动量、RSI、量能、波动率、回撤和突破信号综合打分。"
        "分数越高越偏向买入或加仓，分数越低越偏向减仓或卖出。"
        "这个结果用于研究和辅助决策，不构成投资建议。"
    )


def quant_metrics_text(metrics: QuantMetrics) -> str:
    return (
        "行业常用量化指标\n\n"
        f"样本交易日：{metrics.sample_days}\n"
        f"年化收益 CAGR：{format_percent(metrics.cagr)}\n"
        f"年化波动率：{format_percent(metrics.annual_volatility)}\n"
        f"Sharpe Ratio：{format_optional_number(metrics.sharpe)}\n"
        f"Sortino Ratio：{format_optional_number(metrics.sortino)}\n"
        f"最大回撤：{format_percent(metrics.max_drawdown)}\n"
        f"Calmar Ratio：{format_optional_number(metrics.calmar)}\n"
        f"95% 日 VaR：{format_percent(metrics.var_95)}\n"
        f"95% 日 CVaR：{format_percent(metrics.cvar_95)}\n"
        f"日胜率：{format_percent(metrics.win_rate)}\n"
        f"ATR 14：{metrics.atr_14:.2f}\n"
        f"ATR / 收盘价：{format_percent(metrics.atr_pct)}\n"
        f"Kelly 上限仓位：{format_percent(metrics.kelly_fraction)}\n"
        f"波动率目标仓位：{format_percent(metrics.volatility_target_position)}\n\n"
        "指标含义\n\n"
        "Sharpe：衡量单位总波动带来的超额收益，越高说明风险收益比越好。\n\n"
        "Sortino：只惩罚下行波动，比 Sharpe 更关注亏损风险。\n\n"
        "Calmar：年化收益 / 最大回撤，常用于判断策略承受回撤后是否仍有吸引力。\n\n"
        "VaR / CVaR：历史模拟下的尾部风险。VaR 看较差 5% 情况的亏损阈值，CVaR 看尾部平均亏损。\n\n"
        "ATR：真实波幅均值，常用于止损距离和仓位控制。\n\n"
        "Kelly：基于胜率和盈亏比估算理论仓位。这里做了 25% 上限，避免公式过度激进。\n\n"
        "波动率目标仓位：假设目标年化波动率为 20%，波动越高，建议仓位越低。"
    )


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def format_optional_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


def fintech_stylesheet() -> str:
    return """
    QWidget {
        background: #07111f;
        color: #e6edf3;
        font-size: 14px;
    }
    QGroupBox {
        border: 1px solid #1f6feb;
        border-radius: 8px;
        margin-top: 14px;
        padding: 14px;
        background: #0d1829;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 14px;
        padding: 0 6px;
        color: #7dd3fc;
    }
    QLabel {
        color: #e6edf3;
        background: transparent;
    }
    QLineEdit, QComboBox {
        background: #0b1323;
        border: 1px solid #27364f;
        border-radius: 6px;
        padding: 7px 9px;
        color: #f8fafc;
        selection-background-color: #1f6feb;
    }
    QLineEdit:focus, QComboBox:focus {
        border: 1px solid #38bdf8;
    }
    QPushButton {
        background: #1f6feb;
        border: 1px solid #58a6ff;
        border-radius: 7px;
        padding: 8px 14px;
        color: #ffffff;
        font-weight: 700;
    }
    QPushButton:hover {
        background: #388bfd;
    }
    QPushButton:disabled {
        background: #253247;
        border-color: #303d52;
        color: #8b949e;
    }
    QTabWidget::pane {
        border: 1px solid #1f2a3d;
        background: #07111f;
        top: -1px;
    }
    QTabBar::tab {
        background: #0d1829;
        color: #9fb3c8;
        border: 1px solid #1f2a3d;
        padding: 8px 18px;
        min-width: 78px;
    }
    QTabBar::tab:selected {
        color: #ffffff;
        background: #10233d;
        border-bottom-color: #38bdf8;
    }
    QTextEdit, QTableWidget {
        background: #07111f;
        border: 1px solid #1f2a3d;
        color: #dbeafe;
        selection-background-color: #1f6feb;
    }
    QHeaderView::section {
        background: #10233d;
        color: #dbeafe;
        border: 1px solid #1f2a3d;
        padding: 6px;
    }
    QScrollBar:vertical, QScrollBar:horizontal {
        background: #07111f;
        width: 12px;
        height: 12px;
    }
    QScrollBar::handle {
        background: #27364f;
        border-radius: 6px;
    }
    QLabel#summaryCard {
        border: 1px solid #1f6feb;
        border-radius: 8px;
        padding: 12px;
        background: #0d1829;
        color: #f8fafc;
        font-weight: 700;
    }
    """


class QuantMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("量化股票分析平台")
        self.resize(1220, 780)
        self.setWindowIcon(QIcon(resource_path("assets/quantplatform-icon.png")))
        self.worker: DataWorker | None = None
        self.market_df = pd.DataFrame()
        self.backtest_df = pd.DataFrame()

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        controls = QGroupBox("数据与策略")
        controls_layout = QGridLayout(controls)
        self.data_source = QComboBox()
        self.data_source.addItems(["网络行情", "本地 CSV"])
        self.symbol = QLineEdit("贵州茅台")
        self.start_date = QLineEdit((date.today() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"))
        self.end_date = QLineEdit(date.today().strftime("%Y-%m-%d"))
        self.adjust = QComboBox()
        self.adjust.addItems(["qfq", "hfq", ""])
        self.csv_path = QLineEdit()
        self.csv_path.setReadOnly(True)
        self.fast_window = QLineEdit("20")
        self.slow_window = QLineEdit("60")

        csv_button = QPushButton("选择 CSV")
        csv_button.clicked.connect(self.choose_csv)
        self.analyze_button = QPushButton("读取并分析")
        self.analyze_button.clicked.connect(self.load_data)

        fields = [
            ("数据源", self.data_source),
            ("名称/代码", self.symbol),
            ("开始日期", self.start_date),
            ("结束日期", self.end_date),
            ("复权", self.adjust),
            ("短均线", self.fast_window),
            ("长均线", self.slow_window),
        ]
        for index, (label, widget) in enumerate(fields):
            controls_layout.addWidget(QLabel(label), index // 4, (index % 4) * 2)
            controls_layout.addWidget(widget, index // 4, (index % 4) * 2 + 1)
        controls_layout.addWidget(csv_button, 2, 0)
        controls_layout.addWidget(self.csv_path, 2, 1, 1, 5)
        controls_layout.addWidget(self.analyze_button, 2, 6, 1, 2)
        layout.addWidget(controls)

        self.summary_labels = []
        summary = QHBoxLayout()
        for text in ["股票\n-", "最新收盘\n-", "量化建议\n-", "建议仓位\n-", "评分/置信度\n-", "风险等级\n-"]:
            label = QLabel(text)
            label.setObjectName("summaryCard")
            label.setAlignment(Qt.AlignCenter)
            summary.addWidget(label)
            self.summary_labels.append(label)
        layout.addLayout(summary)

        splitter = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()
        self.price_chart = LineChart("收盘价 / MA20 / MA60")
        self.indicator_chart = LineChart("RSI / MACD")
        self.backtest_chart = LineChart("买入持有 / 均线策略净值")
        self.signal_text = QTextEdit()
        self.signal_text.setReadOnly(True)
        self.signal_text.setText("读取行情后显示量化建议")
        self.method_text = QTextEdit()
        self.method_text.setReadOnly(True)
        self.method_text.setText(analysis_method_text())
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setText("读取行情后显示行业常用量化指标")
        self.table = QTableWidget()
        self.tabs.addTab(self.price_chart, "走势")
        self.tabs.addTab(self.indicator_chart, "指标")
        self.tabs.addTab(self.backtest_chart, "回测")
        self.tabs.addTab(self.signal_text, "建议")
        self.tabs.addTab(self.metrics_text, "量化指标")
        self.tabs.addTab(self.method_text, "方法")
        self.tabs.addTab(self.table, "数据")
        splitter.addWidget(self.tabs)
        layout.addWidget(splitter)

        self.status_label = QLabel("准备就绪")
        layout.addWidget(self.status_label)
        self.setCentralWidget(root)

    def choose_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择行情 CSV", "", "CSV files (*.csv);;All files (*.*)")
        if path:
            self.data_source.setCurrentText("本地 CSV")
            self.csv_path.setText(path)

    def load_data(self) -> None:
        params = {
            "data_source": self.data_source.currentText(),
            "symbol": self.symbol.text(),
            "start_date": self.start_date.text(),
            "end_date": self.end_date.text(),
            "adjust": self.adjust.currentText(),
            "csv_path": self.csv_path.text(),
            "fast_window": self.fast_window.text(),
            "slow_window": self.slow_window.text(),
        }
        self.status_label.setText("正在读取数据...")
        self.analyze_button.setEnabled(False)
        self.worker = DataWorker(params)
        self.worker.succeeded.connect(self.render_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(lambda: self.analyze_button.setEnabled(True))
        self.worker.start()

    def render_result(
        self,
        market_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        metrics: dict[str, float],
        signal: StrategySignal,
        profile: StockProfile,
        quant_metrics: QuantMetrics,
    ) -> None:
        self.market_df = market_df
        self.backtest_df = backtest_df
        self.render_summary(market_df, signal, profile)
        self.render_signal(signal, profile, quant_metrics)
        self.metrics_text.setText(quant_metrics_text(quant_metrics))
        self.method_text.setText(analysis_method_text(profile))
        self.render_charts(market_df, backtest_df)
        self.render_table(market_df)
        self.status_label.setText(f"完成：{profile.name} {profile.symbol}，{len(market_df)} 条行情")

    def render_summary(self, df: pd.DataFrame, signal: StrategySignal, profile: StockProfile) -> None:
        latest = df.iloc[-1]
        previous_close = df["close"].iloc[-2] if len(df) > 1 else latest["close"]
        change = latest["close"] / previous_close - 1 if previous_close else 0
        values = [
            f"股票\n{profile.name} ({profile.symbol})",
            f"最新收盘\n{latest['close']:.2f} ({change:.2%})",
            f"量化建议\n{signal.action}",
            f"建议仓位\n{signal.position_hint}",
            f"评分/置信度\n{signal.score:.1f} / {signal.confidence}",
            f"风险等级\n{signal.risk_level}",
        ]
        for label, value in zip(self.summary_labels, values):
            label.setText(value)

    def render_signal(self, signal: StrategySignal, profile: StockProfile, quant_metrics: QuantMetrics) -> None:
        reasons = "\n".join(f"- {reason}" for reason in signal.reasons)
        warnings = "\n".join(f"- {warning}" for warning in signal.warnings) or "- 暂无额外风险提示"
        self.signal_text.setText(
            f"股票：{profile.name} ({profile.symbol})\n"
            f"名称来源：{profile.source}\n\n"
            f"动作：{signal.action}\n"
            f"建议仓位：{signal.position_hint}\n"
            f"综合评分：{signal.score:.1f}\n"
            f"置信度：{signal.confidence}\n"
            f"风险等级：{signal.risk_level}\n\n"
            f"风控参考\n"
            f"- 波动率目标仓位：{format_percent(quant_metrics.volatility_target_position)}\n"
            f"- Kelly 上限仓位：{format_percent(quant_metrics.kelly_fraction)}\n"
            f"- 95% 日 VaR：{format_percent(quant_metrics.var_95)}\n\n"
            f"触发原因\n{reasons}\n\n"
            f"风险提示\n{warnings}\n\n"
            "说明：以上结果由趋势、动量、均值回归、波动率、回撤和量能规则综合生成，仅用于量化研究，不构成投资建议。"
        )

    def render_charts(self, market_df: pd.DataFrame, backtest_df: pd.DataFrame) -> None:
        market = market_df.tail(180)
        self.price_chart.set_series([(market["close"], "blue", "收盘价"), (market["ma20"], "orange", "MA20"), (market["ma60"], "red", "MA60")])
        self.indicator_chart.set_series([(market["rsi14"], "blue", "RSI14"), (market["macd_dif"], "purple", "DIF"), (market["macd_dea"], "red", "DEA")])
        backtest = backtest_df.tail(260)
        self.backtest_chart.set_series([(backtest["asset_curve"], "gray", "买入持有"), (backtest["strategy_curve"], "green", "均线策略")])

    def render_table(self, df: pd.DataFrame) -> None:
        columns = ["date", "open", "high", "low", "close", "volume", "rsi14", "macd_dif"]
        display = df.sort_values("date", ascending=False).head(300)
        self.table.setRowCount(len(display))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        for row_index, (_, row) in enumerate(display.iterrows()):
            values = [
                row["date"].strftime("%Y-%m-%d"),
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                f"{row['volume']:.0f}",
                f"{row['rsi14']:.2f}",
                f"{row['macd_dif']:.3f}",
            ]
            for column_index, value in enumerate(values):
                self.table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def show_error(self, message: str) -> None:
        self.status_label.setText("读取失败")
        QMessageBox.critical(self, "分析失败", message)


def main() -> int:
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path("assets/quantplatform-icon.png")))
    app.setStyleSheet(fintech_stylesheet())
    window = QuantMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
