"""Генерация текстового отчёта на русском языке по периодам.

Функции строят человеко‑понятный отчёт, используя:
- агрегаты продаж
- сравнение выходные/будни
- лучший/худший день
- аномальные дни (по медиане и MAD)
- ТОП факторов (SHAP) за период и для конкретных дней
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def _fmt_idr(x: float) -> str:
    try:
        return f"{int(round(x)):,.0f} IDR".replace(",", " ")
    except Exception:
        return str(x)


def _weekend_effect(df: pd.DataFrame) -> Tuple[float, float, float]:
    wkd = df[df["is_weekend"] == 1]["total_sales"].mean()
    wk = df[df["is_weekend"] == 0]["total_sales"].mean()
    if pd.isna(wkd):
        wkd = 0.0
    if pd.isna(wk):
        wk = 0.0
    delta = 0.0 if wk == 0 else (wkd - wk) / wk * 100.0
    return float(wkd), float(wk), float(delta)


def _best_worst(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    day_sales = df.groupby("date", as_index=False)["total_sales"].sum()
    best = day_sales.sort_values("total_sales", ascending=False).head(1).iloc[0]
    worst = day_sales.sort_values("total_sales", ascending=True).head(1).iloc[0]
    return best, worst


def _anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # Аномалии по медиане и медианному отклонению (MAD)
    day_sales = df.groupby("date", as_index=False)["total_sales"].sum()
    x = day_sales["total_sales"].values
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1.0
    scores = 0.6745 * (x - med) / mad
    day_sales["anomaly_score"] = scores
    # сильные отклонения
    return day_sales[(np.abs(day_sales["anomaly_score"]) >= 3.5)]


def _format_top_factors(top_list: List[Dict[str, float]], limit: int = 5) -> str:
    lines = []
    for i, item in enumerate(top_list[:limit], start=1):
        lines.append(f"{i}. {item['feature']}: {item['impact_percent']}%")
    return "\n".join(lines) if lines else "нет значимых факторов"


def build_text_report(
    period: str,
    restaurant_id: int | None,
    period_df: pd.DataFrame,
    preds_sum: float,
    top_factors_period: List[Dict[str, float]],
) -> str:
    total_sales = float(period_df["total_sales"].sum())
    orders = int(period_df["orders_count"].sum()) if "orders_count" in period_df.columns else 0
    aov = (total_sales / orders) if orders else 0.0

    # Выходные/будни
    wkd_avg, wk_avg, wkd_effect = _weekend_effect(period_df)

    # Лучший/худший день
    best, worst = _best_worst(period_df)

    # Аномалии
    anom = _anomalies(period_df)

    lines: List[str] = []
    lines.append(f"🏪 Отчёт по ресторану: {restaurant_id if restaurant_id is not None else 'все рестораны'}")
    lines.append(f"🗓️ Период: {period}")
    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Резюме
    lines.append("📊 1. Исполнительное резюме")
    lines.append("----------------------------------------")
    lines.append(f"💰 Выручка за период: {_fmt_idr(total_sales)}")
    lines.append(f"📦 Заказы: {orders}")
    lines.append(f"💵 Средний чек: {_fmt_idr(aov)}")
    lines.append(f"🤖 Прогнозная выручка (сумма): {_fmt_idr(preds_sum)}")
    lines.append("\n")

    # Тренды
    lines.append("📈 2. Продажи и тренды")
    lines.append("----------------------------------------")
    lines.append(f"🗓️ Выходные vs Будни: {_fmt_idr(wkd_avg)} vs {_fmt_idr(wk_avg)} (эффект: {wkd_effect:.1f}%)")
    lines.append(f"🏆 Лучший день: {best['date'].date()} — {_fmt_idr(best['total_sales'])}")
    lines.append(f"📉 Худший день: {worst['date'].date()} — {_fmt_idr(worst['total_sales'])}")
    lines.append("\n")

    # ТОП‑факторы периода
    lines.append("🔍 3. ТОП‑факторы (по SHAP, вклад в %)")
    lines.append("----------------------------------------")
    lines.append(_format_top_factors(top_factors_period, limit=10))
    lines.append("\n")

    # Аномальные дни
    lines.append("🚨 4. Аномалии")
    lines.append("----------------------------------------")
    if anom.empty:
        lines.append("Аномальные дни не обнаружены по порогу |score| ≥ 3.5")
    else:
        for _, row in anom.sort_values("anomaly_score", key=lambda s: np.abs(s), ascending=False).iterrows():
            lines.append(f"{row['date'].date()}: {_fmt_idr(row['total_sales'])} (score={row['anomaly_score']:.2f})")
    lines.append("\n")

    lines.append("✅ Анализ завершён")
    return "\n".join(lines)