from __future__ import annotations

from typing import Optional, Dict, Tuple
from datetime import date
import sqlite3
import pandas as pd
import os

from app.report_basic import (
    build_basic_report,
    build_marketing_report,
    build_quality_report,
)
from etl.data_loader import get_engine
import numpy as np
import re
from ml.inference import load_artifacts, _resolve_preprocessed_feature_groups
from etl.feature_engineering import load_holidays_df
import shap
import json


def _fmt_idr(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{int(round(float(x))):,} IDR".replace(",", " ")
    except Exception:
        return str(x)


def _fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}%"


def _fmt_rate(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def _hms_from_minutes(total_minutes: float) -> str:
    m = max(total_minutes, 0)
    hours = int(m // 60)
    minutes = int(m % 60)
    seconds = int(round((m - int(m)) * 60))
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _sec_to_minutes(avg_seconds: float) -> float:
    return float(avg_seconds) / 60.0


def _section1_exec(basic: Dict) -> str:
    es = basic["executive_summary"]
    grab = es["by_platform"]["grab"]
    gojek = es["by_platform"]["gojek"]

    total_rev = _fmt_idr(es["revenue_total"])
    grab_rev = _fmt_idr(grab.get("sales"))
    gojek_rev = _fmt_idr(gojek.get("sales"))

    total_orders = es.get("orders_total") or 0
    # Optional enriched fields
    fake = es.get("fake_orders", {})
    canc = es.get("cancellations", {})
    lost = es.get("lost_orders", {})
    succ = es.get("successful_orders", {})
    aov = es.get("aov", {})

    lines = []
    lines.append("📊 1. ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ")
    lines.append("—" * 72)
    lines.append(f"💰 Общая выручка: {total_rev} (GRAB: {grab_rev} + GOJEK: {gojek_rev})")
    lines.append(f"📦 Общие заказы: {total_orders}")
    lines.append(f"   ├── 📱 GRAB: {int(grab.get('orders') or 0)} (успешно: {succ.get('grab','—')}, отмены: {canc.get('grab','—')}, потери: {lost.get('grab','—')}, fake: {fake.get('grab','—')})")
    lines.append(f"   └── 🛵 GOJEK: {int(gojek.get('orders') or 0)} (успешно: {succ.get('gojek','—')}, отмены: {canc.get('gojek','—')}, потери: {lost.get('gojek','—')}, fake: {fake.get('gojek','—')})")
    # AOVs
    if aov:
        lines.append(f"💵 Средний чек (успешные): общий { _fmt_idr(aov.get('total')) }; GRAB { _fmt_idr(aov.get('grab')) }; GOJEK { _fmt_idr(aov.get('gojek')) }")
    # Daily revenue
    drw = es.get('daily_revenue_workdays_avg')
    if drw is not None:
        lines.append(f"📊 Дневная выручка: {_fmt_idr(drw)} (средняя по рабочим дням)")
    # Rating
    rat = es.get('rating_avg_total')
    if rat:
        lines.append(f"⭐ Средний рейтинг: {_fmt_rate(float(rat), 2)}/5.0")
    # Clients
    cli = es.get('clients', {})
    if cli:
        tot = cli.get('total_unique')
        lines.append(f"👥 Обслужено клиентов: {tot if tot is not None else '—'}")
        g = cli.get('grab', {})
        j = cli.get('gojek', {})
        lines.append(f"   ├── 📱 GRAB: {g.get('total','—')} (новые: {g.get('new','—')}, повторные: {g.get('repeated','—')}, реактивированные: {g.get('reactivated','—')})")
        lines.append(f"   └── 🛵 GOJEK: {j.get('new','—') + j.get('active','—') + j.get('returned','—') if all(isinstance(j.get(k), int) for k in ['new','active','returned']) else '—'} (новые: {j.get('new','—')}, активные: {j.get('active','—')}, возвратившиеся: {j.get('returned','—')})")
        if tot is not None:
            lines.append(f"   💡 Общий охват: {tot} уникальных клиентов")
    # Marketing budget
    mb = es.get('marketing_budget', {})
    if mb:
        total_b = mb.get('total') or 0.0
        lines.append(f"💸 Маркетинговый бюджет: {_fmt_idr(total_b)} ({_fmt_pct(mb.get('share_of_revenue_pct'))} от выручки)")
        lines.append("📊 Детализация маркетинговых затрат:")
        lines.append("   ┌─ 📱 GRAB:")
        lines.append(f"   │  💰 Бюджет: {_fmt_idr(mb.get('grab'))}")
        # Additional ratios require per-platform revenue; already printed above implicitly; keep budget split concise
        lines.append("   └─ 🛵 GOJEK:")
        lines.append(f"      💰 Бюджет: {_fmt_idr(mb.get('gojek'))}")
    # ROAS summary
    ro = es.get('roas', {})
    if ro:
        lines.append("")
        lines.append("🎯 ROAS АНАЛИЗ:")
        lines.append(f"├── 📱 GRAB: {_fmt_rate(ro.get('grab'))}x")
        lines.append(f"├── 🛵 GOJEK: {_fmt_rate(ro.get('gojek'))}x")
        lines.append(f"└── 🎯 ОБЩИЙ: {_fmt_rate(ro.get('total'))}x")
    return "\n".join(lines)


def _dataset_version_banner() -> str:
    try:
        project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        default_artifacts = os.path.join(project_root, "ml", "artifacts")
        metrics_path = os.path.join(os.getenv("ML_ARTIFACT_DIR", default_artifacts), "metrics.json")
        if not os.path.exists(metrics_path):
            return ""
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        h = str(m.get("dataset_hash",""))
        rows = m.get("dataset_rows")
        ts = m.get("run_at_utc")
        champ = m.get("champion")
        short = h[:10] if h else ""
        return f"Версия датасета: {short} · строк: {rows} · тренировался: {ts} · модель: {champ}"
    except Exception:
        return ""


def _section2_trends(basic: Dict) -> str:
    st = basic["sales_trends"]
    monthly = st.get("monthly", {})
    lines = []
    lines.append("📈 2. АНАЛИЗ ПРОДАЖ И ТРЕНДОВ")
    lines.append("—" * 72)
    lines.append("📊 Динамика по месяцам:")
    for ym in sorted(monthly.keys()):
        m = monthly[ym]
        lines.append(
            f"  {ym}: {_fmt_idr(m['total_sales'])} ({m['days']} дней, {_fmt_idr(m['avg_per_day'])}/день)"
        )
    w = st.get("weekend_vs_weekday", {})
    lines.append("")
    lines.append("🗓️ Выходные vs Будни:")
    lines.append(f"  📅 Средние продажи в выходные: {_fmt_idr(w.get('weekend_avg'))}")
    lines.append(f"  📅 Средние продажи в будни: {_fmt_idr(w.get('weekday_avg'))}")
    lines.append(f"  📊 Эффект выходных: {_fmt_pct(w.get('effect_pct'))}")

    best = st.get("best_day")
    worst = st.get("worst_day")
    if best:
        gp = best.get("by_platform", {})
        lines.append("📊 АНАЛИЗ РАБОЧИХ ДНЕЙ:")
        lines.append(
            f"🏆 Лучший день: {best['date']} - {_fmt_idr(best['total_sales'])}"
        )
        lines.append(
            f"   💰 GRAB: {_fmt_idr(gp.get('grab',{}).get('sales'))} ({int(gp.get('grab',{}).get('orders') or 0)} заказов) | "
            f"GOJEK: {_fmt_idr(gp.get('gojek',{}).get('sales'))} ({int(gp.get('gojek',{}).get('orders') or 0)} заказов)"
        )
    if worst:
        gp = worst.get("by_platform", {})
        lines.append(
            f"📉 Худший день: {worst['date']} - {_fmt_idr(worst['total_sales'])}"
        )
        lines.append(
            f"   💰 GRAB: {_fmt_idr(gp.get('grab',{}).get('sales'))} | GOJEK: {_fmt_idr(gp.get('gojek',{}).get('sales'))}"
        )
    return "\n".join(lines)


def _section4_marketing(mkt: Dict) -> str:
    f = mkt.get("funnel_grab", {})
    rm = mkt.get("roas_by_month", {})
    lines = []
    lines.append("📈 4. МАРКЕТИНГОВАЯ ЭФФЕКТИВНОСТЬ И ВОРОНКА")
    lines.append("—" * 72)
    lines.append("📊 Маркетинговая воронка (только GRAB):")
    lines.append(f"  👁️ Показы рекламы: {int(f.get('impressions') or 0)}")
    lines.append(
        f"  🔗 Посещения меню: {int(f.get('menu_visits') or 0)} (CTR: {_fmt_pct((f.get('ctr') or 0)*100)})"
    )
    lines.append(
        f"  🛒 Добавления в корзину: {int(f.get('add_to_cart') or 0)} (конверсия: {_fmt_pct((f.get('conv_click_to_order') or 0)*100)} от кликов)"
    )
    lines.append(
        f"  📦 Заказы от рекламы: {int(f.get('ads_orders') or 0)} (конверсия: {_fmt_pct((f.get('conv_cart_to_order') or 0)*100)} от корзины)"
    )
    lines.append("")
    lines.append("  📊 КЛЮЧЕВЫЕ КОНВЕРСИИ:")
    lines.append(f"  • 🎯 Показ → Заказ: {_fmt_pct((f.get('show_to_order') or 0)*100)}")
    lines.append(f"  • 🔗 Клик → Заказ: {_fmt_pct((f.get('conv_click_to_order') or 0)*100)}")
    lines.append(f"  • 🛒 Корзина → Заказ: {_fmt_pct((f.get('conv_cart_to_order') or 0)*100)}")
    lines.append("")
    bouncers = int(max((f.get('menu_visits') or 0) - (f.get('add_to_cart') or 0), 0))
    aband = int(max((f.get('add_to_cart') or 0) - (f.get('ads_orders') or 0), 0))
    lines.append("  📉 ДЕТАЛЬНЫЙ АНАЛИЗ ВОРОНКИ:")
    lines.append(
        f"  • 💔 Доля ушедших без покупки: {_fmt_pct((f.get('bounce_rate') or 0)*100)} ({bouncers} ушли без покупки)"
    )
    lines.append(
        f"  • 🛒 Доля неоформленных корзин: {_fmt_pct((f.get('abandoned_carts_rate') or 0)*100)} ({aband} добавили, но не купили)"
    )
    lines.append("")
    lines.append("  💰 ПОТЕНЦИАЛ ОПТИМИЗАЦИИ ВОРОНКИ:")
    up = f.get("uplift_estimations", {})
    lines.append(
        f"  • 📈 Снижение доли ушедших на 10%: {_fmt_idr(up.get('reduce_bounce_10_pct_revenue'))}"
    )
    lines.append(
        f"  • 🛒 Устранение неоформленных корзин: {_fmt_idr(up.get('eliminate_abandoned_revenue'))}"
    )
    lines.append(
        f"  • 🎯 Общий потенциал: {_fmt_idr(up.get('total_uplift'))}"
    )
    lines.append("")
    lines.append("💸 Стоимость привлечения (GRAB):")
    lines.append(
        f"  💰 CPC: {_fmt_idr(f.get('cpc'))} (расчёт: бюджет ÷ посещения меню)" 
    )
    lines.append(
        f"  💰 CPA: {_fmt_idr(f.get('cpa'))} (расчёт: бюджет ÷ заказы от рекламы)"
    )
    return "\n".join(lines)


def _section5_finance(fin: Dict) -> str:
    lines = []
    lines.append("5. 💳 ФИНАНСОВЫЕ ПОКАЗАТЕЛИ")
    lines.append("—" * 72)
    payouts = fin.get("payouts", {})
    total_payouts = payouts.get("total") or 0.0
    grab_p = payouts.get("grab") or 0.0
    gojek_p = payouts.get("gojek") or 0.0
    grab_pct = (grab_p / total_payouts * 100.0) if total_payouts else None
    gojek_pct = (gojek_p / total_payouts * 100.0) if total_payouts else None
    lines.append("💰 Выплаты:")
    lines.append(f"   ├── 📱 GRAB: {_fmt_idr(grab_p)} ({_fmt_pct(grab_pct)})")
    lines.append(f"   ├── 🛵 GOJEK: {_fmt_idr(gojek_p)} ({_fmt_pct(gojek_pct)})")
    lines.append(f"   └── 💎 Общие выплаты: {_fmt_idr(total_payouts)}")

    ad_sales = fin.get("ad_sales")
    ad_share = (fin.get("ad_sales_share") or 0.0) * 100.0
    lines.append("📊 Рекламная эффективность:")
    lines.append(f"   ├── 💰 Общие рекламные продажи: {_fmt_idr(ad_sales)}")
    lines.append(f"   ├── 📈 Доля от общих продаж: {_fmt_pct(ad_share)}")
    lines.append(
        f"   ├── 🎯 GRAB ROAS: {_fmt_rate(fin.get('roas',{}).get('grab'))}x"
    )
    lines.append(
        f"   └── 🎯 GOJEK ROAS: {_fmt_rate(fin.get('roas',{}).get('gojek'))}x"
    )

    tr = fin.get("take_rate", {})
    net_roas = fin.get("net_roas", {})
    lines.append("")
    lines.append("Дополнительно:")
    lines.append(
        f"   • Take rate (доля комиссий и удержаний): GRAB { _fmt_pct((tr.get('grab') or 0.0)*100) }, GOJEK { _fmt_pct((tr.get('gojek') or 0.0)*100) }"
    )
    lines.append(
        f"   • Чистый ROAS: GRAB {_fmt_rate(net_roas.get('grab'))}x; GOJEK {_fmt_rate(net_roas.get('gojek'))}x"
    )
    contrib = fin.get("contribution_per_ad_order_grab")
    if contrib is not None:
        lines.append(
            f"   • Юнит‑экономика рекламного заказа (GRAB): {_fmt_idr(contrib)}"
        )
    return "\n".join(lines)


def _parse_time_to_minutes(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 60 + int(m) + int(sec) / 60.0
        if len(parts) == 2:
            h, m = parts
            return int(h) * 60 + int(m)
        # fallback numeric
        return float(s)
    except Exception:
        return None


def _section6_operations(period: str, restaurant_id: int) -> str:
    eng = get_engine()
    start_str, end_str = period.split("_")
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)

    # GRAB driver_waiting_time JSON average (seconds -> minutes if needed)
    qg = (
        "SELECT driver_waiting_time FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ? "
        "AND driver_waiting_time IS NOT NULL"
    )
    g = pd.read_sql_query(qg, eng, params=(restaurant_id, start_str, end_str))
    grab_wait_vals = []
    for v in g['driver_waiting_time'].dropna().tolist():
        try:
            if isinstance(v, str) and v.strip().startswith('{'):
                d = pd.read_json(pd.io.common.StringIO(v), typ='series') if False else None
                # Fallback manual parse
                import json as _json
                d = _json.loads(v)
                for k in ('avg','average','minutes','mean'):
                    if k in d:
                        val = float(d[k])
                        # heuristic: if seconds, convert to minutes
                        grab_wait_vals.append(val/60.0 if val > 60 else val)
                        break
            elif isinstance(v, (int, float)):
                val = float(v)
                grab_wait_vals.append(val/60.0 if val > 60 else val)
        except Exception:
            pass
    grab_wait_avg = sum(grab_wait_vals)/len(grab_wait_vals) if grab_wait_vals else None

    # GOJEK times
    qj = (
        "SELECT accepting_time, preparation_time, delivery_time, driver_waiting, close_time, stat_date "
        "FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?"
    )
    j = pd.read_sql_query(qj, eng, params=(restaurant_id, start_str, end_str))
    acc = pd.Series([_parse_time_to_minutes(x) for x in j['accepting_time'] if pd.notna(x)])
    prep = pd.Series([_parse_time_to_minutes(x) for x in j['preparation_time'] if pd.notna(x)])
    delv = pd.Series([_parse_time_to_minutes(x) for x in j['delivery_time'] if pd.notna(x)])
    drvw = pd.to_numeric(j['driver_waiting'], errors='coerce').dropna()

    # cancellations
    Cg = pd.read_sql_query(
        "SELECT SUM(cancelled_orders) c FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
        eng, params=(restaurant_id, start_str, end_str)
    ).iloc[0]['c'] or 0
    Cj_row = pd.read_sql_query(
        "SELECT SUM(cancelled_orders) c, SUM(lost_orders) l FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
        eng, params=(restaurant_id, start_str, end_str)
    ).iloc[0]
    Cj = Cj_row['c'] or 0; Lj = Cj_row['l'] or 0
    orders_total = (
        (pd.read_sql_query("SELECT SUM(orders) o FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?", eng, params=(restaurant_id, start_str, end_str)).iloc[0]['o'] or 0)
        + (pd.read_sql_query("SELECT SUM(orders) o FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?", eng, params=(restaurant_id, start_str, end_str)).iloc[0]['o'] or 0)
    )
    cancel_rate = ((Cg + Cj) / orders_total * 100.0) if orders_total else None

    # outages events > 1 hour
    events = []
    # GRAB offline_rate in minutes
    og = pd.read_sql_query(
        "SELECT stat_date, offline_rate FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ? AND offline_rate IS NOT NULL",
        eng, params=(restaurant_id, start_str, end_str)
    )
    for _, row in og.iterrows():
        mins = float(row['offline_rate'] or 0)
        if mins >= 60.0:
            events.append((pd.to_datetime(row['stat_date']).date(), 'GRAB', mins/60.0))
    # GOJEK close_time HH:MM:SS
    oj = pd.read_sql_query(
        "SELECT stat_date, close_time FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
        eng, params=(restaurant_id, start_str, end_str)
    )
    for _, row in oj.iterrows():
        ct = str(row['close_time']) if pd.notna(row['close_time']) else ''
        parts = ct.split(':')
        seconds = None
        try:
            if len(parts) == 3:
                h, m, s = parts
                seconds = int(h)*3600 + int(m)*60 + int(s)
        except Exception:
            seconds = None
        if seconds and seconds >= 3600:
            events.append((pd.to_datetime(row['stat_date']).date(), 'GOJEK', seconds/3600.0))

    # Potential losses
    # average hourly revenue by platform
    # platform sales
    sg = pd.read_sql_query(
        "SELECT SUM(sales) s FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
        eng, params=(restaurant_id, start_str, end_str)
    ).iloc[0]['s'] or 0.0
    sj = pd.read_sql_query(
        "SELECT SUM(sales) s FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
        eng, params=(restaurant_id, start_str, end_str)
    ).iloc[0]['s'] or 0.0
    num_days = (end - start).days + 1
    hr_g = (sg / (num_days*24.0)) if num_days>0 else 0.0
    hr_j = (sj / (num_days*24.0)) if num_days>0 else 0.0

    total_loss_g = sum((hrs*hr_g) for (d,plat,hrs) in events if plat=='GRAB')
    total_loss_j = sum((hrs*hr_j) for (d,plat,hrs) in events if plat=='GOJEK')

    lines = []
    lines.append("6. ⏰ ОПЕРАЦИОННЫЕ МЕТРИКИ")
    lines.append("—" * 72)
    lines.append("🟢 GRAB:")
    lines.append(f"└── ⏰ Время ожидания водителей: {_fmt_rate(grab_wait_avg,1)} мин")
    lines.append("")
    lines.append("🟠 GOJEK:")
    lines.append(f"├── ⏱️ Время приготовления: {_fmt_rate(prep.mean() if not prep.empty else None,1)} мин")
    lines.append(f"├── 🚗 Время доставки: {_fmt_rate(delv.mean() if not delv.empty else None,1)} мин  ")
    lines.append(f"└── ⏰ Время ожидания водителей: {_fmt_rate(drvw.mean() if not drvw.empty else None,1)} мин")
    lines.append("")
    lines.append("⚠️ ОПЕРАЦИОННАЯ ЭФФЕКТИВНОСТЬ")
    lines.append("—" * 72)
    lines.append("🚫 Отмененные заказы:")
    lines.append(f"   ├── 📱 GRAB: {int(Cg)} заказа")
    lines.append(f"   └── 🛵 GOJEK: {int(Cj)} заказа")
    lines.append(f"   💡 Всего отмененных: {int(Cg+Cj)} заказов ({_fmt_pct(cancel_rate)})")
    lines.append("")
    if events:
        total_loss = total_loss_g + total_loss_j
        total_sales = sg + sj
        loss_pct = (total_loss/total_sales*100.0) if total_sales else None
        lines.append("🔧 ОПЕРАЦИОННЫЕ СБОИ ПЛАТФОРМ:")
        # aggregate durations per platform
        dur_g = sum(hrs for (_,plat,hrs) in events if plat=='GRAB')
        dur_j = sum(hrs for (_,plat,hrs) in events if plat=='GOJEK')
        from datetime import timedelta
        def hms_from_hours(h):
            h_int = int(h)
            m = int((h - h_int)*60)
            s = int(round(((h - h_int)*60 - m)*60))
            return f"{h_int}:{m:02d}:{s:02d}"
        lines.append(f"├── 📱 GRAB: {len([1 for _,p,_ in events if p=='GRAB'])} критичных дня ({hms_from_hours(dur_g)} общее время)")
        lines.append(f"├── 🛵 GOJEK: {len([1 for _,p,_ in events if p=='GOJEK'])} критичных дня ({hms_from_hours(dur_j)} общее время)")
        lines.append(f"└── 💸 Потенциальные потери: {_fmt_idr(total_loss)} ({_fmt_pct(loss_pct)})")
        if events:
            lines.append("")
            lines.append("🚨 КРИТИЧЕСКИЕ СБОИ (>1 часа):")
            # sort by date
            for d, plat, hrs in sorted(events, key=lambda x: x[0]):
                loss = hrs*(hr_g if plat=='GRAB' else hr_j)
                lines.append(f"   • {d}: {plat} offline {hms_from_hours(hrs)} (потери: ~{_fmt_idr(loss)})")
    return "\n".join(lines)


def _section3_clients(period: str, restaurant_id: int) -> str:
    eng = get_engine()
    start_str, end_str = period.split("_")
    qg = (
        "SELECT SUM(new_customers) new, SUM(repeated_customers) rep, SUM(reactivated_customers) rea, SUM(total_customers) tot, "
        "SUM(earned_new_customers) enew, SUM(earned_repeated_customers) erep, SUM(earned_reactivated_customers) erea "
        "FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?"
    )
    qj = (
        "SELECT SUM(new_client) new, SUM(active_client) act, SUM(returned_client) ret "
        "FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?"
    )
    g = pd.read_sql_query(qg, eng, params=(restaurant_id, start_str, end_str)).iloc[0].fillna(0)
    j = pd.read_sql_query(qj, eng, params=(restaurant_id, start_str, end_str)).iloc[0].fillna(0)
    grab_new, grab_rep, grab_rea, grab_tot = int(g['new']), int(g['rep']), int(g['rea']), int(g['tot'])
    gojek_new, gojek_act, gojek_ret = int(j['new']), int(j['act']), int(j['ret'])
    total_unique = grab_tot + gojek_new + gojek_act + gojek_ret  # верхняя оценка

    lines = []
    lines.append("👥 3. ДЕТАЛЬНЫЙ АНАЛИЗ КЛИЕНТСКОЙ БАЗЫ")
    lines.append("—" * 72)
    lines.append("📊 Структура клиентской базы (GRAB + GOJEK):")
    lines.append(f"  🆕 Новые клиенты: {grab_new + gojek_new}")
    lines.append(f"    📱 GRAB: {grab_new} | 🛵 GOJEK: {gojek_new}")
    lines.append(f"  🔄 Повторные клиенты: {grab_rep + gojek_act}")
    lines.append(f"    📱 GRAB: {grab_rep} | 🛵 GOJEK: {gojek_act}")
    lines.append(f"  📲 Реактивированные: {grab_rea + gojek_ret}")
    lines.append(f"    📱 GRAB: {grab_rea} | 🛵 GOJEK: {gojek_ret}")
    lines.append("")
    lines.append("💰 Доходность по типам клиентов (только GRAB, только с рекламы):")
    lines.append(f"  🆕 Новые: {_fmt_idr(g['enew'])}")
    lines.append(f"  🔄 Повторные: {_fmt_idr(g['erep'])}")
    lines.append(f"  📲 Реактивированные: {_fmt_idr(g['erea'])}")
    return "\n".join(lines)


def _section7_quality(quality: Dict) -> str:
    r = quality.get("ratings", {})
    lines = []
    lines.append("7. ⭐ КАЧЕСТВО ОБСЛУЖИВАНИЯ И УДОВЛЕТВОРЕННОСТЬ (GOJEK)")
    lines.append("—" * 72)
    total = r.get("total") or 0
    lines.append(f"📊 Распределение оценок (всего: {total}):")
    lines.append(f"  ⭐⭐⭐⭐⭐ 5 звезд: {r.get('five',0)} ({_fmt_pct((r.get('five',0)/total*100) if total else None)})")
    lines.append(f"  ⭐⭐⭐⭐ 4 звезды: {r.get('four',0)} ({_fmt_pct((r.get('four',0)/total*100) if total else None)})")
    lines.append(f"  ⭐⭐⭐ 3 звезды: {r.get('three',0)} ({_fmt_pct((r.get('three',0)/total*100) if total else None)})")
    lines.append(f"  ⭐⭐ 2 звезды: {r.get('two',0)} ({_fmt_pct((r.get('two',0)/total*100) if total else None)})")
    lines.append(f"  ⭐ 1 звезда: {r.get('one',0)} ({_fmt_pct((r.get('one',0)/total*100) if total else None)})")
    lines.append("")
    lines.append(f"📈 Индекс удовлетворенности: {_fmt_rate(r.get('satisfaction_index'))}/5.0")
    lines.append(f"🚨 Негативные отзывы (1-2★): {r.get('negative_1_2',{}).get('count',0)} ({_fmt_pct(r.get('negative_1_2',{}).get('percent'))})")
    lines.append("")
    lines.append("📊 Частота плохих оценок (не 5★):")
    lines.append(f"  📈 Плохих оценок всего: {r.get('not_five',{}).get('count',0)} из {total} ({_fmt_pct(r.get('not_five',{}).get('percent'))})")
    lines.append(f"  📦 Успешных заказов GOJEK на 1 плохую оценку: {_fmt_rate(quality.get('orders_per_not_five_rating'))}")
    return "\n".join(lines)


def _fmt_minutes_to_hhmmss(mins: Optional[float]) -> str:
    if mins is None or (isinstance(mins, float) and np.isnan(mins)):
        return "—"
    try:
        total_seconds = int(round(float(mins) * 60))
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    except Exception:
        return "—"


def _categorize_feature(name: str) -> str:
    n = name.lower()
    if n.startswith("mkt_") or "ads_spend" in n or "impressions" in n or "roas" in n:
        return "Marketing"
    if n.startswith("ops_") or any(k in n for k in ["accepting_time", "preparation_time", "delivery_time", "outage_", "offline_"]):
        return "Operations"
    if n in ("temp", "rain", "wind", "humidity", "tourist_flow", "is_holiday", "day_of_week", "is_weekend") or any(k in n for k in ["temp_", "rain_", "wind_", "humidity_", "tourist_flow_"]):
        return "External"
    if "rating" in n:
        return "Quality"
    return "Other"


def _pretty_feature_name(name: str) -> str:
    n = name.lower()
    mapping = {
        # Marketing
        "mkt_roas_grab": "ROAS (GRAB)",
        "mkt_roas_gojek": "ROAS (GOJEK)",
        "mkt_ads_spend_grab": "Рекламный бюджет (GRAB)",
        "mkt_ads_spend_gojek": "Рекламный бюджет (GOJEK)",
        "ads_spend_total": "Рекламный бюджет (суммарно)",
        "impressions_total": "Показы рекламы",
        # Operations (общие)
        "preparation_time_mean": "Среднее время приготовления (мин)",
        "accepting_time_mean": "Среднее время подтверждения (мин)",
        "delivery_time_mean": "Среднее время доставки (мин)",
        # Operations (GOJEK)
        "ops_preparation_time_gojek": "GOJEK: время приготовления",
        "ops_accepting_time_gojek": "GOJEK: время подтверждения",
        "ops_delivery_time_gojek": "GOJEK: время доставки",
        # Outage/offline
        "outage_offline_rate_grab": "GRAB: оффлайн (мин)",
        "offline_rate_grab": "GRAB: оффлайн (мин)",
        # External
        "rain": "Дождь (мм)",
        "temp": "Температура (°C)",
        "wind": "Ветер",
        "humidity": "Влажность (%)",
        "tourist_flow": "Туристический поток",
        "is_holiday": "Праздник",
        "day_of_week": "День недели",
        "is_weekend": "Выходной",
        # Quality
        "rating": "Средний рейтинг",
    }
    if n in mapping:
        return mapping[n]
    # Heuristics: platform/time metrics
    if n.startswith("ops_preparation_time_"):
        plat = n.split("_")[-1].upper()
        return f"{plat}: время приготовления"
    if n.startswith("ops_accepting_time_"):
        plat = n.split("_")[-1].upper()
        return f"{plat}: время подтверждения"
    if n.startswith("ops_delivery_time_"):
        plat = n.split("_")[-1].upper()
        return f"{plat}: время доставки"
    if n.startswith("mkt_roas_"):
        plat = n.split("_")[-1].upper()
        return f"ROAS ({plat})"
    if n.startswith("mkt_ads_spend_"):
        plat = n.split("_")[-1].upper()
        return f"Рекламный бюджет ({plat})"
    # Fallback: make readable
    pretty = name.replace("_", " ")
    pretty = pretty.replace("grab", "GRAB").replace("gojek", "GOJEK").replace("roas", "ROAS")
    return pretty


# Пороговые значения для стабилизации раздела 8
MIN_NEG_SHARE = 0.02  # Минимальная доля негативного вклада (2%)
MIN_NEG_IDR = 100000  # Минимальный вклад в IDR (100K IDR)
REPORT_STRICT_MODE = True  # Строгий режим: требует минимум данных

def _normalize_feature_name(feature: str) -> str:
    """Нормализация названий фич в человекочитаемые"""
    # Объединяем дублеры времени приготовления
    if any(x in feature.lower() for x in ['prep_time', 'preparation_time']):
        return "Время приготовления"
    
    # Объединяем погодные лаги
    if any(x in feature.lower() for x in ['humidity_lag', 'wind_lag', 'temp_lag', 'rain_lag']):
        weather_type = ""
        if 'humidity' in feature.lower():
            weather_type = "влажность"
        elif 'wind' in feature.lower():
            weather_type = "ветер"
        elif 'temp' in feature.lower():
            weather_type = "температура"
        elif 'rain' in feature.lower():
            weather_type = "дождь"
        return f"Погода: {weather_type} (предыдущие дни)"
    
    # Текущие погодные условия
    if feature.lower() in ['rain', 'humidity', 'wind', 'temp', 'temperature']:
        weather_map = {
            'rain': 'Дождь',
            'humidity': 'Влажность',
            'wind': 'Ветер',
            'temp': 'Температура',
            'temperature': 'Температура'
        }
        return f"Погода: {weather_map.get(feature.lower(), feature)}"
    
    # Маркетинговые метрики
    marketing_map = {
        'ads_spend': 'Рекламный бюджет',
        'ads_sales': 'Рекламные продажи',
        'roas': 'ROAS',
        'grab_ads_spend': 'GRAB: рекламный бюджет',
        'gojek_ads_spend': 'GOJEK: рекламный бюджет',
        'grab_roas': 'GRAB: ROAS',
        'gojek_roas': 'GOJEK: ROAS'
    }
    
    for key, value in marketing_map.items():
        if key in feature.lower():
            return value
    
    # Операционные метрики
    ops_map = {
        'confirm_time': 'Время подтверждения',
        'delivery_time': 'Время доставки',
        'rating': 'Рейтинг',
        'cancelled_orders': 'Отмены',
        'lost_orders': 'Потери'
    }
    
    for key, value in ops_map.items():
        if key in feature.lower():
            return value
    
    # Возвращаем исходное название если не нашли соответствие
    return _pretty_feature_name(feature)


def _section8_critical_days_ml(period: str, restaurant_id: int) -> str:
    """Улучшенный раздел 8 с пороговыми значениями и строгой фильтрацией"""
    try:
        start_str, end_str = period.split("_")
        project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
        df = pd.read_csv(csv_path, parse_dates=["date"])
        sub = df[(df["restaurant_id"] == restaurant_id) & (df["date"] >= start_str) & (df["date"] <= end_str)].copy()
        
        if sub.empty:
            return "8. 🚨 КРИТИЧЕСКИЕ ДНИ\n════════════════════════════════════════════════════════════════════════\nНет данных за выбранный период."
        
        # Строгий режим: проверяем достаточность данных
        if REPORT_STRICT_MODE and len(sub) < 7:
            return "8. 🚨 КРИТИЧЕСКИЕ ДНИ\n════════════════════════════════════════════════════════════════════════\nДанных недостаточно для анализа (минимум 7 дней)."

        # Находим критические дни (падение ≥30% от медианы)
        daily = sub.groupby("date", as_index=False)["total_sales"].sum().sort_values("date")
        median_sales = float(daily["total_sales"].median()) if len(daily) else 0.0
        threshold = 0.70 * median_sales  # 30% падение от медианы
        critical_dates = daily.loc[daily["total_sales"] <= threshold, "date"].dt.normalize().tolist()
        critical_dates = sorted(critical_dates, key=lambda d: daily.loc[daily["date"] == d, "total_sales"].iloc[0])

        lines: list[str] = []
        lines.append("8. 🚨 КРИТИЧЕСКИЕ ДНИ")
        lines.append("════════════════════════════════════════════════════════════════════════")
        
        # Заголовок с основной статистикой
        lines.append(f"📊 Найдено критических дней (падение ≥30%): {len(critical_dates)} из {len(daily)} ({len(critical_dates)/len(daily)*100:.1f}%)")
        lines.append(f"📈 Медианные продажи: {_fmt_idr(median_sales)}")
        lines.append(f"📉 Порог критичности: {_fmt_idr(threshold)}")
        
        if not critical_dates:
            lines.append("")
            lines.append("✅ В периоде нет критических дней (падение ≥30%)")
            return "\\n".join(lines)
        
        # Подсчитываем общие потери
        total_losses = 0.0
        for critical_date in critical_dates:
            day_sales = daily.loc[daily["date"] == critical_date, "total_sales"].iloc[0]
            loss = max(median_sales - day_sales, 0)
            total_losses += loss
        
        lines.append(f"💸 Общие потери от критических дней: {_fmt_idr(total_losses)}")
        lines.append("")
        
        def _analyze_critical_day_improved(critical_date: pd.Timestamp) -> list[str]:
            """Улучшенный анализ критического дня с пороговыми значениями"""
            day_lines = []
            
            # Получаем данные дня
            day_data = sub[sub["date"] == critical_date].iloc[0] if not sub[sub["date"] == critical_date].empty else None
            if day_data is None:
                return [f"🔴 {critical_date.strftime('%Y-%m-%d')}: нет данных"]
            
            day_sales = float(day_data["total_sales"])
            loss_amount = max(median_sales - day_sales, 0)
            loss_pct = ((day_sales - median_sales) / median_sales * 100) if median_sales > 0 else 0
            
            day_lines.append(f"🔴 {critical_date.strftime('%Y-%m-%d')}")
            day_lines.append("")
            
            # Ключевые цифры
            day_lines.append("### 📊 **КЛЮЧЕВЫЕ ЦИФРЫ**")
            day_lines.append(f"- **Продажи:** {_fmt_idr(day_sales)} (медиана: {_fmt_idr(median_sales)}) → **{loss_pct:+.1f}%**")
            day_lines.append(f"- **Потери:** {_fmt_idr(loss_amount)}")
            
            # Получаем заказы и средний чек если есть
            orders = day_data.get("orders_count", 0) or 0
            if orders > 0:
                avg_check = day_sales / orders
                day_lines.append(f"- **Заказы:** {orders} шт")
                day_lines.append(f"- **Средний чек:** {_fmt_idr(avg_check)}")
            
            day_lines.append("")
            
            # ML анализ причин с пороговыми значениями
            try:
                model, features, background = load_artifacts()
                if model is None or not features:
                    if REPORT_STRICT_MODE:
                        day_lines.append("### ⚠️ **АНАЛИЗ НЕДОСТУПЕН**")
                        day_lines.append("ML модель не обучена. Запустите обучение для получения детального анализа.")
                        return day_lines
                
                # Получаем SHAP значения для дня
                day_features = day_data[features] if all(f in day_data.index for f in features) else None
                if day_features is None:
                    if REPORT_STRICT_MODE:
                        day_lines.append("### ⚠️ **ДАННЫХ НЕДОСТАТОЧНО**")
                        day_lines.append("Отсутствуют необходимые features для ML анализа.")
                        return day_lines
                
                X_day = day_features.values.reshape(1, -1)
                pre = model.named_steps["pre"]
                mdl = model.named_steps["model"]
                X_pre = pre.transform(X_day)
                
                if background is not None and not background.empty:
                    bg_pre = pre.transform(background[features])
                    explainer = shap.TreeExplainer(mdl, data=bg_pre, feature_perturbation="interventional")
                else:
                    explainer = shap.TreeExplainer(mdl, feature_perturbation="interventional")
                
                shap_values = explainer.shap_values(X_pre)[0]
                
                # Фильтруем негативные факторы по пороговым значениям
                negative_factors = []
                positive_factors = []
                
                for i, (feature, shap_val) in enumerate(zip(features, shap_values)):
                    if shap_val < 0:  # Негативный вклад
                        contribution_idr = abs(shap_val)
                        contribution_share = abs(shap_val) / loss_amount if loss_amount > 0 else 0
                        
                        # Применяем пороговые значения
                        if contribution_share >= MIN_NEG_SHARE and contribution_idr >= MIN_NEG_IDR:
                            normalized_name = _normalize_feature_name(feature)
                            negative_factors.append((normalized_name, contribution_idr, contribution_share * 100))
                    
                    elif shap_val > 0:  # Позитивный вклад (что помогло)
                        contribution_idr = shap_val
                        normalized_name = _normalize_feature_name(feature)
                        positive_factors.append((normalized_name, contribution_idr))
                
                # Сортируем по вкладу и берем топ-5
                negative_factors.sort(key=lambda x: x[1], reverse=True)
                negative_factors = negative_factors[:5]
                
                positive_factors.sort(key=lambda x: x[1], reverse=True)
                positive_factors = positive_factors[:3]  # Топ-3 помогающих фактора
                
                # Строгий режим: проверяем достаточность факторов
                if REPORT_STRICT_MODE and len(negative_factors) < 2:
                    day_lines.append("### ⚠️ **ДАННЫХ НЕДОСТАТОЧНО**")
                    day_lines.append("Найдено менее 2 значимых факторов. ML анализ неточен.")
                    return day_lines
                
                # Реальные причины
                day_lines.append("### 🔍 **РЕАЛЬНЫЕ ПРИЧИНЫ**")
                
                for i, (factor_name, contribution_idr, contribution_pct) in enumerate(negative_factors, 1):
                    priority = "🔴" if contribution_pct >= 15.0 else ("🟡" if contribution_pct >= 7.0 else "🟢")
                    day_lines.append(f"**{i}. {priority} {factor_name.upper()}**")
                    day_lines.append(f"- **Влияние:** {_fmt_idr(contribution_idr)} ({contribution_pct:.1f}% от потерь)")
                    day_lines.append("")
                
                # Внешние факторы (праздники и погода) только если превышают пороги
                day_lines.append("### 🌍 **ВНЕШНИЕ ФАКТОРЫ**")
                
                # Праздники - только если holiday_flag==1 и вклад >= порога
                is_holiday = int(day_data.get("is_holiday", 0)) == 1
                holiday_contribution = 0
                
                # Ищем вклад праздника в негативных факторах
                for factor_name, contribution_idr, contribution_pct in negative_factors:
                    if "праздник" in factor_name.lower() or "holiday" in factor_name.lower():
                        holiday_contribution = contribution_idr
                        break
                
                if is_holiday and holiday_contribution >= MIN_NEG_IDR:
                    holiday_info = _check_holiday_by_date_simple(critical_date.strftime('%Y-%m-%d'))
                    day_lines.append(f"**🕌 Праздники:** {holiday_info}")
                elif is_holiday:
                    holiday_info = _check_holiday_by_date_simple(critical_date.strftime('%Y-%m-%d'))
                    day_lines.append(f"**🕌 Праздники:** {holiday_info} (влияние незначительное)")
                else:
                    day_lines.append("**🕌 Праздники:** обычный день")
                
                # Дождь - только если rain_mm >= 2.0 и вклад >= порога
                rain_mm = float(day_data.get("rain", 0)) if pd.notna(day_data.get("rain")) else 0.0
                rain_contribution = 0
                
                # Ищем вклад дождя в негативных факторах
                for factor_name, contribution_idr, contribution_pct in negative_factors:
                    if "дождь" in factor_name.lower() or "rain" in factor_name.lower():
                        rain_contribution = contribution_idr
                        break
                
                if rain_mm >= 2.0 and rain_contribution >= MIN_NEG_IDR:
                    rain_desc = "сильный дождь" if rain_mm >= 10.0 else "дождь"
                    day_lines.append(f"**🌧️ Погода:** {rain_desc} {rain_mm:.1f}мм — снизил активность курьеров")
                elif rain_mm >= 2.0:
                    day_lines.append(f"**🌧️ Погода:** дождь {rain_mm:.1f}мм (влияние незначительное)")
                else:
                    temp = day_data.get("temp", 0) or 0
                    day_lines.append(f"**🌧️ Погода:** без дождя, комфортная температура {temp}°C")
                
                day_lines.append("")
                
                # Что помогло избежать больших потерь (позитивные факторы)
                if positive_factors:
                    day_lines.append("### ✅ **ЧТО ПОМОГЛО ИЗБЕЖАТЬ БОЛЬШИХ ПОТЕРЬ**")
                    for factor_name, contribution_idr in positive_factors:
                        day_lines.append(f"**💪 {factor_name}:**")
                        day_lines.append(f"- Положительный эффект: +{_fmt_idr(contribution_idr)}")
                    day_lines.append("")
                
                # Конкретные рекомендации с финансовым эффектом
                day_lines.append("### 🎯 **КОНКРЕТНЫЕ РЕКОМЕНДАЦИИ**")
                
                recommendations = []
                total_potential = 0
                
                for i, (factor_name, contribution_idr, contribution_pct) in enumerate(negative_factors[:3], 1):
                    priority = "🔴" if contribution_pct >= 15.0 else ("🟡" if contribution_pct >= 7.0 else "🟢")
                    
                    # Генерируем рекомендации на основе типа фактора
                    if "бюджет" in factor_name.lower():
                        rec_effect = contribution_idr * 0.8  # 80% восстановления
                        recommendations.append(f"**{i}. {priority} Оптимизировать рекламный бюджет**")
                        recommendations.append(f"- **Потенциальный эффект:** {_fmt_idr(rec_effect)}")
                        total_potential += rec_effect
                    elif "время" in factor_name.lower():
                        rec_effect = contribution_idr * 0.6  # 60% восстановления
                        recommendations.append(f"**{i}. {priority} Ускорить операционные процессы**")
                        recommendations.append(f"- **Потенциальный эффект:** {_fmt_idr(rec_effect)}")
                        total_potential += rec_effect
                    elif "рейтинг" in factor_name.lower():
                        rec_effect = contribution_idr * 0.7  # 70% восстановления
                        recommendations.append(f"**{i}. {priority} Улучшить качество сервиса**")
                        recommendations.append(f"- **Потенциальный эффект:** {_fmt_idr(rec_effect)}")
                        total_potential += rec_effect
                    else:
                        rec_effect = contribution_idr * 0.5  # 50% восстановления по умолчанию
                        recommendations.append(f"**{i}. {priority} Исправить {factor_name.lower()}**")
                        recommendations.append(f"- **Потенциальный эффект:** {_fmt_idr(rec_effect)}")
                        total_potential += rec_effect
                
                for rec in recommendations:
                    day_lines.append(rec)
                
                day_lines.append("")
                day_lines.append("### 💰 **ФИНАНСОВЫЙ ИТОГ**")
                recovery_pct = (total_potential / loss_amount * 100) if loss_amount > 0 else 0
                day_lines.append(f"- **Общий потенциал восстановления:** {_fmt_idr(total_potential)} ({recovery_pct:.0f}% от потерь)")
                day_lines.append("")
                
            except Exception as e:
                if REPORT_STRICT_MODE:
                    day_lines.append("### ⚠️ **ML АНАЛИЗ НЕДОСТУПЕН**")
                    day_lines.append(f"Ошибка ML анализа: {str(e)}")
                    day_lines.append("Используйте базовый анализ или переобучите модель.")
                    day_lines.append("")
            
            return day_lines
        
        # Анализируем ВСЕ критические дни с новой логикой
        for critical_date in critical_dates:
            day_analysis = _analyze_critical_day_improved(critical_date)
            lines.extend(day_analysis)
        
        # Общие выводы
        lines.append(f"💸 **ОБЩИЕ ПОТЕРИ ОТ ВСЕХ КРИТИЧЕСКИХ ДНЕЙ: {_fmt_idr(total_losses)}**")
        lines.append("")
        
        lines.append("📊 ОБЩИЕ ВЫВОДЫ")
        lines.append("────────────────────────────────────────")
        lines.append(f"📊 Всего критических дней: {len(critical_dates)} из {len(daily)} ({len(critical_dates)/len(daily)*100:.1f}%)")
        
        # Анализ типов критических дней
        holiday_days = 0
        rainy_days = 0
        avg_loss = total_losses / len(critical_dates) if critical_dates else 0
        
        for critical_date in critical_dates:
            day_data = sub[sub["date"] == critical_date].iloc[0] if not sub[sub["date"] == critical_date].empty else None
            if day_data is not None:
                if int(day_data.get("is_holiday", 0)) == 1:
                    holiday_days += 1
                if float(day_data.get("rain", 0) or 0) >= 2.0:
                    rainy_days += 1
        
        lines.append(f"🕌 Праздничных дней: {holiday_days} ({holiday_days/len(critical_dates)*100:.0f}%)")
        lines.append(f"🌧️ Дождливых дней: ~{rainy_days} ({rainy_days/len(critical_dates)*100:.0f}%)")
        lines.append(f"📈 Средние потери на критический день: {_fmt_idr(avg_loss)}")
        lines.append("")
        lines.append("🎯 ПРИОРИТЕТНАЯ РЕКОМЕНДАЦИЯ:")
        lines.append("Разработать стратегию работы в праздничные и дождливые дни")
        lines.append("")
        lines.append("📋 ИСТОЧНИКИ ДАННЫХ:")
        lines.append("- SQLite (grab_stats, gojek_stats) — операционные данные")
        lines.append("- Open-Meteo API — погодные данные")
        lines.append("- Holidays cache — праздники (мусульманские, балийские, индонезийские, международные)")
        lines.append("- ML модель (Random Forest) — SHAP анализ факторов")
        
        return "\\n".join(lines)
    
    except Exception as e:
        return f"8. 🚨 КРИТИЧЕСКИЕ ДНИ\\n════════════════════════════════════════════════════════════════════════\\nОшибка анализа: {str(e)}\\nПроверьте наличие ML модели и данных."
def _section9_recommendations(period: str, restaurant_id: int) -> str:
    try:
        # Use SHAP over the whole period to prioritize levers; exclude trivial features
        start_str, end_str = period.split("_")
        project_root = os.getenv("PROJECT_ROOT", os.getcwd())
        csv_path = os.getenv("ML_DATASET_CSV", os.path.join(project_root, "data", "merged_dataset.csv"))
        df = pd.read_csv(csv_path, parse_dates=["date"]) if os.path.exists(csv_path) else pd.DataFrame()
        sub = df[(df.get("restaurant_id") == restaurant_id) & (df.get("date") >= start_str) & (df.get("date") <= end_str)].copy() if not df.empty else pd.DataFrame()
        lines = []
        lines.append("9. 🎯 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ")
        lines.append("—" * 72)
        if sub.empty:
            lines.append("Нет данных за период.")
            return "\n".join(lines)

        # Load model and compute feature importances
        model, features, background = load_artifacts(os.getenv("ML_ARTIFACT_DIR", os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "ml", "artifacts")))
        X = sub[features]
        pre = model.named_steps["pre"]
        mdl = model.named_steps["model"]
        X_pre = pre.transform(X)
        try:
            if background is not None and not background.empty:
                bg_pre = pre.transform(background[features])
                explainer = shap.TreeExplainer(mdl, data=bg_pre, feature_perturbation="interventional")
            else:
                explainer = shap.TreeExplainer(mdl, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_pre)
        except Exception:
            explainer = shap.TreeExplainer(mdl)
            shap_values = explainer.shap_values(X_pre)
        _, groups = _resolve_preprocessed_feature_groups(pre)
        import re as _re
        pat = [_re.compile(r"^orders_count(?!.*conversion).*"), _re.compile(r"^total_sales.*"), _re.compile(r"^restaurant_id$")]
        def is_excl(n: str) -> bool:
            return any(p.search(n) for p in pat)
        abs_sv = np.abs(shap_values)
        agg: Dict[str, float] = {}
        for feat, idxs in groups.items():
            if is_excl(feat) or not idxs:
                continue
            agg[feat] = float(abs_sv[:, idxs].mean())
        top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:8]

        # Group by categories
        cats: Dict[str, float] = {}
        for f, v in agg.items():
            c = _categorize_feature(f)
            cats[c] = cats.get(c, 0.0) + v
        tot = sum(cats.values()) or 1.0
        for k in list(cats.keys()):
            cats[k] = round(100.0 * cats[k] / tot, 1)

        lines.append("Приоритеты по факторам (ML):")
        for f, v in top:
            lines.append(f"  • [{_categorize_feature(f)}] {_pretty_feature_name(f)}")
        lines.append("")
        lines.append("Вклад групп факторов:")
        for k in ["Operations", "Marketing", "External", "Quality", "Other"]:
            if k in cats:
                lines.append(f"  • {k}: {cats[k]}%")
        lines.append("")
        lines.append("Рекомендуемые действия:")
        if cats.get("Operations", 0) >= 30.0:
            lines.append("  • Сократить SLA (prep/accept/delivery) в пиковые окна; предзаготовки, слотирование, контроль выдачи")
        if cats.get("Marketing", 0) >= 20.0:
            lines.append("  • Перераспределить бюджет в связки с лучшим ROAS; тест креативов и аудиторий; корректировка ставок")
        lines.append("  • Погодные промо и бонусы курьерам в дождь; перенос активностей на «сухие» окна")
        lines.append("  • Учитывать локальные праздники в планировании (снижение бюджета/акции на следующий день)")
        lines.append("  • Контроль качества и рейтингов: работа с негативом, улучшение времени ожидания")
        return "\n".join(lines)
    except Exception:
        return "9. 🎯 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ\n" + ("—" * 72) + "\nНе удалось построить рекомендации."


def generate_full_report(period: str, restaurant_id: int) -> str:
    basic = build_basic_report(period, restaurant_id)
    marketing = build_marketing_report(period, restaurant_id)
    quality = build_quality_report(period, restaurant_id)
    finance = basic.get("finance") if isinstance(basic, dict) else None

    parts = []
    # Dataset version banner
    banner = _dataset_version_banner()
    if banner:
        parts.append(banner)
        parts.append("")
    # Section 1
    parts.append(_section1_exec(basic))
    parts.append("")
    # Section 2
    parts.append(_section2_trends(basic))
    parts.append("")
    # Section 3
    parts.append(_section3_clients(period, restaurant_id))
    parts.append("")
    # Section 4
    parts.append(_section4_marketing(marketing))
    parts.append("")
    # Section 5
    if finance:
        parts.append(_section5_finance(finance))
        parts.append("")
    # Section 6
    parts.append(_section6_operations(period, restaurant_id))
    parts.append("")
    # Section 7
    parts.append(_section7_quality(quality))
    parts.append("")
    # Section 8 (ML Critical Days)
    parts.append(_section8_critical_days_ml(period, restaurant_id))
    parts.append("")
    # Section 9 (Recommendations)
    parts.append(_section9_recommendations(period, restaurant_id))
    parts.append("")
    return "\n".join(parts)