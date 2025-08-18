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
        metrics_path = "/workspace/ml/artifacts/metrics.json"
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


def _section8_critical_days_ml(period: str, restaurant_id: int) -> str:
    try:
        start_str, end_str = period.split("_")
        df = pd.read_csv("/workspace/data/merged_dataset.csv", parse_dates=["date"])  # daily rows per restaurant
        sub = df[(df["restaurant_id"] == restaurant_id) & (df["date"] >= start_str) & (df["date"] <= end_str)].copy()
        if sub.empty:
            return "8. 🚨 КРИТИЧЕСКИЕ ДНИ (ML)\n" + ("—" * 72) + "\nНет данных за выбранный период."

        # Median per day and critical threshold (≤ -30% к медиане)
        daily = sub.groupby("date", as_index=False)["total_sales"].sum().sort_values("date")
        med = float(daily["total_sales"].median()) if len(daily) else 0.0
        thr = 0.7 * med
        critical_dates = daily.loc[daily["total_sales"] <= thr, "date"].dt.normalize().tolist()

        lines: list[str] = []
        lines.append("8. 🚨 КРИТИЧЕСКИЕ ДНИ (ML)")
        lines.append("—" * 72)
        def _expected_baseline_for_day(daily_df: pd.DataFrame, d: pd.Timestamp) -> float:
            try:
                dow = int(d.dayofweek)
                window = daily_df[daily_df['date'] < d].tail(56)
                same_dow = window[window['date'].dt.dayofweek == dow]
                series = same_dow['total_sales'] if not same_dow.empty else window['total_sales']
                if series.empty:
                    series = daily_df['total_sales']
                return float(series.median()) if not series.empty else 0.0
            except Exception:
                return 0.0
        if not critical_dates:
            lines.append("В периоде нет дней с падением ≥ 30% к медиане.")
            # Добавим краткий причинный срез по дождю/праздникам для периода
            sub['heavy_rain'] = (sub['rain'].fillna(0.0) >= 10.0).astype(int)
            def _mean(series):
                s = pd.to_numeric(series, errors='coerce')
                return float(s.mean()) if len(s) else 0.0
            by_rain = sub.groupby('heavy_rain')['total_sales'].mean().to_dict()
            if 0 in by_rain:
                dr = (by_rain.get(1, by_rain[0]) - by_rain[0]) / (by_rain[0] or 1.0) * 100.0
                lines.append(f"🌧️ Эффект дождя (простая разница средних): {_fmt_pct(dr)}")
            by_h = sub.groupby(sub['is_holiday'].fillna(0).astype(int))['total_sales'].mean().to_dict()
            if 0 in by_h:
                dh = (by_h.get(1, by_h[0]) - by_h[0]) / (by_h[0] or 1.0) * 100.0
                lines.append(f"🎌 Эффект праздников (простая разница средних): {_fmt_pct(dh)}")
            return "\n".join(lines)

        # Prepare SHAP per-row
        model, features, background = load_artifacts()
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
        # Exclude trivial features
        pat = [re.compile(r"^orders_count(?!.*conversion).*"), re.compile(r"^total_sales.*"), re.compile(r"^restaurant_id$")]
        def is_excluded(n: str) -> bool:
            return any(p.search(n) for p in pat)

        eng = get_engine()
        # Period baselines (для человеческих объяснений)
        qg_all = pd.read_sql_query(
            "SELECT stat_date, ads_spend, ads_sales, cancelled_orders, offline_rate FROM grab_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
            eng, params=(restaurant_id, start_str, end_str)
        )
        qj_all = pd.read_sql_query(
            "SELECT stat_date, ads_spend, ads_sales, accepting_time, preparation_time, delivery_time, close_time, cancelled_orders FROM gojek_stats WHERE restaurant_id=? AND stat_date BETWEEN ? AND ?",
            eng, params=(restaurant_id, start_str, end_str)
        )
        qg_all['stat_date'] = pd.to_datetime(qg_all['stat_date'], errors='coerce').dt.date
        qj_all['stat_date'] = pd.to_datetime(qj_all['stat_date'], errors='coerce').dt.date
        def _to_min_p(v):
            if v is None:
                return None
            s = str(v)
            parts = s.split(":")
            try:
                if len(parts) == 3:
                    h, m, sec = parts
                    return int(h) * 60 + int(m) + int(sec) / 60.0
            except Exception:
                pass
            try:
                return float(s)
            except Exception:
                return None
        def _safe_mean(series):
            s = pd.to_numeric(series, errors='coerce').dropna()
            return float(s.mean()) if not s.empty else None
        # Baselines
        spend_g_avg = _safe_mean(qg_all.get('ads_spend'))
        spend_j_avg = _safe_mean(qj_all.get('ads_spend'))
        roas_g_avg = None
        if not qg_all.empty:
            tmp = []
            for _, r in qg_all.iterrows():
                a, s_ = r.get('ads_spend'), r.get('ads_sales')
                try:
                    a = float(a); s_ = float(s_)
                    if a > 0:
                        tmp.append(s_ / a)
                except Exception:
                    pass
            roas_g_avg = float(np.mean(tmp)) if tmp else None
        roas_j_avg = None
        if not qj_all.empty:
            tmp = []
            for _, r in qj_all.iterrows():
                a, s_ = r.get('ads_spend'), r.get('ads_sales')
                try:
                    a = float(a); s_ = float(s_)
                    if a > 0:
                        tmp.append(s_ / a)
                except Exception:
                    pass
            roas_j_avg = float(np.mean(tmp)) if tmp else None
        canc_g_avg = _safe_mean(qg_all.get('cancelled_orders'))
        canc_j_avg = _safe_mean(qj_all.get('cancelled_orders'))
        off_g_avg = _safe_mean(qg_all.get('offline_rate'))
        prep_avg = _safe_mean(qj_all.get('preparation_time').apply(_to_min_p)) if 'preparation_time' in qj_all.columns else None
        accept_avg = _safe_mean(qj_all.get('accepting_time').apply(_to_min_p)) if 'accepting_time' in qj_all.columns else None
        deliv_avg = _safe_mean(qj_all.get('delivery_time').apply(_to_min_p)) if 'delivery_time' in qj_all.columns else None

        for d in critical_dates:
            day_mask = sub["date"].dt.normalize() == d
            idxs = np.where(day_mask.values)[0]
            if len(idxs) == 0:
                continue
            # Aggregate contributions over all rows of that date (should typically be 1 per date)
            contrib_sum: Dict[str, float] = {}
            for i in idxs:
                for feat, cols in groups.items():
                    if is_excluded(feat):
                        continue
                    if not cols:
                        continue
                    val = float(np.sum(shap_values[i, cols]))
                    contrib_sum[feat] = contrib_sum.get(feat, 0.0) + val
            # Select significant factors by |impact|: cover ~95% cumulatively and include any with share >=1%
            contrib_sorted = sorted(contrib_sum.items(), key=lambda x: abs(x[1]), reverse=True)
            total_abs = sum(abs(v) for v in contrib_sum.values()) or 1.0
            selected = []
            cum = 0.0
            for feat, val in contrib_sorted:
                share = abs(val) / total_abs
                if share >= 0.01 or cum < 0.95:
                    selected.append((feat, val))
                    cum += share
                else:
                    break

            # Group shares
            group_shares: Dict[str, float] = {}
            for feat, val in contrib_sum.items():
                cat = _categorize_feature(feat)
                group_shares[cat] = group_shares.get(cat, 0.0) + abs(val)
            for k in list(group_shares.keys()):
                group_shares[k] = round(100.0 * group_shares[k] / total_abs, 1)

            # Day-level raw details from stats
            ds = str(d.date())
            qg = pd.read_sql_query(
                "SELECT sales, orders, ads_spend, ads_sales, offline_rate, cancelled_orders FROM grab_stats WHERE restaurant_id=? AND stat_date=?",
                eng, params=(restaurant_id, ds)
            )
            qj = pd.read_sql_query(
                "SELECT sales, orders, ads_spend, ads_sales, accepting_time, preparation_time, delivery_time, close_time, cancelled_orders FROM gojek_stats WHERE restaurant_id=? AND stat_date=?",
                eng, params=(restaurant_id, ds)
            )
            grab_off_mins = float(qg.iloc[0]["offline_rate"]) if (not qg.empty and pd.notna(qg.iloc[0]["offline_rate"])) else None
            gojek_close = str(qj.iloc[0]["close_time"]) if (not qj.empty and pd.notna(qj.iloc[0]["close_time"])) else ""
            # close_time may be HH:MM:SS
            def _hms_close(s: str) -> str:
                parts = s.split(":") if s else []
                try:
                    if len(parts) == 3:
                        h, m, sec = parts
                        return f"{int(h)}:{int(m):02d}:{int(sec):02d}"
                except Exception:
                    pass
                return "—"

            # Weather/holiday from dataset row (first match of the date)
            row = sub.loc[day_mask].iloc[0]
            rain = float(row.get("rain")) if pd.notna(row.get("rain")) else None
            temp = float(row.get("temp")) if pd.notna(row.get("temp")) else None
            wind = float(row.get("wind")) if pd.notna(row.get("wind")) else None
            hum = float(row.get("humidity")) if pd.notna(row.get("humidity")) else None
            is_hol = int(row.get("is_holiday")) if pd.notna(row.get("is_holiday")) else 0
            total_sales_day = float(daily.loc[daily["date"] == d, "total_sales"].iloc[0])
            delta_pct = ((total_sales_day - med) / med * 100.0) if med else None
            delta_idr = max(med - total_sales_day, 0.0) if med else 0.0
            expected_idr = _expected_baseline_for_day(daily, d)
            drop_idr = max(0.0, expected_idr - total_sales_day)

            lines.append(f"📉 КРИТИЧЕСКИЙ ДЕНЬ: {ds} (выручка: {_fmt_idr(total_sales_day)}; отклонение к медиане: {_fmt_pct(delta_pct)})")
            lines.append("—" * 72)
            # Compute business-oriented factor sets (threshold 3%)
            def _share(v: float) -> float:
                return round(100.0 * abs(v) / total_abs, 1)
            def _is_technical(name: str) -> bool:
                n = name.lower()
                return ('lag' in n) or ('rolling' in n)
            sig_all = [(f, v, _share(v)) for f, v in selected if _share(v) >= 3.0]
            sig = [(f, v, s) for f, v, s in sig_all if not _is_technical(f)] or sig_all
            neg = [(f, v, s) for f, v, s in sig if v < 0]
            pos = [(f, v, s) for f, v, s in sig if v > 0]
            neg = sorted(neg, key=lambda x: x[2], reverse=True)[:5]
            pos = sorted(pos, key=lambda x: x[2], reverse=True)[:2]

            # Compute monetary effect for negative factors and deduplicate
            neg_total_abs = sum(abs(v) for f, v in contrib_sum.items() if v < 0) or 1.0
            factor_rows_neg: list[tuple[str, float, int]] = []
            seen_canon: set[str] = set()
            def _canon(name: str) -> str:
                n = name.lower()
                n = n.replace("preparation_time_mean","preparation_time").replace("accepting_time_mean","accepting_time").replace("delivery_time_mean","delivery_time")
                return n
            for f, v, s in neg:
                canon = _canon(f)
                if canon in seen_canon:
                    continue
                seen_canon.add(canon)
                money = round((abs(v) / neg_total_abs) * drop_idr)
                if s < 5.0 and money < 50000:
                    continue
                factor_rows_neg.append((f, s, money))

            # Day-level metrics snapshot for comments
            # Build baselines already computed above: roas_g_avg, roas_j_avg, prep/accept/deliv avg, etc.
            day_roas_g = None; day_roas_j = None; day_spend_g = None; day_spend_j = None
            day_ads_sales_g = None; day_ads_sales_j = None
            if not qg.empty:
                gs = qg.iloc[0]
                day_spend_g = float(gs.get('ads_spend')) if pd.notna(gs.get('ads_spend')) else None
                day_roas_g = (float(gs.get('ads_sales')) / float(gs.get('ads_spend'))) if (pd.notna(gs.get('ads_spend')) and float(gs.get('ads_spend'))>0) else None
                day_ads_sales_g = float(gs.get('ads_sales')) if pd.notna(gs.get('ads_sales')) else None
            if not qj.empty:
                js = qj.iloc[0]
                day_spend_j = float(js.get('ads_spend')) if pd.notna(js.get('ads_spend')) else None
                day_roas_j = (float(js.get('ads_sales')) / float(js.get('ads_spend'))) if (pd.notna(js.get('ads_spend')) and float(js.get('ads_spend'))>0) else None
                day_ads_sales_j = float(js.get('ads_sales')) if pd.notna(js.get('ads_sales')) else None
            d_prep = _to_min_p(qj.iloc[0].get('preparation_time')) if not qj.empty else None
            d_acc = _to_min_p(qj.iloc[0].get('accepting_time')) if not qj.empty else None
            d_del = _to_min_p(qj.iloc[0].get('delivery_time')) if not qj.empty else None

            def _comment_for(feat_name: str, is_positive: bool) -> str:
                n = feat_name.lower()
                # Marketing
                if 'roas' in n:
                    # choose platform
                    if 'grab' in n and day_roas_g is not None and roas_g_avg is not None:
                        if roas_g_avg > 0:
                            diff = (day_roas_g - roas_g_avg) / roas_g_avg * 100.0
                            trend = 'вырос' if diff > 0 else ('упал' if diff < 0 else 'без изменений')
                            if trend == 'без изменений':
                                return 'реклама GRAB — без изменений'
                            return f"ROAS GRAB {trend} на {abs(diff):.0f}% ({day_roas_g:.2f}x против {roas_g_avg:.2f}x)"
                        return "рекламная эффективность GRAB — без данных"
                    if 'gojek' in n and day_roas_j is not None and roas_j_avg is not None:
                        if roas_j_avg > 0:
                            diff = (day_roas_j - roas_j_avg) / roas_j_avg * 100.0
                            trend = 'вырос' if diff > 0 else ('упал' if diff < 0 else 'без изменений')
                            if trend == 'без изменений':
                                return 'реклама GOJEK — без изменений'
                            return f"ROAS GOJEK {trend} на {abs(diff):.0f}% ({day_roas_j:.2f}x против {roas_j_avg:.2f}x)"
                        return "рекламная эффективность GOJEK — без данных"
                    return "рекламная эффективность ниже нормы" if not is_positive else "рекламная эффективность выше нормы"
                if 'ads_spend' in n or 'budget' in n:
                    if 'grab' in n and day_spend_g is not None and spend_g_avg is not None:
                        if spend_g_avg > 0:
                            diff = (day_spend_g - spend_g_avg) / spend_g_avg * 100.0
                            trend = 'выше' if diff > 0 else ('ниже' if diff < 0 else 'на уровне')
                            if trend == 'на уровне':
                                return 'бюджет GRAB — без изменений'
                            return f"бюджет GRAB {trend} среднего на {abs(diff):.0f}% ({_fmt_idr(day_spend_g)} против {_fmt_idr(spend_g_avg)})"
                        return "бюджет GRAB — без данных"
                    if 'gojek' in n and day_spend_j is not None and spend_j_avg is not None:
                        if spend_j_avg > 0:
                            diff = (day_spend_j - spend_j_avg) / spend_j_avg * 100.0
                            trend = 'выше' if diff > 0 else ('ниже' if diff < 0 else 'на уровне')
                            if trend == 'на уровне':
                                return 'бюджет GOJEK — без изменений'
                            return f"бюджет GOJEK {trend} среднего на {abs(diff):.0f}% ({_fmt_idr(day_spend_j)} против {_fmt_idr(spend_j_avg)})"
                        return "бюджет GOJEK — без данных"
                    return "изменение рекламной активности"
                # Operations
                if 'preparation_time' in n:
                    if d_prep is not None and prep_avg is not None and prep_avg > 0:
                        diff = (d_prep - prep_avg) / prep_avg * 100.0
                        if abs(diff) < 0.5:
                            return "время приготовления — без изменений"
                        trend = 'ниже нормы' if diff < 0 else 'выше нормы'
                        return f"время приготовления {trend} на {abs(diff):.0f}% ({d_prep:.1f} против {prep_avg:.1f} мин)"
                    return "скорость приготовления"
                if 'accepting_time' in n:
                    if d_acc is not None and accept_avg is not None and accept_avg > 0:
                        diff = (d_acc - accept_avg) / accept_avg * 100.0
                        if abs(diff) < 0.5:
                            return "подтверждение — без изменений"
                        trend = 'быстрее обычного' if diff < 0 else 'дольше обычного'
                        return f"подтверждение {trend} на {abs(diff):.0f}% ({d_acc:.1f} против {accept_avg:.1f} мин)"
                    return "скорость подтверждения"
                if 'delivery_time' in n:
                    if d_del is not None and deliv_avg is not None and deliv_avg > 0:
                        diff = (d_del - deliv_avg) / deliv_avg * 100.0
                        if abs(diff) < 0.5:
                            return "доставка — без изменений"
                        trend = 'быстрее обычного' if diff < 0 else 'дольше обычного'
                        return f"доставка {trend} на {abs(diff):.0f}% ({d_del:.1f} против {deliv_avg:.1f} мин)"
                    return "скорость доставки"
                if 'offline' in n or 'outage' in n:
                    return "платформа была недоступна (оффлайн)"
                # External
                if 'rain' in n:
                    return "дождь снизил спрос" if not is_positive else "погода благоприятна"
                if 'day_of_week' in n or 'weekend' in n:
                    return "слабый день недели" if not is_positive else "сильный день недели"
                if 'humidity' in n or 'wind' in n or 'temp' in n:
                    return "погодные условия снизили спрос" if not is_positive else "погодные условия помогли"
                if 'rating' in n:
                    return "рейтинг повлиял на спрос"
                return "влияющий фактор периода"

            # Consulting-style summary
            lines.append("📊 Факт:")
            lines.append(f"- Выручка: {_fmt_idr(total_sales_day)} против медианы {_fmt_idr(med)} ({_fmt_pct(delta_pct)})")
            lines.append(f"💸 Потеря денег: {_fmt_idr(delta_idr)}")
            if neg:
                topn = ", ".join([_pretty_feature_name(f) for f, _, _ in neg[:2]])
                lines.append(f"🔑 Главные драйверы: {topn}")
            # Control hypotheses by categories
            cats = set(_categorize_feature(f) for f,_,_ in neg[:3])
            hyp_map = {"Marketing": "неэффективные креативы/аудитории, ставки/распределение бюджета",
                       "Operations": "перегруз кухни/бутылочные горлышки, нехватка персонала в пик",
                       "External": "меньше курьеров/выше ETA (дождь/праздник)",
                       "Quality": "оценки и опыт влияли на конверсию"}
            if cats:
                lines.append("📌 Контрольные гипотезы: " + "; ".join([hyp_map.get(c, c) for c in cats]))
            if grab_off_mins and grab_off_mins > 0:
                lines.append(f"- Доступность: оффлайн GRAB {_fmt_minutes_to_hhmmss(grab_off_mins)}")
            if rain and rain >= 5.0:
                lines.append(f"- Внешний фактор: сильный дождь {rain} мм")
            lines.append("")

            # Marketing block (per-channel deltas vs period average)
            def _pct_delta(val: Optional[float], avg: Optional[float]) -> Optional[float]:
                try:
                    if val is None or avg is None or float(avg) == 0.0:
                        return None
                    return (float(val) - float(avg)) / float(avg) * 100.0
                except Exception:
                    return None
            avg_ads_sales_g = _safe_mean(qg_all.get('ads_sales')) if 'ads_sales' in qg_all.columns else None
            avg_ads_sales_j = _safe_mean(qj_all.get('ads_sales')) if 'ads_sales' in qj_all.columns else None
            spend_g_delta = _pct_delta(day_spend_g, spend_g_avg)
            spend_j_delta = _pct_delta(day_spend_j, spend_j_avg)
            sales_g_delta = _pct_delta(day_ads_sales_g, avg_ads_sales_g)
            sales_j_delta = _pct_delta(day_ads_sales_j, avg_ads_sales_j)
            # Headline driver examples
            mk_headlines = []
            if spend_j_delta is not None and sales_j_delta is not None and spend_j_delta > 0 and sales_j_delta < 0:
                mk_headlines.append(f"GOJEK Ads: бюджет +{abs(spend_j_delta):.1f}% при падении продаж {abs(sales_j_delta):.1f}%")
            if spend_g_delta is not None and sales_g_delta is not None and spend_g_delta < 0 and sales_g_delta < 0:
                mk_headlines.append(f"GRAB Ads: бюджет −{abs(spend_g_delta):.1f}% и продажи −{abs(sales_g_delta):.1f}% (снижение активности)")
            if mk_headlines:
                for hl in mk_headlines[:2]:
                    lines.append(f"- {hl}.")
                lines.append("")
            # Marketing table
            lines.append("Маркетинг:")
            lines.append("| Канал | Ads Spend | Ads Sales | Комментарий |")
            lines.append("|---|---:|---:|---|")
            def _eff_label(spend_d: Optional[float], sales_d: Optional[float]) -> str:
                if spend_d is None or sales_d is None:
                    return "—"
                if spend_d > 0 and sales_d < 0:
                    return "бюджет ↑, продажи ↓ → эффективность плохая"
                if spend_d < 0 and sales_d > 0:
                    return "бюджет ↓, продажи ↑ → эффективность хорошая"
                if spend_d < 0 and sales_d < 0:
                    return "бюджет ↓, продажи ↓"
                if spend_d > 0 and sales_d > 0:
                    return "бюджет ↑, продажи ↑"
                return "—"
            def _mk_row(label: str, spend_val, spend_d, sales_val, sales_d) -> str:
                cmt = []
                if spend_d is not None:
                    cmt.append("бюджет ↑" + f"{abs(spend_d):.1f}%" if spend_d > 0 else "бюджет ↓" + f"{abs(spend_d):.1f}%")
                if sales_d is not None:
                    cmt.append("продажи ↑" + f"{abs(sales_d):.1f}%" if sales_d > 0 else "продажи ↓" + f"{abs(sales_d):.1f}%")
                cmt.append(_eff_label(spend_d, sales_d))
                comment = ", ".join([p for p in cmt if p]) if cmt else "—"
                return f"| {label} | {_fmt_idr(spend_val)} ({_fmt_pct(spend_d)}) | {_fmt_idr(sales_val)} ({_fmt_pct(sales_d)}) | {comment} |"
            lines.append(_mk_row("GOJEK", day_spend_j, spend_j_delta, day_ads_sales_j, sales_j_delta))
            lines.append(_mk_row("GRAB", day_spend_g, spend_g_delta, day_ads_sales_g, sales_g_delta))
            lines.append("")

            # External factors
            lines.append("Внешние факторы:")
            lines.append(f"- Праздник: {'да' if is_hol else 'нет'}{(' (возможна низкая доступность курьеров)') if is_hol else ''}")
            if rain is not None:
                lines.append(f"- Погода: дождь {rain:.1f} мм")
            # Cancellations snapshot
            canc_g_day = int(qg.iloc[0]["cancelled_orders"]) if (not qg.empty and pd.notna(qg.iloc[0]["cancelled_orders"])) else 0
            canc_j_day = int(qj.iloc[0]["cancelled_orders"]) if (not qj.empty and pd.notna(qj.iloc[0]["cancelled_orders"])) else 0
            lines.append(f"- Отмены: GRAB {canc_g_day}; GOJEK {canc_j_day}")
            lines.append("")

            # Unified causes (marketing + ML + externals) — marketing-friendly wording
            causes: list[str] = []
            if spend_j_delta is not None and sales_j_delta is not None and spend_j_delta > 0 and sales_j_delta < 0:
                causes.append(f"реклама GOJEK оказалась неэффективной: бюджет увеличили (+{abs(spend_j_delta):.1f}%), а продажи упали (−{abs(sales_j_delta):.1f}%)")
            if spend_g_delta is not None and sales_g_delta is not None and spend_g_delta < 0 and sales_g_delta < 0:
                causes.append(f"снижение активности в GRAB: бюджет снизили (−{abs(spend_g_delta):.1f}%), продажи также просели (−{abs(sales_g_delta):.1f}%)")
            # External chain: holiday -> couriers -> cancels -> sales
            canc_g_avg = _safe_mean(qg_all.get('cancelled_orders'))
            canc_j_avg = _safe_mean(qj_all.get('cancelled_orders'))
            canc_up = ((canc_g_day and canc_g_avg and canc_g_day > canc_g_avg) or (canc_j_day and canc_j_avg and canc_j_day > canc_j_avg))
            if is_hol and canc_up:
                causes.append("праздник → меньше курьеров → больше отмен → падение продаж")
            elif is_hol:
                causes.append("праздник снизил доступность курьеров и спрос")
            if rain is not None and rain > 0:
                causes.append("дождь снизил готовность заказывать и увеличил ETA")
            # Add top 1–2 ML factors as plain language
            for f, v, s in neg[:2]:
                cmt = _comment_for(f, False)
                label = _pretty_feature_name(f)
                causes.append(f"{label}: {cmt}")
            # Deduplicate while preserving order
            seen = set()
            causes_unique = []
            for c in causes:
                key = c.lower()
                if key in seen:
                    continue
                seen.add(key)
                causes_unique.append(c)
            if causes_unique:
                lines.append("Причины:")
                for c in causes_unique:
                    lines.append(f"- {c}.")
                lines.append("")

            # Short summary and factor tables
            lines.append("Краткое резюме:")
            lines.append(f"- Просадка: −{_fmt_idr(drop_idr)} ({_fmt_pct(delta_pct)} к медиане/ожиданию).")
            if factor_rows_neg:
                topn = ", ".join([f"{_pretty_feature_name(f)} (−{_fmt_idr(m)})" for f, _, m in factor_rows_neg[:2]])
                lines.append(f"- Главные причины: {topn}.")
            lines.append("")
            if factor_rows_neg:
                lines.append("Негативные факторы (ТОП‑5):")
                lines.append("| Фактор | Вклад | Комментарий |")
                lines.append("|---|---:|---|")
                for f, s, money in sorted(factor_rows_neg, key=lambda x: (x[2], x[1]), reverse=True)[:5]:
                    lines.append(f"| {_pretty_feature_name(f)} | −{_fmt_idr(money)} ({s}%) | {_comment_for(f, False)} |")
                lines.append("")
            if pos:
                lines.append("Что помогло (до 2 факторов):")
                lines.append("| Фактор | Влияние | Комментарий |")
                lines.append("|---|---:|---|")
                for f, v, s in pos:
                    lines.append(f"| {_pretty_feature_name(f)} | {s}% | {_comment_for(f, True)} |")
                lines.append("")

            # Priorities helpers
            def _priority_tag(share: float) -> str:
                return "🔴" if share >= 15.0 else ("🟠" if share >= 7.0 else "🟢")
            def _confidence(share: float) -> str:
                return "High" if share >= 15.0 else ("Medium" if share >= 8.0 else "Low")

            # Build full lists by category (significant negative >=3%)
            neg_sorted = sorted(neg, key=lambda x: x[2], reverse=True)
            pos_sorted = sorted(pos, key=lambda x: x[2], reverse=True)
            cat_to_neg: Dict[str, list] = {}
            for f, v, s in neg_sorted:
                cat_to_neg.setdefault(_categorize_feature(f), []).append((f, s))
            cat_to_pos: Dict[str, list] = {}
            for f, v, s in pos_sorted:
                cat_to_pos.setdefault(_categorize_feature(f), []).append((f, s))

            # Narrative 1–2 предложения
            top_cats = sorted([(c, sum(s for _, s in fs)) for c, fs in cat_to_neg.items()], key=lambda x: x[1], reverse=True)
            if top_cats:
                label = {"Marketing": "реклама", "Operations": "операции на кухне/доставке", "External": "внешние условия (погода/календарь)", "Quality": "качество сервиса"}
                cats = [label.get(c, c) for c, _ in top_cats[:2]]
                if len(cats) >= 2:
                    lines.append(f"Этот день провалился из‑за комбинации {cats[0]} и {cats[1]}.")
                else:
                    lines.append(f"Основная причина просадки — {cats[0]}.")
                lines.append("")

            # Главные причины (все значимые по категориям)
            lines.append("Главные причины (по категориям):")
            for cat in ["Marketing", "Operations", "External", "Quality"]:
                if cat in cat_to_neg:
                    for f, s in cat_to_neg[cat]:
                        lines.append(f"{_priority_tag(s)} [{cat}] {_pretty_feature_name(f)} ({s}%): {_comment_for(f, False)}")
            # Всегда указываем погоду и праздник
            rain_share = _share(contrib_sum.get('rain', 0.0)) if 'rain' in contrib_sum else 0.0
            hol_share = _share(contrib_sum.get('is_holiday', 0.0)) if 'is_holiday' in contrib_sum else 0.0
            lines.append(f"• [External] Дождь: {rain if rain is not None else '—'} мм ({rain_share}%): {'снизил спрос' if (rain or 0)>0 else 'влияние незначительное'}")
            lines.append(f"• [External] Праздник: {'да' if is_hol else 'нет'} ({hol_share}%): {'снизил спрос' if is_hol else 'влияние незначительное'}")
            lines.append("")

            # Что смягчало
            if pos_sorted:
                lines.append("✅ Что смягчало:")
                added = 0
                for f, v, s in pos_sorted:
                    cmt = _comment_for(f, True)
                    if 'без изменений' in cmt:
                        continue
                    lines.append(f"• {_pretty_feature_name(f)} (+{s}%): {cmt}")
                    added += 1
                    if added >= 2:
                        break
                lines.append("")

            # Evidence lines are summarized in comments; skip protocol

            # What-if: отдельные рычаги и комбинированный сценарий
            try:
                row_idx = idxs[0]
                xrow = X.iloc[[row_idx]].copy(deep=True)
                for col in ["preparation_time_mean", "accepting_time_mean", "delivery_time_mean"]:
                    if col in xrow.columns and pd.notna(xrow.iloc[0][col]):
                        xrow.loc[xrow.index[0], col] = max(0.0, float(xrow.iloc[0][col]) * 0.9)
                if "outage_offline_rate_grab" in xrow.columns and pd.notna(xrow.iloc[0]["outage_offline_rate_grab"]):
                    xrow.loc[xrow.index[0], "outage_offline_rate_grab"] = 0.0
                if "ads_spend_total" in xrow.columns and pd.notna(xrow.iloc[0]["ads_spend_total"]):
                    xrow.loc[xrow.index[0], "ads_spend_total"] = float(xrow.iloc[0]["ads_spend_total"]) * 1.1
                uplift = float(model.predict(xrow)[0] - model.predict(X.iloc[[row_idx]])[0])
                # Individual levers
                base_pred = float(model.predict(X.iloc[[row_idx]])[0])
                # SLA only
                x_sla = X.iloc[[row_idx]].copy(deep=True)
                for col in ["preparation_time_mean", "accepting_time_mean", "delivery_time_mean"]:
                    if col in x_sla.columns and pd.notna(x_sla.iloc[0][col]):
                        x_sla.loc[x_sla.index[0], col] = max(0.0, float(x_sla.iloc[0][col]) * 0.9)
                uplift_sla = float(model.predict(x_sla)[0] - base_pred)
                # Budget only
                x_bud = X.iloc[[row_idx]].copy(deep=True)
                if "ads_spend_total" in x_bud.columns and pd.notna(x_bud.iloc[0].get("ads_spend_total")):
                    x_bud.loc[x_bud.index[0], "ads_spend_total"] = float(x_bud.iloc[0]["ads_spend_total"]) * 1.1
                else:
                    if "mkt_ads_spend_grab" in x_bud.columns and pd.notna(x_bud.iloc[0].get("mkt_ads_spend_grab")):
                        x_bud.loc[x_bud.index[0], "mkt_ads_spend_grab"] = float(x_bud.iloc[0]["mkt_ads_spend_grab"]) * 1.1
                    if "mkt_ads_spend_gojek" in x_bud.columns and pd.notna(x_bud.iloc[0].get("mkt_ads_spend_gojek")):
                        x_bud.loc[x_bud.index[0], "mkt_ads_spend_gojek"] = float(x_bud.iloc[0]["mkt_ads_spend_gojek"]) * 1.1
                uplift_bud = float(model.predict(x_bud)[0] - base_pred)
                # Offline only
                x_off = X.iloc[[row_idx]].copy(deep=True)
                for col in ["outage_offline_rate_grab", "offline_rate_grab"]:
                    if col in x_off.columns and pd.notna(x_off.iloc[0].get(col)):
                        x_off.loc[x_off.index[0], col] = 0.0
                uplift_off = float(model.predict(x_off)[0] - base_pred)

                # Confidence per lever (by category contributions)
                cat_contribs = {k: group_shares.get(k, 0.0) for k in ["Marketing", "Operations", "External"]}
                conf_mark = _confidence(cat_contribs.get("Marketing", 0.0))
                conf_ops = _confidence(cat_contribs.get("Operations", 0.0))
                conf_off = _confidence(10.0 if (grab_off_mins and grab_off_mins > 0) else 0.0)
                lines.append("Рекомендации:")
                lines.append(f"🔴 Срочно — перенаправить бюджет на кампании с высокой отдачей (+10%): ≈ {_fmt_idr(uplift_bud)} ({conf_mark})")
                lines.append(f"🟠 Важно — сократить среднее время приготовления/подтверждения на 10%: ≈ {_fmt_idr(uplift_sla)} ({conf_ops})")
                lines.append(f"🟢 Дополнительно — исключить оффлайн на платформах: ≈ {_fmt_idr(uplift_off)} ({conf_off})")
                lines.append("")
                lines.append(f"💰 Потенциал восстановления: ~{_fmt_idr(uplift)}")
                lines.append(f"💰 Потенциал восстановления: {_fmt_idr(min_pot)} — {_fmt_idr(max_pot)} (база: {_fmt_idr(base_pot)})")
                lines.append(f"📈 Прогноз восстановления (7 дней): {_fmt_idr(min_pot*7)} — {_fmt_idr(max_pot*7)}")
            except Exception:
                pass
            lines.append("")

        # Справка по периоду (без методологии)
        lines.append("Справка по периоду:")
        sub['heavy_rain'] = (sub['rain'].fillna(0.0) >= 10.0).astype(int)
        by_rain = sub.groupby('heavy_rain')['total_sales'].mean().to_dict()
        if 0 in by_rain:
            dr = (by_rain.get(1, by_rain[0]) - by_rain[0]) / (by_rain[0] or 1.0) * 100.0
            lines.append(f"  • 🌧️ Дождь: {_fmt_pct(dr)}")
        by_h = sub.groupby(sub['is_holiday'].fillna(0).astype(int))['total_sales'].mean().to_dict()
        if 0 in by_h:
            dh = (by_h.get(1, by_h[0]) - by_h[0]) / (by_h[0] or 1.0) * 100.0
            lines.append(f"  • 🎌 Праздники: {_fmt_pct(dh)}")
        lines.append("")
        lines.append("Источники: SQLite (grab_stats, gojek_stats), Open‑Meteo, Holidays cache")
        return "\n".join(lines)
    except Exception:
        return "8. 🚨 КРИТИЧЕСКИЕ ДНИ (ML)\n" + ("—" * 72) + "\nНе удалось построить раздел (ошибка обработки данных)."


def _section9_recommendations(period: str, restaurant_id: int) -> str:
    try:
        # Use SHAP over the whole period to prioritize levers; exclude trivial features
        start_str, end_str = period.split("_")
        df = pd.read_csv("/workspace/data/merged_dataset.csv", parse_dates=["date"]) if os.path.exists("/workspace/data/merged_dataset.csv") else pd.DataFrame()
        sub = df[(df.get("restaurant_id") == restaurant_id) & (df.get("date") >= start_str) & (df.get("date") <= end_str)].copy() if not df.empty else pd.DataFrame()
        lines = []
        lines.append("9. 🎯 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ")
        lines.append("—" * 72)
        if sub.empty:
            lines.append("Нет данных за период.")
            return "\n".join(lines)

        # Load model and compute feature importances
        model, features, background = load_artifacts("/workspace/ml/artifacts")
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