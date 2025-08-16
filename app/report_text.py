from __future__ import annotations

from typing import Optional, Dict
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
            # Top-10 by |impact|
            top10 = sorted(contrib_sum.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            total_abs = sum(abs(v) for v in contrib_sum.values()) or 1.0

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
                "SELECT sales, orders, ads_spend, ads_sales, offline_rate FROM grab_stats WHERE restaurant_id=? AND stat_date=?",
                eng, params=(restaurant_id, ds)
            )
            qj = pd.read_sql_query(
                "SELECT sales, orders, ads_spend, ads_sales, accepting_time, preparation_time, delivery_time, close_time FROM gojek_stats WHERE restaurant_id=? AND stat_date=?",
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

            lines.append(f"📉 КРИТИЧЕСКИЙ ДЕНЬ: {ds} (выручка: {_fmt_idr(total_sales_day)}; отклонение к медиане: {_fmt_pct(delta_pct)})")
            lines.append("—" * 72)
            # Factors table (concise)
            lines.append("🔎 ТОП‑факторы (ML):")
            for feat, val in top10:
                cat = _categorize_feature(feat)
                direction = "↑" if val > 0 else "↓"
                share = round(100.0 * abs(val) / total_abs, 1)
                lines.append(f"  • [{cat}] {feat}: {direction} вклад ~{_fmt_idr(abs(val))} ({share}%)")
            lines.append("")
            lines.append("📊 Вклад групп факторов:")
            for cat in ["Operations", "Marketing", "External", "Quality", "Other"]:
                if cat in group_shares:
                    lines.append(f"  • {cat}: {group_shares[cat]}%")
            lines.append("")
            lines.append("📅 Контекст дня:")
            # Platforms/offline
            lines.append(f"  • 📱 GRAB оффлайн: {_fmt_minutes_to_hhmmss(grab_off_mins)}")
            lines.append(f"  • 🛵 GOJEK оффлайн: {_hms_close(gojek_close)}")
            # Marketing
            if not qg.empty:
                gs = qg.iloc[0]
                roas_g = (float(gs["ads_sales"]) / float(gs["ads_spend"])) if (pd.notna(gs["ads_spend"]) and float(gs["ads_spend"])>0) else None
                lines.append(f"  • 🎯 GRAB: spend {_fmt_idr(gs['ads_spend'])}, ROAS {_fmt_rate(roas_g)}x")
            if not qj.empty:
                js = qj.iloc[0]
                roas_j = (float(js["ads_sales"]) / float(js["ads_spend"])) if (pd.notna(js["ads_spend"]) and float(js["ads_spend"])>0) else None
                lines.append(f"  • 🎯 GOJEK: spend {_fmt_idr(js['ads_spend'])}, ROAS {_fmt_rate(roas_j)}x")
            # Operations (GOJEK times)
            if not qj.empty:
                def _to_min(v):
                    s = str(v)
                    parts = s.split(":")
                    try:
                        if len(parts) == 3:
                            h, m, sec = parts
                            return int(h)*60 + int(m) + int(sec)/60.0
                    except Exception:
                        return None
                    try:
                        return float(s)
                    except Exception:
                        return None
                lines.append(f"  • ⏱️ Приготовление: {_fmt_rate(_to_min(qj.iloc[0].get('preparation_time')))} мин")
                lines.append(f"  • ⏳ Подтверждение: {_fmt_rate(_to_min(qj.iloc[0].get('accepting_time')))} мин")
                lines.append(f"  • 🚗 Доставка: {_fmt_rate(_to_min(qj.iloc[0].get('delivery_time')))} мин")
            # Weather/holiday
            lines.append(f"  • 🌧️ Дождь: {rain if rain is not None else '—'} мм; 🌡️ Темп.: {temp if temp is not None else '—'}°C; 🌬️ Ветер: {wind if wind is not None else '—'}; 💧Влажность: {hum if hum is not None else '—'}")
            lines.append(f"  • 🎌 Праздник: {'да' if is_hol else 'нет'}")
            lines.append("")
            # What-if: улучшение SLA и маркетинга, снятие оффлайна
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
                lines.append(f"🔮 What‑if (−10% SLA, +10% бюджет, без оффлайна): ожидаемый прирост ~{_fmt_idr(uplift)}")
            except Exception:
                pass
            lines.append("")

        # Диагностика периода: простые оценки эффекта дождя и праздников
        lines.append("Диагностика факторов за период:")
        sub['heavy_rain'] = (sub['rain'].fillna(0.0) >= 10.0).astype(int)
        by_rain = sub.groupby('heavy_rain')['total_sales'].mean().to_dict()
        if 0 in by_rain:
            dr = (by_rain.get(1, by_rain[0]) - by_rain[0]) / (by_rain[0] or 1.0) * 100.0
            lines.append(f"  • 🌧️ Дождь (простая разница средних): {_fmt_pct(dr)}")
        by_h = sub.groupby(sub['is_holiday'].fillna(0).astype(int))['total_sales'].mean().to_dict()
        if 0 in by_h:
            dh = (by_h.get(1, by_h[0]) - by_h[0]) / (by_h[0] or 1.0) * 100.0
            lines.append(f"  • 🎌 Праздники (простая разница средних): {_fmt_pct(dh)}")
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
            lines.append(f"  • [{_categorize_feature(f)}] {f}")
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