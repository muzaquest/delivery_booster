from __future__ import annotations

from typing import Optional, Dict
from datetime import date
import sqlite3
import pandas as pd

from app.report_basic import (
    build_basic_report,
    build_marketing_report,
    build_quality_report,
)
from etl.data_loader import get_engine


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

    lines = []
    lines.append("📊 1. ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ")
    lines.append("—" * 72)
    lines.append(f"💰 Общая выручка: {total_rev} (GRAB: {grab_rev} + GOJEK: {gojek_rev})")
    lines.append(f"📦 Общие заказы: {total_orders}")
    lines.append(f"   ├── 📱 GRAB: {int(grab.get('orders') or 0)}")
    lines.append(f"   └── 🛵 GOJEK: {int(gojek.get('orders') or 0)}")
    return "\n".join(lines)


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


def generate_full_report(period: str, restaurant_id: int) -> str:
    basic = build_basic_report(period, restaurant_id)
    marketing = build_marketing_report(period, restaurant_id)
    quality = build_quality_report(period, restaurant_id)
    finance = basic.get("finance") if isinstance(basic, dict) else None

    parts = []
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
    return "\n".join(parts)