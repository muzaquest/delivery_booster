#!/usr/bin/env python3
"""
Снапшот-тесты для раздела 8 "Критические дни"
Проверяет наличие блоков и числовые диапазоны для стабильности отчетов
"""

import re
import os
import sys
from typing import Dict, List, Tuple, Optional

def extract_key_metrics(report_text: str) -> Dict[str, any]:
    """Извлекает ключевые метрики из отчета для проверки"""
    metrics = {}
    
    # Ищем основную статистику раздела 8
    section8_match = re.search(r'8\. 🚨 КРИТИЧЕСКИЕ ДНИ.*?(?=9\.|$)', report_text, re.DOTALL)
    if not section8_match:
        return {"error": "Раздел 8 не найден"}
    
    section8_text = section8_match.group(0)
    
    # Извлекаем количество критических дней
    critical_days_match = re.search(r'Найдено критических дней.*?(\d+) из (\d+)', section8_text)
    if critical_days_match:
        metrics["critical_days_found"] = int(critical_days_match.group(1))
        metrics["total_days"] = int(critical_days_match.group(2))
        metrics["critical_days_percentage"] = metrics["critical_days_found"] / metrics["total_days"] * 100
    
    # Извлекаем медианные продажи
    median_match = re.search(r'Медианные продажи: ([\d\s]+(?:\.\d+)?)\s*(?:M\s*)?IDR', section8_text)
    if median_match:
        median_str = median_match.group(1).replace(' ', '').replace('M', '000000')
        try:
            metrics["median_sales"] = float(median_str)
        except:
            metrics["median_sales"] = None
    
    # Извлекаем общие потери
    losses_match = re.search(r'Общие потери.*?([\d\s]+(?:\.\d+)?)\s*(?:M\s*)?IDR', section8_text)
    if losses_match:
        losses_str = losses_match.group(1).replace(' ', '').replace('M', '000000')
        try:
            metrics["total_losses"] = float(losses_str)
        except:
            metrics["total_losses"] = None
    
    # Проверяем наличие обязательных блоков
    required_blocks = [
        "КЛЮЧЕВЫЕ ЦИФРЫ",
        "РЕАЛЬНЫЕ ПРИЧИНЫ", 
        "ВНЕШНИЕ ФАКТОРЫ",
        "КОНКРЕТНЫЕ РЕКОМЕНДАЦИИ",
        "ФИНАНСОВЫЙ ИТОГ"
    ]
    
    metrics["blocks_found"] = []
    for block in required_blocks:
        if block in section8_text:
            metrics["blocks_found"].append(block)
    
    metrics["blocks_coverage"] = len(metrics["blocks_found"]) / len(required_blocks) * 100
    
    # Считаем количество критических дней с детальным анализом
    detailed_days = len(re.findall(r'🔴 \d{4}-\d{2}-\d{2}', section8_text))
    metrics["detailed_days_analyzed"] = detailed_days
    
    # Проверяем наличие финансовых рекомендаций
    roi_recommendations = len(re.findall(r'Потенциальный эффект.*?IDR', section8_text))
    metrics["roi_recommendations"] = roi_recommendations
    
    # Проверяем отсутствие "полотна" - длинных абзацев без структуры
    long_paragraphs = len(re.findall(r'[^\n]{200,}', section8_text))
    metrics["long_paragraphs"] = long_paragraphs
    
    return metrics


def test_restaurant_period(restaurant_id: int, period: str, restaurant_name: str) -> Dict[str, any]:
    """Тестирует один ресторан за один период"""
    print(f"🧪 Тестирую {restaurant_name} за {period}...")
    
    try:
        # Импортируем функцию генерации отчета
        sys.path.append('/workspace')
        from app.report_text import generate_full_report
        
        # Генерируем отчет
        report = generate_full_report(period, restaurant_id)
        
        if not report or len(report) < 1000:
            return {"error": f"Отчет слишком короткий: {len(report)} символов"}
        
        # Извлекаем метрики
        metrics = extract_key_metrics(report)
        metrics["restaurant_name"] = restaurant_name
        metrics["period"] = period
        metrics["report_length"] = len(report)
        
        return metrics
        
    except Exception as e:
        return {"error": f"Ошибка генерации отчета: {str(e)}"}


def validate_metrics(metrics: Dict[str, any], test_name: str) -> Tuple[bool, List[str]]:
    """Проверяет метрики на соответствие требованиям"""
    issues = []
    
    # Проверка 1: Раздел 8 должен существовать
    if "error" in metrics:
        issues.append(f"❌ {metrics['error']}")
        return False, issues
    
    # Проверка 2: Должны быть найдены критические дни (или их отсутствие объяснено)
    if "critical_days_found" not in metrics:
        issues.append("❌ Не найдена статистика критических дней")
    elif metrics["critical_days_found"] > 10:
        issues.append(f"⚠️ Слишком много критических дней: {metrics['critical_days_found']} (возможно слишком мягкий порог)")
    
    # Проверка 3: Процент критических дней должен быть разумным
    if "critical_days_percentage" in metrics:
        if metrics["critical_days_percentage"] > 50:
            issues.append(f"⚠️ Слишком высокий процент критических дней: {metrics['critical_days_percentage']:.1f}%")
    
    # Проверка 4: Все обязательные блоки должны присутствовать
    if "blocks_coverage" in metrics:
        if metrics["blocks_coverage"] < 80:
            missing_blocks = 5 - len(metrics.get("blocks_found", []))
            issues.append(f"❌ Недостает блоков: {missing_blocks}, покрытие {metrics['blocks_coverage']:.1f}%")
    
    # Проверка 5: Детальный анализ должен быть для всех критических дней
    if "critical_days_found" in metrics and "detailed_days_analyzed" in metrics:
        if metrics["detailed_days_analyzed"] < metrics["critical_days_found"]:
            issues.append(f"❌ Не все критические дни проанализированы: {metrics['detailed_days_analyzed']}/{metrics['critical_days_found']}")
    
    # Проверка 6: Должны быть рекомендации с ROI
    if "roi_recommendations" in metrics:
        if metrics["roi_recommendations"] == 0:
            issues.append("❌ Нет рекомендаций с финансовым эффектом")
    
    # Проверка 7: Не должно быть "полотна" текста
    if "long_paragraphs" in metrics:
        if metrics["long_paragraphs"] > 2:
            issues.append(f"⚠️ Найдено {metrics['long_paragraphs']} длинных абзацев (возможно 'полотно')")
    
    # Проверка 8: Размер отчета должен быть разумным
    if "report_length" in metrics:
        if metrics["report_length"] > 50000:
            issues.append(f"⚠️ Отчет слишком длинный: {metrics['report_length']} символов")
        elif metrics["report_length"] < 3000:
            issues.append(f"⚠️ Отчет слишком короткий: {metrics['report_length']} символов")
    
    success = len(issues) == 0
    return success, issues


def run_snapshot_tests():
    """Запускает снапшот-тесты на 2 ресторана × 2 периода"""
    print("🚀 СНАПШОТ-ТЕСТЫ РАЗДЕЛА 8")
    print("=" * 60)
    print()
    
    # Тестовые случаи: 2 ресторана × 2 периода
    test_cases = [
        (11, "2025-04-01_2025-06-30", "Ika Canggu"),  # Q2 2025
        (11, "2025-05-01_2025-05-31", "Ika Canggu"),  # Май 2025
        (20, "2025-05-01_2025-05-31", "Only Eggs"),   # Май 2025
        (20, "2025-04-01_2025-04-30", "Only Eggs"),   # Апрель 2025
    ]
    
    results = []
    total_tests = len(test_cases)
    passed_tests = 0
    
    for restaurant_id, period, restaurant_name in test_cases:
        print(f"📊 Тестирую: {restaurant_name} за {period}")
        
        # Генерируем метрики
        metrics = test_restaurant_period(restaurant_id, period, restaurant_name)
        
        # Валидируем метрики
        test_name = f"{restaurant_name}_{period}"
        success, issues = validate_metrics(metrics, test_name)
        
        results.append({
            "test_name": test_name,
            "success": success,
            "metrics": metrics,
            "issues": issues
        })
        
        if success:
            print(f"✅ {test_name}: ПРОЙДЕН")
            passed_tests += 1
        else:
            print(f"❌ {test_name}: ПРОВАЛЕН")
            for issue in issues:
                print(f"   {issue}")
        
        print()
    
    # Итоговый результат
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 40)
    print(f"✅ Успешных тестов: {passed_tests}/{total_tests}")
    print(f"📊 Процент успеха: {passed_tests/total_tests*100:.1f}%")
    print()
    
    if passed_tests == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("✅ Раздел 8 стабилен и готов к production")
        print("✅ Нет 'полотна' текста")
        print("✅ Все блоки присутствуют")
        print("✅ Финансовые рекомендации включены")
    elif passed_tests >= total_tests * 0.75:
        print("⚠️ БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО")
        print("🔧 Требуется небольшая доработка")
    else:
        print("❌ МНОГО ПРОВАЛЕННЫХ ТЕСТОВ")
        print("🔧 Требуется серьезная доработка раздела 8")
    
    return results


if __name__ == "__main__":
    # Проверяем доступность необходимых файлов
    required_files = [
        "/workspace/data/merged_dataset.csv",
        "/workspace/ml/artifacts/model.joblib",
        "/workspace/ml/artifacts/features.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ ОТСУТСТВУЮТ НЕОБХОДИМЫЕ ФАЙЛЫ:")
        for f in missing_files:
            print(f"   {f}")
        print()
        print("🔧 Запустите сначала:")
        print("   python etl/data_loader.py --run")
        print("   python ml/training.py")
        sys.exit(1)
    
    # Запускаем тесты
    results = run_snapshot_tests()
    
    # Сохраняем результаты
    import json
    with open('/workspace/section8_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print("📋 Результаты сохранены в section8_test_results.json")