# 📊 АРХИТЕКТУРА ДАННЫХ ПОСЛЕ РАЗВЕРТЫВАНИЯ

## 🔄 **ЭТАПЫ РАБОТЫ С ДАННЫМИ:**

### **1. 🎯 ОБУЧЕНИЕ ML (одноразово)**
```
📁 /workspace/data/merged_dataset.csv (файл)
    ↓
🤖 python3 ml/training.py --csv merged_dataset.csv
    ↓
💾 /workspace/ml/artifacts/ (модель готова)
```

**ИЛИ** (если есть live API):
```
🌐 Live API → PostgreSQL
    ↓
🤖 python3 ml/training.py --from-db
    ↓ (экспорт в CSV)
📁 /workspace/data/live_dataset.csv
    ↓
💾 /workspace/ml/artifacts/ (модель готова)
```

### **2. 📈 РАБОТА ПРИЛОЖЕНИЯ (постоянно)**

#### **🔍 Вариант A: Только файл (если нет DATABASE_URL)**
```
DataAdapter.use_postgres = False
    ↓
📁 SQLite из merged_dataset.csv
    ↓
📊 Отчеты + AI анализ
```

#### **🌐 Вариант B: Live API (если есть DATABASE_URL)**
```
DataAdapter.use_postgres = True
    ↓
🌐 PostgreSQL с live API данными
    ↓
📊 Отчеты + AI анализ
```

---

## ⚡ **ОТВЕТ НА ВАШ ВОПРОС:**

### **🤖 ML ОБУЧЕНИЕ:**
- **По умолчанию:** обучается на **файле** `merged_dataset.csv`
- **Опционально:** может обучиться на **live API** с флагом `--from-db`

### **📊 РАБОТА ПРИЛОЖЕНИЯ:**
- **Автоматически определяет** источник по `DATABASE_URL`
- **Если есть DATABASE_URL:** работает с **live API** 
- **Если нет DATABASE_URL:** работает с **файлом**

---

## 🎯 **ПРАКТИЧЕСКИЕ СЦЕНАРИИ:**

### **🔧 Сценарий 1: Только файл (разработка)**
```bash
# Нет DATABASE_URL в окружении
export DATABASE_URL=""

# ML обучается на файле
python3 ml/training.py

# Приложение работает с SQLite
streamlit run streamlit_app.py
```
**Результат:** ML + приложение работают на одних данных из файла ✅

### **🌐 Сценарий 2: Гибрид (продакшн)**
```bash
# Есть DATABASE_URL для live API
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# ML обучается на файле (стабильные исторические данные)
python3 ml/training.py

# Приложение работает с live API
streamlit run streamlit_app.py
```
**Результат:** ML обучен на истории, приложение показывает актуальные данные ✅

### **🚀 Сценарий 3: Полный live (продакшн)**
```bash
# Есть DATABASE_URL для live API
export DATABASE_URL="postgresql://user:pass@host:5432/db"

# ML обучается на live данных
python3 ml/training.py --from-db

# Приложение работает с live API
streamlit run streamlit_app.py
```
**Результат:** ML + приложение работают на одних live данных ✅

---

## 🎯 **ИТОГОВЫЙ ОТВЕТ:**

**НЕТ, не обязательно!** 

🔄 **ML может обучиться на файле, а приложение работать с API**

**Это даже ЛУЧШЕ для продакшна:**
- 🎯 **ML обучается** на стабильных исторических данных (файл)
- 📊 **Приложение показывает** актуальные данные (API)
- 🔄 **Периодически переобучаем** ML на свежих данных

**Гибкость системы:**
- ✅ Работает в любой конфигурации
- ✅ Автоматически определяет источник данных
- ✅ Graceful fallback между источниками
- ✅ Готова к любому сценарию развертывания