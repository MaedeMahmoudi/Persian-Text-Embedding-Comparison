 
# 📊 تحلیل غزلیات سعدی با TF-IDF، SVD و امبدینگ

## 📝 توضیح پروژه

این پروژه به عنوان بخشی از درس **جبر خطی** انجام شده است. هدف اصلی، تبدیل متون فارسی (غزلیات سعدی) به بردارهای عددی با استفاده از روش‌های مختلف و سپس مقایسه‌ی شباهت معنایی بین آن‌ها است.

### روش‌های پیاده‌سازی شده:

1. **TF-IDF**: ساخت ماتریس وزنی بر اساس اهمیت آماری کلمات
2. **SVD**: کاهش ابعاد ماتریس TF-IDF برای استخراج الگوهای پنهان
3. **امبدینگ سفارشی**: ترکیب TF-IDF و SVD برای ساخت بردارهای معنایی با ابعاد کمتر

---

## 🗂️ ساختار پروژه

---

## 🛠️ نصب و راه‌اندازی

### ۱. کلون کردن پروژه

```bash
git clone <repository-url>
cd project4

۲. ساخت محیط مجازی
bash

python3 -m venv venv
source venv/bin/activate  # در لینوکس
# یا برای ویندوز:
# venv\Scripts\activate

۳. نصب وابستگی‌ها
bash

pip install -r requirements.txt

اگر فایل requirements.txt موجود نیست، کتابخانه‌ها را دستی نصب کنید:
bash

pip install numpy pandas scikit-learn hazm transformers sentencepiece torch

۴. اجرای پروژه
bash

cd scripts
python3 06_main.py# Persian-Text-Embedding-Comparison
