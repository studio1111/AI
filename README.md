# AI Studio

پروژه‌ای برای اجرای چت متنی و تولید تصویر در Google Colab با رابط Gradio، مطابق ساختار فایل‌های ارسالی.

## امکانات
- چت متنی جریانی با مدل گفت‌وگوی ۸B و بارگذاری ۴ بیتی.
- تولید تصویر SDXL با مدل Pony، VAE اصلاح‌شده و زمان‌بند Euler Ancestral.
- تنظیم پرامپت، پرامپت منفی، نسبت تصویر، Steps، CFG و Seed.
- بارگذاری تنبل مدل‌ها و کنترل خطای نبود GPU.
- GitHub Actions برای بررسی کد و نگهداری ۳۰ اجرای اخیر workflowها.

## اجرا در Google Colab
1. فایل `colab/AI_Studio_Colab.ipynb` را باز کنید.
2. از Runtime → Change runtime type یک GPU انتخاب کنید. دسترسی رایگان به GPU تضمین‌شده نیست.
3. سلول‌های نصب و بررسی GPU را اجرا کنید.
4. رابط چت و رابط تصویر را جداگانه اجرا کنید؛ قبل از تغییر رابط، اجرای قبلی را متوقف کنید تا حافظه GPU آزاد شود.

## اجرا
```bash
pip install -r requirements.txt
python chat.py
```
برای تولید تصویر:
```bash
python image.py
```

شناسه مدل‌ها از طریق متغیرهای محیطی `CHAT_MODEL_ID`، `IMAGE_MODEL_ID` و `SDXL_VAE_ID` قابل تغییر است.

## تست‌ها
```bash
python -m compileall -q app.py chat.py image.py studio_utils.py
python -m pytest -q
```

## نکات
- مدل‌ها از Hugging Face دریافت می‌شوند. پیش از استفاده، مجوز و شرایط مدل‌ها را در صفحات رسمی آن‌ها بررسی کنید.
- لینک اشتراکی Gradio ممکن است برای هر دارنده لینک قابل دسترسی باشد. اطلاعات خصوصی وارد نکنید و پس از اتمام، نشست Colab را متوقف کنید.
- Google Colab ممکن است محدودیت زمان اجرا، حافظه و سهمیه GPU داشته باشد.
