# AI Studio ✦

محیط آزمایشی متن‌باز برای اجرای **چت‌بات متنی** و **تولید تصویر** با رابط Gradio روی Google Colab، با تمرکز بر GPUهای محدود مانند NVIDIA T4.

## امکانات

- رابط فارسی و ساده با دو بخش گفت‌وگو و تولید تصویر.
- چت مدل متنی با پاسخ جریانی (streaming)، تاریخچه گفتگو و تنظیمات system prompt، تعداد توکن و دما.
- تولید تصویر با مدل SDXL، پرامپت منفی، ابعاد متنوع، تعداد گام‌ها، guidance و seed.
- بارگذاری تنبل مدل‌ها و تخلیه مدل قبلی برای کاهش خطای کمبود حافظه GPU.
- متغیرهای محیطی برای انتخاب مدل‌های جایگزین.
- تست‌های سبک و GitHub Actions برای بررسی کد پایتون.
- گردش‌کار زمان‌بندی‌شده برای نگه‌داشتن ۳۰ اجرای اخیر GitHub Actions.

## اجرای سریع در Google Colab

1. نوت‌بوک [AI_Studio_Colab.ipynb](colab/AI_Studio_Colab.ipynb) را در Google Colab باز کنید.
2. از مسیر **Runtime → Change runtime type**، شتاب‌دهنده GPU را انتخاب کنید. دسترسی رایگان به GPU و زمان اجرای آن تضمین‌شده نیست.
3. سلول‌ها را به‌ترتیب اجرا کنید. دانلود اولیه مدل‌ها چند دقیقه طول می‌کشد.
4. در صورت نمایش خطای حافظه، Runtime را راه‌اندازی مجدد کنید و فقط یک مدل را در هر نوبت اجرا کنید.

## اجرای دستی روی ماشین دارای CUDA

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

در ویندوز، فعال‌سازی محیط مجازی از مسیر `.venv\\Scripts\\activate` انجام می‌شود. برای اجرای مدل‌ها باید نسخه سازگار PyTorch و CUDA نصب باشد. روی سیستم بدون GPU، برنامه عمداً اجرای مدل را متوقف می‌کند و پیام راهنما نشان می‌دهد.

## انتخاب مدل‌ها

مدل‌های پیش‌فرض در اولین اجرا از Hugging Face دریافت می‌شوند:

- گفت‌وگو: `Qwen/Qwen2.5-7B-Instruct` با بارگذاری ۴ بیتی.
- تصویر: `stabilityai/stable-diffusion-xl-base-1.0`.

برای انتخاب مدل دیگر، قبل از اجرا متغیرهای محیطی را تنظیم کنید:

```bash
export CHAT_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
export IMAGE_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
python app.py
```

توجه داشته باشید که مجوز، محدودیت‌ها و شرایط استفاده هر مدل به مخزن رسمی همان مدل وابسته است. مدل‌های بزرگ ممکن است روی T4 جا نشوند یا به تنظیمات متفاوت نیاز داشته باشند.

## تست و کنترل کیفیت

```bash
python -m pip install pytest
python -m compileall -q app.py studio_utils.py
python -m pytest -q
```

GitHub Actions این بررسی‌ها را هنگام push و pull request اجرا می‌کند. گردش‌کار **Keep latest 30 workflow runs** هفتگی اجرا می‌شود و اجرای دستی نیز دارد.

## نکات امنیت و حریم خصوصی

- هرگز توکن Hugging Face، کلید API یا رمز عبور را داخل کد یا مخزن قرار ندهید.
- لینک اشتراکی Gradio ممکن است برای هر دارنده لینک قابل دسترسی باشد؛ داده حساس وارد نکنید و پس از استفاده Runtime را متوقف کنید.
- محتوای ورودی و خروجی مدل‌ها را بررسی کنید و شرایط استفاده مدل‌ها و قوانین محل خود را رعایت کنید.
- Google Colab رایگان، GPU دائمی یا فضای ذخیره‌سازی پایدار تضمین نمی‌کند؛ فایل‌های تولیدی را جداگانه ذخیره کنید.

## ساختار پروژه

```text
AI/
├── app.py
├── studio_utils.py
├── requirements.txt
├── tests/
│   └── test_studio_utils.py
├── colab/
│   └── AI_Studio_Colab.ipynb
└── .github/workflows/
    ├── ci.yml
    └── cleanup-runs.yml
```

## مجوز

تا زمانی که مجوز مشخصی به این مخزن اضافه نشده، کد را نباید به‌طور خودکار دارای مجوز بازاستفاده عمومی فرض کرد. مجوز هر مدل وابسته به ناشر آن است.
