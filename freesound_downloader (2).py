"""
سكربت بيبحث بموقع Freesound عن أصوات:
- بريكات (break)
- عمود التوازن / جلود (sway)
- قشاط الدينمو (belt)

وبينزلهم بصيغة wav مع تسمية كل ملف حسب اللابل تبعه.

المتطلبات قبل التشغيل:
1) حساب على https://freesound.org وتسجيل تطبيق للحصول على API Key:
   - سجل دخول → روح ع https://freesound.org/apiv2/apply/
   - عبي الفورم (اسم التطبيق، وصف بسيط) وخد الـ API Key (اسمها Client secret/API key)
   - حط القيمة تبعتك مكان "PUT_YOUR_API_KEY_HERE" تحت

2) تثبيت المكتبات:
   pip install requests --break-system-packages

3) تثبيت ffmpeg (لازم لتحويل المعاينة من mp3 لـ wav):
   - Windows: نزل من https://ffmpeg.org/download.html وضيفه على PATH
   - Mac: brew install ffmpeg
   - Linux: sudo apt install ffmpeg

4) شغل السكربت:
   python freesound_downloader.py

ملاحظة مهمة:
- Freesound بيسمح بتنزيل "المعاينة" (preview) مباشرة عن طريق الـ API key بس،
  والمعاينة عادة mp3/ogg مش wav، فالسكربت بيحول المعاينة لـ wav تلقائياً عن طريق ffmpeg.
- لو بدك تنزل الملف الأصلي (original file) اللي بيكون غالباً wav فعلياً وبجودة أعلى،
  هاد بده OAuth2 (تسجيل دخول المستخدم) مش بس API key. اذا حابب هيك خبرني وبعطيك
  نسخة من الكود فيها OAuth2 flow كامل.

بخصوص الكميات الكبيرة (مئات الأصوات):
- السكربت هلق بيدور بكل الصفحات لكل كلمة بحث (مش بس أول 5 نتائج) لحد ما يوصل
  MAX_PER_LABEL تحت، أو تخلص النتائج المتوفرة فعلياً على الموقع لهاي الكلمة.
- ضفت تأخير بسيط (REQUEST_DELAY) بين كل طلب وطلب حتى ما ينحظر الحساب من كثر الطلبات.
- إذا كلمة بحث معينة ما عندها نتائج كافية على الموقع أصلاً، مش رح يقدر السكربت
  "يخترع" أصوات مش موجودة — العدد النهائي بيعتمد على وش متوفر فعلياً بمكتبة Freesound.
"""

import os
import time
import requests

# ============ الإعدادات ============
API_KEY = "PUT_YOUR_API_KEY_HERE"
BASE_URL = "https://freesound.org/apiv2"
OUTPUT_DIR = "sounds"

# كم صوت بدك تجمع (كحد أقصى) لكل تصنيف — مجموع الثلاث تصنيفات ممكن يوصل لمئات
# لو الكلمة ما عندها نتائج كافية، بيوقف عندها وبيكمل عالكلمة يلي بعدها
MAX_PER_LABEL = 200

# كم نتيجة يجيب بكل طلب واحد (أقصى شي بيقبله Freesound هو 150)
PAGE_SIZE = 150

# تأخير بالثواني بين كل طلب وطلب حتى ما ينحظر الحساب (Freesound: 60 طلب/دقيقة تقريباً)
REQUEST_DELAY = 1.1

# كل label مربوط بلستة كلمات بحث (مرادفات لنفس المشكلة) — تقدر تزيد/تحذف براحتك
# break  = صوت البريكات (صفير/احتكاك تيل الفرامل)
# sway   = صوت جلود/كراسي عمود التوازن (طقطقة/دق)
# belt   = صوت قشاط الدينمو (صفير عند الدوران/تعليق)
SEARCH_QUERIES = {
    "break": [
        "car brake squeal",       # صفير الفرامل
        "brake pad squeak",       # صرير تيل الفرامل
        "disc brake noise",       # صوت الفرامل القرصية
        "brake grinding",         # طحن/احتكاك الفرامل (تيل خلص)
        "car brake screech",      # صرير حاد للفرامل
        "brake squeak",          # صرير فرامل (عام)
        "brakes screech",        # صرير فرامل (عام)
        "car stopping brakes",   # صوت توقف بالفرامل
        "bicycle brake squeal",  # صفير فرامل دراجة (بديل قريب)
    ],
    "sway": [
        "sway bar rattle",             # طقطقة عمود التوازن
        "suspension bushing rattle",   # طقطقة جلدة الجمبازات
        "suspension clunk noise",      # صوت دق بالتعليق
        "car suspension knock",        # طرقعة بالتعليق
        "worn bushing noise",          # صوت جلدة مهترئة
        "car rattle bump",             # طقطقة عامة بالسيارة عند المطبات
        "car rough road rattle",       # طقطقة بطريق وعر
        "loose parts rattle car",      # طقطقة أجزاء مفكوكة
        "metal rattle noise",          # طقطقة معدنية عامة (مش شرط سيارة)
        "clunk sound",                 # صوت "كلنك/دقة" عامة
        "knocking rattle",             # طرقعة/طقطقة
        "car underbody noise",         # صوت من تحت السيارة
        "squeak rattle car",           # صرير وطقطقة بالسيارة
        "vibration rattle metal",      # اهتزاز وطقطقة معدنية
        "rubber bushing squeak",       # صرير جلدة مطاطية
        "chassis rattle",              # طقطقة هيكل السيارة
    ],
    "belt": [
        "serpentine belt squeal",   # صفير قشاط الدينمو
        "fan belt squeal",          # صفير قشاط المروحة/الدينمو
        "belt slip noise",          # صوت انزلاق القشاط
        "alternator belt noise",    # صوت قشاط الدينمو
        "engine belt screech",      # صرير قشاط المحرك
        "belt squeal",              # صفير قشاط (عام)
        "car engine squeal",        # صفير بالمحرك (عام)
        "rubber belt friction",     # احتكاك قشاط مطاطي
        "belt whine",                # أزيز قشاط
        "engine whine noise",        # أزيز بالمحرك
        "conveyor belt squeal",      # صفير سير ناقل (بديل قريب لصوت الاحتكاك المطاطي)
        "rubber friction squeak",    # صرير احتكاك مطاطي عام
        "pulley squeal",             # صفير بكرة (متعلق بحزام الدينمو)
        "machine belt noise",        # صوت حزام آلة (عام)
        "high pitch squeal engine",  # صفير حاد بالمحرك
        "car idle squeal",           # صفير أثناء دوران المحرك بالضبط
    ],
}
# =====================================


def search_sounds_page(query, page, page_size):
    """بيجيب صفحة وحدة من نتائج البحث."""
    headers = {"Authorization": f"Token {API_KEY}"}
    params = {
        "query": query,
        "fields": "id,name,previews,duration",
        "page_size": page_size,
        "page": page,
    }
    resp = requests.get(f"{BASE_URL}/search/text/", headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def search_sounds_all(query, max_results):
    """بيدور على كل الصفحات لكلمة بحث معينة لحد ما يوصل max_results أو تخلص النتائج."""
    all_results = []
    page = 1
    while len(all_results) < max_results:
        time.sleep(REQUEST_DELAY)  # تأخير بسيط لتفادي الحظر
        try:
            data = search_sounds_page(query, page, PAGE_SIZE)
        except requests.HTTPError as e:
            print(f"    خطأ بالبحث (صفحة {page}): {e}")
            break

        results = data.get("results", [])
        if not results:
            break  # ما في صفحات إضافية

        all_results.extend(results)

        if data.get("next") is None:
            break  # وصلنا آخر صفحة

        page += 1

    return all_results[:max_results]


def download_and_convert(label, sound, index):
    """بينزل معاينة الصوت (mp3) وبيحولها wav."""
    preview_url = sound["previews"].get("preview-hq-mp3") or sound["previews"].get(
        "preview-lq-mp3"
    )
    if not preview_url:
        print(f"  [تخطي] ما في preview لـ: {sound.get('name')}")
        return

    mp3_path = os.path.join(OUTPUT_DIR, f"{label}_{index}.mp3")
    wav_path = os.path.join(OUTPUT_DIR, f"{label}_{index}.wav")

    r = requests.get(preview_url)
    r.raise_for_status()
    with open(mp3_path, "wb") as f:
        f.write(r.content)

    # تحويل mp3 -> wav عن طريق ffmpeg
    result = os.system(f'ffmpeg -y -loglevel error -i "{mp3_path}" "{wav_path}"')
    os.remove(mp3_path)

    if result == 0 and os.path.exists(wav_path):
        print(f"  تم تنزيل: {wav_path}  (اسم الصوت الأصلي: {sound.get('name')})")
    else:
        print(f"  [فشل التحويل] {sound.get('name')} — تأكد إنه ffmpeg مثبت وعامل PATH")


def main():
    if API_KEY == "PUT_YOUR_API_KEY_HERE":
        print("لازم تحط الـ API Key تبعك أول بالمتغير API_KEY فوق بالسكربت.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, queries in SEARCH_QUERIES.items():
        print(f"\n===== تصنيف: {label} =====")
        file_index = 0  # عداد مستمر عشان ما تتكرر أسماء الملفات بين مرادفات نفس التصنيف
        seen_ids = set()  # لتفادي تنزيل نفس الصوت مرتين لو ظهر بأكتر من مرادفة

        for query in queries:
            if file_index >= MAX_PER_LABEL:
                print(f"  وصلنا الحد الأقصى ({MAX_PER_LABEL}) لهاد التصنيف، بتخطى باقي الكلمات.")
                break

            remaining = MAX_PER_LABEL - file_index
            print(f"  بدور بكلمة: '{query}' (بدي لحد {remaining} صوت إضافي) ...")

            results = search_sounds_all(query, remaining)

            if not results:
                print("    ما في نتائج لهاي الكلمة.")
                continue

            for sound in results:
                if file_index >= MAX_PER_LABEL:
                    break
                if sound["id"] in seen_ids:
                    continue  # نفس الصوت طلع بمرادفة سابقة، تخطاه
                seen_ids.add(sound["id"])
                download_and_convert(label, sound, file_index)
                file_index += 1

    print(f"\nخلصت! الملفات موجودة بمجلد: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
