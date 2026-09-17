# GarageAI — مراجعة البريزنتيشن v3 + النسخة المختصرة (20 سلايد)

تاريخ المراجعة: 2026-09-02. مبني على: memory files, README.md, SESSION_HANDOFF_20260827.md,
processed_data/full_benchmark_summary.csv, processed_data/ensemble_config.pkl,
processed_data/ensemble_report.txt, traditional_ml_report.txt.

---

## أولاً: أخطاء لازم تتصلح بالنسخة الحالية

| السلايد | الغلط | الصح |
|---|---|---|
| 40 "Live Today" | "73/73 passing backend tests" | الرقم قديم. الصح: **67 pytest + 62 Jest + 11 Playwright** (اتصحّح بالـ docs) |
| 42 "Dataset scale" | "Labels are the original poster's stated diagnosis, not re-verified by a certified mechanic" | **غلط الآن.** كل تسجيلة **راجعها ميكانيكي بالسمع** وأكّد/صحّح الصنف. القيد الصحيح: ما تأكّدنا بفحص فيزيائي للسيارة، وتمييز brake عن sway بالأذن ممكن يبقى ملتبس → احتمال ضجيج تسمية صغير متبقّي |
| 38 "Node.js Gateway" | "Pass-through API layer that decouples the client from ML internals" | الـ gateway **مش pass-through**: بيملك JWT auth + sessions/revocation + حفظ الـ history + rate-limiting؛ بس نداءين ML بيتمرروا للـ FastAPI |
| 36 "Per-Class (SVM)" | صف "Ensemble* 86.47%*" + "Sway misread as Belt (16 of 100)" | الـ ensemble المنشور = **89.47%** مش 86.47%. وعينات sway بالـ test = 44 مش 100 → استخدم **Recall 0.77** بدل "16 of 100" |
| 8 / 22 | "269 features" بدون تفصيل — و README لسا مكتوب فيه "84" | 269 هو الصح (بعد تطوير 84→269). تأكد تذكر: MFCC + **Δ/ΔΔ + spectral contrast + rolloff** + centroid/bandwidth/RMS/ZCR (HPSS-enhanced) |
| 40 "Coming next" | إعادة نشر الموديل التقليدي كـ XGBoost وإعادة ضبط الـ ensemble — مكتوب كأنه شبه منجز | هاي **متابعة غير منجزة** — حطها بوضوح تحت Future Work، مش "coming next this week" |

## ثانياً: نواقص انضافت (كنا تعبنا عليها وما كانت موجودة)

- **قصة الـ seed / الاستقرار** (كانت مفقودة تماماً) → سلايد 9 جديد.
- **Augmentation**: SMOTE اتجرّب وانرفض، mic/device frequency-response augmentation اتجرّب وانرفض،
  pitch-shift **تجنبناه عمداً**، N=2 نسخ، الفائدة الحقيقية الوحيدة = تقليل تذبذب الـ CNN → سلايد 10 موسّع.
- **بصمة الصوت (audio fingerprint)**: الصياغة الصريحة إن الموديل تعلّم recording-chain artifacts → سلايد 11-12.
- **SVC(probability=True) hang 17+ ساعة CPU** + قصص هندسية → سلايد 19 جديد.
- **أرقام الـ overfitting** (SVM 99.7→86.5، YAMNet 100→78.2) → سلايد 16.
- **الجهاز CPU-only، 6GB RAM، الرام الحرة نزلت لـ 62MB** → يفسّر معمارية opt-in → سلايد 18-19.
- **قصة threshold الـ "other" gate** (0.65 غلط → 0.46) → سلايد 17.
- **سبب 0/194 على Freesound ما اتأكد بالسمع** + الاستبعاد قرار سياسة كمان → سلايد 15.
- **تسرّب الـ split الساذج ضخّم الدقة ~2.3 نقطة** → سلايد 5.
- **XGBoost 92.6% ما اتفحص تحت الضغط أبداً** (بند مفتوح صريح) → سلايد 12 و 20.

## ثالثاً: مشاكل ترتيب

نقطة "XGBoost يسجّل أعلى بس منحتفظ بالـ ensemble للـ compression robustness" متكررة بـ 6 سلايدات
(10, 27, 29, 32, 35, 43). بالنسخة المختصرة انحكت **مرة وحدة بوضوح** بسلايد 8 + 13.

---

# النسخة المختصرة — 20 سلايد

> فواصل الأقسام (Section 1/2/3…) اختيارية وما بتنعدّ من الـ20. لو التمبليت بيلزمك فواصل،
> ضيفها وبتصير 20 محتوى + فواصل.

---

## Slide 1 — Title
- **Audio-Based Automotive Fault Classification — GarageAI**
- الفريق: Mohammed Yadak · Lama Qasem · Maryam Rawajbeh
- إشراف: Dr. Adnan Salman — An-Najah National University

---

## Slide 2 — The Problem & The Research Question
- **التشخيص اليدوي**: أصوات القشاط/الفرامل/التعليق مميزة، بس قراءتها بتعتمد على فني خبير حاضر —
  بطيء، مكلف، وغير متسق بين الفاحصين.
- **سؤال البحث**: هل موديل صوتي كبير مدرَّب مسبقاً فعلاً بيتفوّق على pipeline مهندَس بعناية
  ومخصّص للمهمة؟ — أو إن هاد الافتراض غير مفحوص؟ منجاوب تجريبياً.
- **النطاق**: 3 أعطال — Belt / Brake / Sway — + بوابة "other" بترفض أي صوت مش عطل سيارة قبل التصنيف.

---

## Slide 3 — System Overview (دياجرام)
- Record / Upload → Preprocessing → Feature Extraction → (10 model configs) → Weighted Ensemble
  → Fault Classification.
- بالتوازي: **"Other" gate** بيشتغل أول شي؛ **Ask GarageAI** (RAG chat) للأسئلة بعد النتيجة.
- 3 خدمات شغّالة end-to-end: React frontend · Node gateway · FastAPI inference.

---

## Slide 4 — Dataset
- **893 تسجيلة حقيقية** (مش صناعية): Belt 302 · Brake 294 · Sway 297.
- **47 براند**, **540 توليفة make/model/year**. مصدر عام: مجتمعات ومساهمون في مجال السيارات.
- **التسمية**: كل تسجيلة راجعها ميكانيكي بالسمع وأكّد/صحّح الصنف (بما فيها ~220 مقطع belt من Reddit).
  ما في فحص فيزيائي مستقل للسيارة.
- 9 ملفات ما انطابقت بين الإكسل والمجلدات (أخطاء تسمية أصلية) — موثقة بـ `unmatched_files.csv`.

---

## Slide 5 — Data Preparation & Leakage Prevention
- **Preprocessing**: Mono · 22,050 Hz · قص الصمت (trim @25 dB) · تطبيع [-1,1] ·
  نافذة 5 ثواني بالضبط (حشو بالآخر) · حذف أي ملف < 0.3s بعد قص الصمت.
- **ليش 5 ثواني**: أصوات الأعطال دورية فمقطع قصير بيلتقط تكرارات؛ طول ثابت لازم لكل الموديلات؛ توازن معلومة/حساب.
- **Group-Aware + Class-Balanced Split** 70/15/15 = 626 / 134 / 133.
  السيارات مجمّعة بـ make/model/year (belt 217 · brake 104 · sway 168 مجموعة) — **نفس السيارة أبداً ما بتظهر
  بالتدريب والاختبار مع بعض**.
- تقسيم ساذج (بدون تجميع) كان **بيضخّم الدقة ~2.3 نقطة** عبر التسرّب.

---

## Slide 6 — Feature Extraction (مسارين متوازيين)
- **Traditional ML**: **269 خاصية صوتية مهندَسة** — MFCC + Δ/ΔΔ, spectral contrast, spectral rolloff,
  centroid, bandwidth, RMS, ZCR (متوسطات + انحرافات على 5 ثواني، HPSS-enhanced). التاريخ: 84 → 269.
- **CNN**: Log-Mel spectrogram **128 × 216** (تردد × زمن) — بدون تلخيص يدوي.
- PCA + heatmap: الأصناف الثلاثة بتنفصل جيداً، كل صنف له بصمة زمن-تردد مميزة.

---

## Slide 7 — Models Compared: 10 Configurations, 1 Protocol
- **Traditional ML** — SVM (RBF) · Random Forest · XGBoost على الـ269 خاصية، grid-search
  (26 توليفة hyperparameters).
- **CNN من الصفر** — 3 بلوكات Conv (16→32→64) + BatchNorm + MaxPool، Global Avg Pool، Dense(64)،
  Dropout، class weights، SpecAugment.
- **6 backbones مجمّدة** — YAMNet · PANNs/CNN14 · AST · CLAP · PaSST · BEATs (frozen embedding + classifier خفيف).
- **fine-tune حقيقي واحد** — EfficientAT (MobileNetV3 `mn10_as`), آخر 3 بلوكات مفكوكة (~71% من الأوزان).
- **Weighted Ensemble**.
- **البروتوكول**: تقسيم بدون تسرّب، **5 seeds**, الداتا الكاملة (626)، نفس المقاييس (Accuracy/Precision/Recall/Macro-F1/Confusion).

---

## Slide 8 — Results: All 10 Configurations (5-seed mean)
- XGBoost **92.6%** ← أفضل موديل مفرد
- CLAP 90.2 · PaSST 90.2 · EfficientAT 89.2 · **Ensemble (deployed) 89.47** · AST 88.0 ·
  BEATs 86.5 · YAMNet 83.5 · PANNs 82.0 · CNN 78.2
- **العنوان**: على الداتا الكاملة، **ولا موديل transfer تفوّق على الموديل التقليدي المهندَس** —
  الصيغة القوية لنتيجة فجوة الدومين.
- الـ embeddings المجمّدة بتخسر غالباً؛ بس transformer قوي بذاته أو fine-tune حقيقي بيقدر ينافس.
- الـ ensemble المنشور (89.47%) **محتفَظ فيه لمقاومة الضغط المُتحقَّق منها، مش لأنه الأعلى نظيفاً** (بيتشرح سلايد 13).

---

## Slide 9 — Seed & Stability
- 5 seeds مستقلة لكل توليفة.
- الموديلات المجمّدة (YAMNet/PANNs/AST/CLAP/PaSST/BEATs): **std = 0.00** —
  استخراج embeddings حتمي + تدريب classifier حتمي على داتا كاملة → كل الـ5 seeds نفس الرقم بالضبط.
  **هاد متوقّع، مش bug** — يُذكر صراحة.
- التذبذب الحقيقي بس بالـ **CNN من الصفر** (random init): std ≈ 11%، فيه run واحد سيء الحظ نزل 56%.
- الـ augmentation **بينصّف تذبذب الـ CNN** (11% → 4.4%) حتى لو ما رفع المتوسط — فائدة استقرار للإنتاج.

---

## Slide 10 — Data Augmentation: 90-Run Study, Net Negative
- **Waveform** (train فقط، N=2 نسخ): Time Shift · Volume Change · Add Noise.
  **Pitch-shift تجنبناه عمداً** — ممكن يغيّر الخاصية التشخيصية الحقيقية للعطل.
- **SpecAugment**: إخفاء شريط تردد/زمن — على mel spectrogram الـ CNN فقط، أبداً على val/test.
- **90 تجربة** (9 موديلات × 5 seeds × مع/بدون): **ولا موديل استفاد**. Traditional ML (−7 نقاط)، AST، PaSST
  اتضرروا بفرق إحصائي حقيقي؛ الباقي داخل التذبذب. الفائدة الحقيقية الوحيدة = تقليل تذبذب الـ CNN.
- **SMOTE**: مفحوص جوّا الـ CV folds (اختبار أقدم مسرّب أظهر +5 نقاط وهمية) → ما ساعد.
- **mic/device frequency-response augmentation**: مفحوص → **مرفوض** (خسّر مقاومة الضغط الحقيقية مقابل مكسب صناعي أصغر).
- نتيجة سلبية اتبلّغت عن قصد. الرافعة الوحيدة المتبقّية = تسجيلات حقيقية جديدة مُصنّفة صح.

---

## Slide 11 — The Audio "Fingerprint" Problem
- الموديل: **89.5% على WAV نظيف** — بس pipeline التطبيق الحقيقي
  (browser MediaRecorder → WebM/Opus → ffmpeg decode) نزّله لـ **62.4% @ 32 kbps**،
  و **Brake انهار من 97.7% → 18.2%**.
- **السبب الجذري**: الموديل تعلّم **بصمة سلسلة التسجيل** (كيف اتلقط المقطع واترمّز أصلاً)
  بدل الخاصية الصوتية الحقيقية للـ belt/brake/sway.

---

## Slide 12 — Compression-Matched Fix + Result
- **الإصلاح**: كل تسجيلة **تدريب** (أبداً val/test) بتمرّ round-trip عبر نفس pipeline الضغط الحقيقي
  عند 16/32/64 kbps وتُضاف كداتا تدريب إضافية.
- **النتيجة @ 32 kbps**: 62.4% → **88.0%**؛ Brake وحده: 18.2% → **88.6%**؛
  كلفة صغيرة على النظيف: 89.5% → 86.5%؛ الـ ensemble الكامل: 86.7% نظيف → 93.3% تحت الضغط.
- هاي النسخة المضغوطة-المُدرَّبة هي **يلي شغّالة بالإنتاج اليوم**.
- **بند مفتوح**: XGBoost 92.6% رقم على صوت نظيف فقط — **ما اتفحص أبداً تحت WebM/Opus**.

---

## Slide 13 — Ensemble: Design & Why It's Kept
- **Weighted average**: Traditional ML **0.30** · CNN **0.60** · YAMNet **0.10** (grid-search؛
  4 من 5 folds بالـ CV تقاربوا على نفس الأوزان).
- **مش stacking**: single-split أظهر تفوّق stacking بـ 0.67 نقطة F1 — nested 5-fold CV بيّن إنه تذبذب
  على 134 عينة validation (stacking فاز 3/5 folds بس).
- محتفَظ فيه **لمقاومة الضغط المُتحقَّق منها**، مش لأعلى دقة نظيفة.
- PANNs و EfficientAT محمّلين **للمقارنة فقط** — وزن صفر بالدمج.

---

## Slide 14 — Ensemble: Rejected Refinements (موثّقة عن قصد)
- **الشكوى**: مستخدم بلّغ "Brake 55%" والموديلات متعارضة بشدة → تتبّعناها لاختيار stacking-vs-weights
  المدفوع بالضجيج على validation صغير → تحويل لـ weighted average المستقر + تقليل تأثير YAMNet بحالات التعارض.
- **محاولة retune**: دمج EfficientAT المُدرَّب بالأوزان (وزن 0.64) رفع الدقة النظيفة بالـ CV لـ **90.23% (+0.76)** —
  بس **انهار لـ 76.2% تحت الضغط الحقيقي** (مقابل 84.0% للمنشور) لأن EfficientAT ما تدرّب على الضغط → **مرفوض**.

---

## Slide 15 — External Data (Freesound): Tested & Rejected
- الفكرة: مع 893 تسجيلة بس، جرّبنا صوت Freesound مرخّص (520 عام + 135 brake بعد الفلترة) لتكبير التدريب.
- الاختبار: agreement-scoring مقابل الموديل المنشور، إضافة لـ **التدريب فقط**, مقارنة head-to-head على 4 موديلات.
- النتيجة: XGBoost غير متأثر أو متضرر؛ CNN اتضرر بكل الـ6 توليفات؛ **بس PANNs/CNN14 استفاد** (82.0 → 85.0).
  الموديل المنشور سجّل **0/194** على مقاطع Freesound "brake".
- سبب الـ 0/194 (فجوة دومين مقابل تسمية غلط) **ما اتأكّد بالسمع**. القرار: استُبعد — نتيجة سلبية + قرار سياسة
  (تسميات Freesound غير موثوقة).

---

## Slide 16 — Per-Class Performance & Confusion
- الـ ensemble المنشور: Belt P0.83 / R0.98 · Brake P0.93 / R0.93 · **Sway P0.94 / R0.77**.
- **Sway هو الأصعب لكل الموديلات** — طقطقة/نقر متقطع أقرب لضجيج عام، بعكس صرير القشاط النغمي (~500 Hz)
  وصرير الفرامل ضيّق النطاق (5–6 kHz). التشوّش الرئيسي: sway تُقرأ belt.
- **Overfitting مقيس** (على الموديلات المنشورة، features مُعاد استخراجها): SVM 99.7% train → 86.5% test؛
  YAMNet 100% → 78.2%؛ CNN الفجوة أصغر بكثير. موثّق كقيد — **ما اتطبّق تخفيف بعد**.

---

## Slide 17 — The "Other" Gate
- RandomForest مُشرَف رباعي الأصناف بيشتغل **قبل** أي موديل belt/brake/sway:
  **95.0%** دقة 4-أصناف، **100% (31/31)** تعميم على مجموعة محجوبة.
- بيانات "other" = **100% صناعية** (engine hum, impulse train, road noise, formant voice, tones, noise, chirps)
  + أصوات نظام Windows — **بدون Freesound عمداً**.
- قاعدة النشر: **P(other) ≥ 0.46** (مش argmax). اختيار 0.65 الأول سبّب regression على probe محجوب →
  ضُبط على **مجموعتين** (عينات الموديل + probe أصعب).
- الكلفة المتبقّية: **2/267 (0.75%)** عطل حقيقي بينرفض كـ "other" — موثّقة ومقبولة.

---

## Slide 18 — System & Deployment
- **React** (Vite/TS/Tailwind, 5173): تسجيل مايك + رفع + نتائج حيّة + dark/light + severity checklist +
  history + JWT + مؤشر خطورة.
- **Node/Express + Swagger gateway** (5000): **بيملك** الـ auth + sessions/revocation + حفظ history +
  rate-limiting؛ بس نداءين ML بيتمرروا.
- **FastAPI inference** (8001, **جهاز CPU-only**): 5 موديلات always-on (Traditional, CNN, YAMNet, PANNs,
  EfficientAT) + "other" gate + ensemble؛ AST/CLAP/PaSST/BEATs **opt-in** بسبب الرام.
- **Ask GarageAI**: RAG — `multilingual-e5-small` embeddings فوق قاموسين عربيين منسّقين +
  Gemini `gemini-3.5-flash-lite`، بوابة `<<NO_MATCH>>`، وسم `<<SEVERITY>>`، عربي عامي، **استشاري فقط**،
  ممنوع يخترع تشخيص برّا المقاطع المسترجَعة.
- الاختبارات: **67 pytest + 62 Jest + 11 Playwright**.

---

## Slide 19 — Engineering Challenges & Lessons
- **`SVC(probability=True)`** فعّل معايرة Platt الداخلية (5-fold) عند الـ fit حتى بدون `predict_proba` →
  **علّق 17+ ساعة CPU** قبل ما ننتبه؛ الإصلاح: تأجيل `probability=True` للموديل الفايز بس (7 ملفات).
- **staleness trap**: تقارير `processed_data/` ممكن تصف موديل أقدم مش المنشور → دايماً نتحقق من الموديل المحمّل
  فعلياً مقابل تقريره.
- **ما نثق أبداً بفرق دقة CNN من run واحد** (تذبذب seed ضخم) — دايماً ≥ 5 seeds.
- **جهاز تطوير 6GB، الرام الحرة نزلت لـ 62MB** → هي السبب المعماري لـ opt-in models.

---

## Slide 20 — Conclusion & Future Work
- **المُنجَز**: مقارنة 10 توليفات بدون تسرّب · pipeline 269 خاصية · ensemble منشور مقاوم للضغط
  (89.47% نظيف / ~84% واقعي تحت الضغط) · تطبيق 3 طبقات شغّال + "other" gate + RAG chat ·
  **كل بديل مرفوض موثّق**.
- **جواب سؤال البحث**: pipeline مهندَس ومخصّص للمهمة (XGBoost 92.6%) **تفوّق على كل** backbone كبير مدرَّب
  مسبقاً على هاي المشكلة.
- **Future Work**: compression-augment لـ EfficientAT/AST ليدخل backbone أقوى للـ ensemble · تسجيلات Sway حقيقية
  جديدة (الرافعة #1) · تنظيم لتقليل الـ overfitting · فحص XGBoost تحت الضغط · نشر on-device/offline ·
  تحقق فيزيائي مستقل من التسميات.
