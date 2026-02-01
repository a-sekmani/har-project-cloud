# شكل البيانات المتوقع من النظام (Edge → Cloud)

## 1. نقطة الدخول: `POST /v1/edge/events`

النظام يتوقع **Payload إطار واحد من الحافة** بالشكل التالي (EdgeFrameEvent v1).

### المطلوب

| الحقل | النوع | وصف |
|-------|--------|-----|
| `event_type` | string | **يجب أن يكون بالضبط** `"frame_event"` (ليس `"frame"`) |
| `source.device_id` | string | معرف الجهاز (مطلوب) |
| `source.session_id` | string | معرف الجلسة (مطلوب) |
| `frame.ts_unix_ms` | integer أو float | وقت الإطار بوحدة ميلي ثانية (Unix). يُقبل int أو float؛ داخل السحابة يُحوَّل إلى int |
| `persons` | array | **حقل مطلوب** (يجب وجوده في الـ payload). **يمكن أن تكون القيمة مصفوفة فارغة** `[]` |
| `persons[].track_id` | integer | معرف المسار للشخص (≥ 0) — عند وجود أشخاص |
| `persons[].keypoints` | array | **17 عنصراً**، كل عنصر كائن `{ name, x, y, c }`. الأسماء يجب أن تكون **مجموعة COCO-17** (الترتيب في الطلب غير مشترط؛ السحابة ترتّب حسب COCO-17 عند التطبيع) |

### صيغة النقطة (Keypoint)

كل عنصر في `persons[].keypoints` هو كائن:

```json
{ "name": "nose", "x": 320.5, "y": 180.2, "c": 0.9 }
```

- **name**: اسم النقطة من **COCO-17** (مثل nose, left_eye, right_eye, …). يجب أن تكون مجموعة الأسماء بالضبط COCO-17.
- **x**, **y**: إحداثيات النقطة. **حالياً في الـ ingestion تُعتبر دائماً بوحدة البكسل (pixel)**. النقاط غير المكتشفة يمكن أن تُرسل كـ -1.
- **c**: الثقة (confidence) بين 0 و 1.

النقاط غير المكتشفة: إذا `x` أو `y` = -1 أو `c` = 0 تُخزَّن كما هي في التطبيع الداخلي.

**ملاحظة عن الإحداثيات:** في edge الحالي الإحداثيات تُرسل كـ pixel. المثال minimal أدناه يستخدم قيم بكسل. دعم normalized (0..1) غير مفعّل في الـ ingestion حالياً.

### أولوية camera_id (حرفياً)

1. **source.camera_id** إن وُجد
2. **Query** `?camera_id=...`
3. **Header** `X-Camera-Id`
4. **Default** (مثلاً `"cam-1"` من الإعدادات)

### الحقول الإضافية

الحقول الإضافية مثل `bbox`, `coords`, `score` **مسموحة** ولا يرفض الطلب بسببها (`extra="allow"` على النماذج). لا يوجد حقل `confidence` على مستوى الشخص مطلوباً من الحافة.

### أسماء COCO-17 (بالترتيب المستخدم في التطبيع)

```
nose, left_eye, right_eye, left_ear, right_ear,
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle
```

الترتيب داخل الطلب **غير مشترط**؛ السحابة ترتّب النقاط حسب هذا الترتيب قبل تحويلها إلى `[[x,y,c], ...]` لتفادي أن تخرج نوافذ التجميع بترتيب خاطئ.

### مثال JSON minimal (إطار واحد، شخص واحد، 17 نقطة — إحداثيات pixel)

```json
{
  "event_type": "frame_event",
  "source": {
    "device_id": "edge-1",
    "session_id": "sess-abc",
    "camera_id": "cam-1"
  },
  "frame": {
    "ts_unix_ms": 1737970000000
  },
  "persons": [
    {
      "track_id": 0,
      "keypoints": [
        { "name": "nose", "x": 320, "y": 180, "c": 0.9 },
        { "name": "left_eye", "x": 310, "y": 175, "c": 0.85 },
        { "name": "right_eye", "x": 330, "y": 175, "c": 0.87 },
        { "name": "left_ear", "x": 305, "y": 182, "c": 0.8 },
        { "name": "right_ear", "x": 335, "y": 182, "c": 0.82 },
        { "name": "left_shoulder", "x": 280, "y": 220, "c": 0.9 },
        { "name": "right_shoulder", "x": 360, "y": 220, "c": 0.9 },
        { "name": "left_elbow", "x": 250, "y": 280, "c": 0.85 },
        { "name": "right_elbow", "x": 390, "y": 280, "c": 0.85 },
        { "name": "left_wrist", "x": 230, "y": 340, "c": 0.8 },
        { "name": "right_wrist", "x": 410, "y": 340, "c": 0.8 },
        { "name": "left_hip", "x": 290, "y": 350, "c": 0.88 },
        { "name": "right_hip", "x": 350, "y": 350, "c": 0.88 },
        { "name": "left_knee", "x": 285, "y": 420, "c": 0.82 },
        { "name": "right_knee", "x": 355, "y": 420, "c": 0.82 },
        { "name": "left_ankle", "x": 278, "y": 500, "c": 0.75 },
        { "name": "right_ankle", "x": 362, "y": 500, "c": 0.75 }
      ]
    }
  ]
}
```

مثال مع `ts_unix_ms` كـ float (مقبول ولا يُرجع 422):

```json
"frame": { "ts_unix_ms": 1700000123456.0 }
```

### الاستجابة

- **202 Accepted** مع `{"status": "accepted"}` عند قبول الـ payload.
- **401** إذا `X-API-Key` غير موجود أو غير صحيح.
- **422** إذا كان الـ payload غير صالح (مثلاً `event_type` ليس `"frame_event"`، أو نقص في الحقول المطلوبة، أو عدد keypoints ليس 17، أو أسماء ليست مجموعة COCO-17).

---

## 2. التطبيع الداخلي (داخل السحابة)

بعد القبول:

- **ts_unix_ms**: يُحوَّل إلى `ts_ms = int(ts_unix_ms)`.
- **keypoints**: تُحوَّل من قائمة 17 كائناً `{ name, x, y, c }` إلى قائمة 17 صفاً `[[x, y, c], ...]` **مرتّبة حسب ترتيب COCO-17** (ترتيب الطلب يُتجاهل).

يتم التجميع لكل مفتاح `(device_id, camera_id, track_id)`. عند اكتمال **30 إطاراً** (أو `EDGE_WINDOW_SIZE` من الإعدادات) تُبنى **نافذة Cloud** بالشكل التالي.

---

## 3. صيغة النافذة بعد التجميع (داخل السحابة)

النافذة المكتملة تطابق مخطط الاستدعاء السحابي:

- **window**: `ts_start_ms`, `ts_end_ms`, `size` (مثلاً 30), `fps` (محسوبة).
- **people**: مصفوفة، كل عنصر:
  - `track_id`
  - `pose_conf`: متوسط `c` على **كل** الإطارات و**كل** الـ 17 نقطة في النافذة.
  - `keypoints`: `[T][17][3]` (T = عدد الإطارات، كل إطار 17 نقطة [x, y, c] بترتيب COCO-17).

هذه النافذة تُسجَّل (log) وتُخزَّن بياناتها الوصفية لـ `/debug/windows` ولا تُرسل تلقائياً إلى `/v1/activity/infer` في هذه المرحلة.

---

## 4. نقاط التصحيح (Debug)

- **GET /debug/buffers** (مع `X-API-Key`): يعرض قائمة الـ buffers الحالية: مفتاح `device_id|camera_id|track_id`، عدد الإطارات، آخر `ts_ms`.
- **GET /debug/windows?n=20** (مع `X-API-Key`): يعرض آخر N نوافذ مكتملة (بيانات وصفية فقط: device_id, camera_id, track_id, ts_start_ms, ts_end_ms, size, fps) بدون keypoints الكاملة.

---

## ملخص سريع

| ما يرسله Edge | ما تتوقعه السحابة في `/v1/edge/events` |
|----------------|----------------------------------------|
| `event_type` | `"frame_event"` فقط |
| `source` | `device_id` + `session_id` (مطلوبان)، `camera_id` اختياري |
| `frame.ts_unix_ms` | int أو float (يُحوَّل إلى int داخلياً) |
| `persons` | حقل مطلوب؛ القيمة يمكن أن تكون `[]` |
| `persons[].keypoints` | 17 كائناً `{ name, x, y, c }`؛ الأسماء = مجموعة COCO-17؛ الترتيب حر، السحابة ترتّب حسب COCO-17 |
| `x`, `y` | حالياً: pixel في الـ ingestion |
| حقول إضافية (bbox, score, …) | مسموحة (`extra="allow"`) |

جميع الاختلافات بين صيغة الحافة وصيغة التجميع تُحل في مكان واحد: **cloud ingestion** (التحقق + التطبيع + التجميع).
