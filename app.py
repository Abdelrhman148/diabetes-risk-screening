import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils import FEATURES, zeros_to_nan  # noqa: F401  (needed to unpickle the pipeline)

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="centered")

BASE = Path(__file__).parent

TXT = {
    "en": {
        "lang": "Language / اللغة",
        "title": "🩺 Diabetes Risk Screening",
        "caption": "Educational demo built for the NextGen Institute ML final project. **Not a medical device** and not a diagnosis.",
        "intro": "Enter the clinic measurements below. If a value was not measured, tick **Not measured** and the model will handle it.",
        "preg": "Pregnancies", "glucose": "Glucose (mg/dL, 2h OGTT)", "bp": "Diastolic blood pressure (mm Hg)",
        "bp_na": "Blood pressure not measured", "age": "Age (years)", "skin": "Skin-fold thickness (mm)",
        "skin_na": "Skin-fold not measured", "insulin": "2-hour insulin (mu U/ml)", "insulin_na": "Insulin not measured",
        "bmi": "BMI (kg/m²)", "dpf": "Diabetes pedigree function",
        "dpf_help": "Family-history score; higher means more relatives with diabetes.",
        "submit": "Check risk", "err": "Could not score this input: ", "score": "Estimated risk score",
        "high": "**Refer for a confirmatory diabetes test.** The score is above the screening threshold ({t:.0%}).",
        "low": "**Lower risk.** The score is below the screening threshold ({t:.0%}). This is not a guarantee: seek care if you have symptoms.",
        "about": "About this model",
        "about_txt": "**Model:** {m} · **Screening threshold:** {t:.2f} (chosen on training data so that at least 80% of diabetic cases are caught).",
        "col": "Test-set result", "recall": "Recall (diabetic cases caught)", "prec": "Precision", "acc": "Accuracy",
        "limits": "**Limitations:** trained on only 768 women of Pima Indian heritage aged 21+; may not generalize to other groups. Data: Pima Indians Diabetes Database (NIDDK).",
    },
    "ar": {
        "lang": "Language / اللغة",
        "title": "🩺 فحص خطر الإصابة بالسكري",
        "caption": "نموذج تعليمي لمشروع NextGen Institute النهائي. **ليس جهازاً طبياً وليس تشخيصاً.**",
        "intro": "أدخل القياسات من العيادة. لو قياس مش متاح، علّم على **غير مقاس** والنموذج هيتعامل معاه.",
        "preg": "عدد مرات الحمل", "glucose": "الجلوكوز (mg/dL بعد اختبار تحمل الجلوكوز)",
        "bp": "ضغط الدم الانبساطي (mm Hg)", "bp_na": "ضغط الدم غير مقاس", "age": "العمر (سنة)",
        "skin": "سُمك ثنية الجلد (mm)", "skin_na": "سُمك الجلد غير مقاس",
        "insulin": "الإنسولين بعد ساعتين (mu U/ml)", "insulin_na": "الإنسولين غير مقاس",
        "bmi": "مؤشر كتلة الجسم BMI (kg/m²)", "dpf": "مؤشر التاريخ العائلي للسكري",
        "dpf_help": "كل ما زاد، زاد عدد الأقارب المصابين بالسكري.",
        "submit": "احسب الخطر", "err": "مقدرتش أحسب النتيجة للمدخلات دي: ", "score": "درجة الخطر المقدّرة",
        "high": "**يُنصح بعمل تحليل تأكيدي للسكري.** الدرجة أعلى من حد الفحص ({t:.0%}).",
        "low": "**خطر أقل.** الدرجة أقل من حد الفحص ({t:.0%}). ده مش ضمان، ولو عندك أعراض روح للدكتور.",
        "about": "عن النموذج",
        "about_txt": "**النموذج:** {m} · **حد الفحص:** {t:.2f} (اتحدد من بيانات التدريب عشان يكشف 80% على الأقل من الحالات).",
        "col": "نتيجة بيانات الاختبار", "recall": "الحساسية (الحالات المكتشفة)", "prec": "الدقة الإيجابية (Precision)",
        "acc": "الدقة الكلية",
        "limits": "**حدود النموذج:** اتدرّب على 768 سيدة فقط من أصل بيما الهندي، أعمارهن 21+، فممكن ميكونش مناسب لفئات تانية. البيانات: Pima Indians Diabetes Database (NIDDK).",
    },
}

choice = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True, label_visibility="collapsed")
L = "ar" if choice == "العربية" else "en"
t = TXT[L]
if L == "ar":
    st.markdown("""<style>
    .stApp, .stMarkdown, label, p, h1, h2, h3, [data-testid="stMetric"] { direction: rtl; text-align: right; }
    </style>""", unsafe_allow_html=True)


@st.cache_resource
def load():
    bundle = joblib.load(BASE / "models" / "diabetes_pipeline.joblib")
    metrics = json.load(open(BASE / "models" / "metrics.json"))
    return bundle, metrics


bundle, metrics = load()
pipe, threshold = bundle["pipeline"], bundle["threshold"]

st.title(t["title"])
st.caption(t["caption"])
st.write(t["intro"])

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input(t["preg"], 0, 20, 2)
        glucose = st.number_input(t["glucose"], 40, 300, 120)
        bp = st.number_input(t["bp"], 30, 130, 70)
        bp_na = st.checkbox(t["bp_na"])
        age = st.number_input(t["age"], 18, 100, 35)
    with c2:
        skin = st.number_input(t["skin"], 5, 100, 25)
        skin_na = st.checkbox(t["skin_na"])
        insulin = st.number_input(t["insulin"], 5, 900, 100)
        insulin_na = st.checkbox(t["insulin_na"])
        bmi = st.number_input(t["bmi"], 10.0, 70.0, 28.0, step=0.1)
        dpf = st.number_input(t["dpf"], 0.05, 2.5, 0.4, step=0.01, help=t["dpf_help"])
    submitted = st.form_submit_button(t["submit"], type="primary")

if submitted:
    row = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": 0 if bp_na else bp, "SkinThickness": 0 if skin_na else skin,
        "Insulin": 0 if insulin_na else insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age,
    }])[FEATURES]
    try:
        proba = float(pipe.predict_proba(row)[0, 1])
    except Exception as e:  # keep the app alive on unexpected input
        st.error(t["err"] + str(e))
        st.stop()

    st.metric(t["score"], f"{proba:.0%}")
    st.progress(min(max(proba, 0.0), 1.0))
    if proba >= threshold:
        st.warning(t["high"].format(t=threshold))
    else:
        st.success(t["low"].format(t=threshold))

with st.expander(t["about"]):
    tm = metrics["test_metrics"]
    st.write(t["about_txt"].format(m=metrics["model"], t=threshold))
    st.table(pd.DataFrame({t["col"]: {
        t["recall"]: f"{tm['recall']:.1%}", t["prec"]: f"{tm['precision']:.1%}",
        "ROC-AUC": f"{tm['roc_auc']:.3f}", t["acc"]: f"{tm['accuracy']:.1%}"}}))
    st.write(t["limits"])
