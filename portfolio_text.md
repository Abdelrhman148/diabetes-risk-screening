# CV bullet (all numbers come from the notebook)
Built and deployed a diabetes-risk screening app (Streamlit) from 768 patient records; compared 3 tuned models with cross-validation and tuned the decision threshold to catch 81.5% of diabetic cases on a held-out test set (ROC-AUC 0.82); Python, scikit-learn, pandas, Streamlit.

# LinkedIn post draft
Clinics can't lab-test everyone for diabetes, so who should they test first?

In my NextGen Institute ML internship, I built an app that flags people who should get a confirmatory diabetes test, using 8 basic clinic measurements.

• Compared 3 tuned models on 768 real patient records
• Our best model catches 81.5% of diabetic cases on unseen data
• The biggest lesson: accuracy was misleading. A model that says "nobody has diabetes" is 65% accurate and catches nobody, so I tuned for recall.

Also learned that ~49% of the Insulin column was actually missing data hidden as zeros.

Try the live app: [link] · Code: [GitHub link]
Dataset: Pima Indians Diabetes Database (NIDDK)
#MachineLearning #DataScience #Healthcare
(Tag NextGen Institute; add a 30-60 s demo video.)
