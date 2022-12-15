# HTransformer1D
Hierarchical Transformer Based Human Genes AI Processor

---
This dataset cannot be published. I received it from University of Siena, Preventive Medicine Dpt., Italy.
Dataset name is called [GENCOVID](https://sites.google.com/dbm.unisi.it/gen-covid?pli=1)

Every patient has a _presence_/_Not Presence_ string of genes indicated using 1 for presence and 0 for absence.

Every patient in this dataset has a Clinical Grade Number representing the severity of the patient after being affected by COVID-19.<br>
The Clinical Grade Number is a number between -1 and 6, where -1 is _Resistent to Infection_ and 6 is _Deceased after Vaccination_.<br>
So we decided to put a threshold for labeling patients.<br>
If the Clinical grade is greater or equal to 2 patient is labeled as _*Severe*_ otherwise _*Not Severe*_.

---

# The Architecture

<p align="center">
<img src="https://user-images.githubusercontent.com/74437465/207886698-347d913c-fcf0-4411-8dd1-fe214e6bd3e3.svg" width="350" class="center">
</p>

So the whole architecture is similar to a _Visual Transformer_ for classification tasks. 

# Results
MCC Score based results listed here below.

1. Women Patients MCC Score = 0.41 _Strong Positive Relation_
2. Men Patients MCC Score = 0.43 _Strong Positive Relation_
