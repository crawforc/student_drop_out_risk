# Model Card: Student Dropout Risk Prediction Model

## Model Details
- **Developer:** Cindy Crawford, WGU BS Data Analytics Capstone
- **Date:** November 2026
- **Model type:** Random Forest classifier (scikit-learn)
- **Settings:** 100 trees, maximum depth of 10, balanced class weights, random_state = 42
- **Output:** A yes/no prediction (at risk / not at risk) and a risk score from 0 to 1

## Intended Use
- **Purpose:** Help schools identify students who may be at risk of dropping out, so staff can offer support early.
- **Intended users:** Academic advisors and student support teams.
- **How it should be used:** As an early-warning signal that prompts an in person check in from a person. A person should always review the result.
- **Not intended for:** Admissions decisions, discipline, financial aid decisions, or any decision that penalizes a student.

## Data
- **Source:** "Student Academic Performance and Dropout Risk" by Mohan Krishna Thalla, published on Kaggle under a CC0 Public Domain license.
- **Size:** 100,000 students, 34 columns. No missing values and no duplicate rows.
- **Synthetic data:** The dataset is synthetic, not real student records.
- **Target:** `dropout` (1 = dropped out, 0 = stayed). 32% of students dropped out.
- **Split:** 80% training (80,000 students) and 20% testing (20,000 students), stratified so both groups keep the 32% dropout rate.

## Features Used
- **Used:** 30 academic, financial, wellbeing, and background features (for example GPA, units failed, attendance, tuition status, work hours, stress level).
- **Excluded on purpose:** `gender` and `nationality`. These showed no relationship with dropout (dropout rates were 30-33% for every group) and could introduce bias. They were kept only for fairness testing.
- **Removed:** `person_id`, because it is only a row number.

## Metrics
Results on the 20,000 test students:

| Metric | Score |
|---|---|
| Accuracy | 80.8% |
| Precision | 66.5% |
| Recall | 80.9% |
| F1 | 73.0% |

**Why recall matters most:** A missed at-risk student gets no support, while a false alarm only costs a check-in. The first model reached 83.0% accuracy but only 64.3% recall. Balanced class weights raised recall to 80.9%, with fewer accuracy and precision points as the trade-off.

**Most important factors:** Semester 1 GPA and semester 2 GPA were by far the strongest predictors, followed by units failed, missed deadlines, attendance, and library visits.

## Fairness Testing
Performance was checked for each group of gender, nationality, family income level, first-generation status, disability status, and international student status (see `fairness_report.csv`).

- **Gender:** Recall was even across groups (Female 81.1%, Male 80.8%, Other 78.1%).
- **Nationality:** Recall ranged from 76.3% to 88.3% across the 15 countries. Some groups are small (under 600 test students), so their results are less reliable.
- **Family income:** Recall was 84.8% for low income, 79.3% for middle income, and 73.2% for high income. High-income students also had the lowest precision (56.6%), meaning more false alarms in that group.
- **First-generation, disability, and international status:** Recall stayed within about 5 points between groups.

## Ethical Considerations
- **Bias:** Gender and nationality were excluded from the features, and every group was tested. The income gap should be monitored.
- **Labeling risk:** Being flagged "at risk" could lead to unfair treatment or lowered expectations. Results should be used only to offer support, never to limit opportunities.
- **Privacy:** Real student data is protected by FERPA. A school retraining this model must store data securely, limit access, and remove identifying information.
- **Human review:** The model should support, not replace, staff judgment.

## Caveats and Limitations
- The model was trained on synthetic data, so its accuracy may not carry over to real students. A school must retrain it on its own data before using it.
- Correlation is not causation. Factors linked to dropout do not necessarily cause it.
- About 1 in 5 at-risk students is still missed, and about 1 in 3 flagged students will not actually drop out.
- The model reflects one point in time. It should be retrained regularly as student populations change.