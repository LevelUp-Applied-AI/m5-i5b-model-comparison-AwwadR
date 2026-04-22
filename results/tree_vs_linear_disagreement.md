# Tree vs. Linear Disagreement Analysis

## Sample Details

- **Test-set index:** 4060
- **True label:** 0
- **RF predicted P(churn=1):** 0.5998
- **LR predicted P(churn=1):** 0.1700
- **Probability difference:** 0.4299

## Feature Values

- **tenure:** 36.0
- **monthly_charges:** 20.0
- **total_charges:** 1077.33
- **num_support_calls:** 2.0
- **senior_citizen:** 0.0
- **has_partner:** 0.0
- **has_dependents:** 0.0
- **contract_months:** 1.0

## Structural Explanation

The two models disagreed strongly on this sample: the random forest predicted **0.5998** for churn, while logistic regression predicted only **0.1700**, a gap of **0.4299**. A likely reason is that the random forest could treat **`contract_months = 1`** as a strong threshold signal for month-to-month customers and combine it with other risk-related values such as **no partner (0)** and **no dependents (0)**. At the same time, logistic regression combines all features in one linear way, so values like **tenure = 36**, **monthly_charges = 20.0**, and **total_charges = 1077.33** may have pulled the prediction downward and led to a much lower churn score.

This example shows the kind of pattern a tree model can capture more easily: a specific interaction or threshold effect, where a customer can still look risky because of their contract type and household-related features even if some other values look less risky on their own. The true label for this case was **0**, so the random forest was more aggressive here, while logistic regression was more conservative.
