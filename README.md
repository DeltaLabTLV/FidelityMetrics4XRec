# Refining Fidelity Metrics for Explainable Recommendations

This repository provides the implementation of refined counterfactual metrics for assessing explanation fidelity in recommender systems.

## 🔍 Problem Statement
Existing AUC-based metrics suffer from three key limitations:
1. They do not ensure concise explanations, treating all user history elements equally.
2. They fail to differentiate between supporting and contradictory features.
3. They rely on fixed-percentage perturbations, leading to inconsistent evaluation across users.

To address these issues, we introduce **refined counterfactual metrics** that:
- Evaluate only the most relevant **Ke explaining features**.
- Exclude contradictory elements that suppress recommendations.
- Use fixed-length perturbations instead of fixed-percentage removals.

## 📌 Key Contributions
- **POS@Kr, Ke** – Measures the impact of removing explaining features on recommendation rank.
- **CNDCG@Ke** – Evaluates ranking degradation when key features are excluded.
- **INS@Ke** – Tracks confidence restoration when important features are added back.
- **DEL@Ke** – Assesses confidence reduction when top-explaining elements are removed.

