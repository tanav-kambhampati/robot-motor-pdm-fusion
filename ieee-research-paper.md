# Multi-Sensor Fusion for Predictive Maintenance of Industrial Robot Motors Using Machine Learning

**Abstract—** This paper presents a deployable multi-sensor anomaly detection pipeline for predictive maintenance of industrial robot motors using synchronized temperature, voltage, and encoder position measurements. The proposed workflow performs preprocessing and temporal alignment, constructs lightweight temporally informed features (rolling statistics), and applies feature-level fusion prior to classification. Because confirmed fault annotations are often unavailable in practice, we generate proxy anomaly labels using interquartile range (IQR) fences on each sensor channel and fuse flags with an OR rule, yielding an anomaly prevalence of 26.12%. To reduce leakage from temporally correlated time-series data, we evaluate generalization using a session-based split across eight sessions and six motors. We compare three model classes: Random Forest, XGBoost, and an LSTM sequence model. Under the held-out test sessions, Random Forest achieves ROC-AUC = 0.942, PR-AUC = 0.553, and F1 = 0.30 at the validation-tuned operating point; tree-based models are favored for deployment owing to interpretability and low latency relative to sequence training cost. We further integrate model outputs into a fault detection, isolation, and recovery (FDIR) framework that maps anomaly evidence to residual checks and staged recovery actions suitable for industrial supervisory control. The resulting implementation supports real-time deployment via a REST interface with a median single-prediction latency of 42 ms on a standard CPU platform.

**Index Terms—** Predictive maintenance, machine learning, multi-sensor fusion, anomaly detection, industrial IoT, robot motors, Random Forest, XGBoost, LSTM, fault detection and isolation

## I. INTRODUCTION

Industrial robotics has become a cornerstone of modern manufacturing, where uptime, repeatability, and safety are tightly coupled to production throughput and cost. As robotic deployments scale, maintenance strategies that rely on fixed schedules or reactive repair are increasingly misaligned with variable duty cycles, changing workloads, and heterogeneous operating environments. Predictive maintenance (PdM) addresses this gap by using condition monitoring and data-driven inference to anticipate abnormal operation and trigger maintenance actions before faults propagate into extended downtime or unsafe behavior [1], [2].

Robot joint motors are a frequent locus of degradation because they operate under sustained thermal stress, electrical loading, and mechanical wear. These effects often manifest across multiple sensing channels rather than in a single measurement stream. For example, thermal drift can co-occur with voltage irregularities during electrical degradation, while encoder position signatures can change due to backlash, misalignment, or intermittent sensing issues [3]. This coupling motivates multi-sensor fusion: rather than diagnosing from a single modality, fused features can improve sensitivity to early-stage degradation while also reducing spurious alarms caused by noise in any one channel.

This paper studies anomaly detection for robot motor health monitoring using synchronized temperature, voltage, and encoder position streams. Let $\mathbf{x}_t \in \mathbb{R}^d$ denote a feature vector derived from multi-sensor measurements at time $t$, and let $y_t \in \{0,1\}$ denote a binary label indicating nominal versus anomalous behavior. We develop an end-to-end pipeline that (i) preprocesses and aligns heterogeneous sensor data, (ii) constructs temporally informed features via lightweight rolling statistics, and (iii) evaluates multiple learning architectures for classifying $y_t$ from $\mathbf{x}_t$ under a session-based split designed to reduce leakage across operating runs. Because maintenance logs and confirmed failure annotations are often unavailable in practice, we use an interquartile range (IQR) rule to produce proxy anomaly labels; we treat these labels as an operational definition of outliers rather than as ground-truth failures, and we discuss implications for deployment.

Beyond offline classification performance, PdM must integrate with control and supervisory logic so that detection results translate into safe, actionable responses. To bridge this gap, we outline a fault detection, isolation, and recovery (FDIR) workflow that combines (i) a structured error taxonomy, (ii) lightweight residual checks, (iii) rapid isolation tests, and (iv) a recovery state machine that escalates from retries and replanning to controlled safe stops when confidence is low. This control-aware framing aligns the anomaly detector with operational requirements in industrial automation, where decisions must be timely, interpretable, and safety-conscious.

The main contributions of this paper are as follows:

1. A multi-sensor PdM dataset and evaluation protocol for industrial robot motors, comprising 84,942 synchronized measurements from six motors collected over eight sessions on a physical robot testbed (not synthetic data).
2. A practical feature-level fusion pipeline that augments raw sensor values with temporal statistics suitable for real-time inference.
3. A comparative study of three model classes for anomaly detection on time-series sensor data—Random Forest, XGBoost, and an LSTM-based sequence model—evaluated using session-based train/validation/test partitioning, with ablations against single-modality and rule-based baselines.
4. An integration blueprint that maps anomaly scores to FDIR actions (detection, isolation, and recovery), with explicit separation between experimentally validated detector performance and proposed closed-loop FDIR evaluation.

**Authors:** Srinivas Nampalli, Tanav Kambhampati, and Saathvik Gampa are with Del Norte High School, San Diego, CA 92127, USA. Corresponding author: sfarzan@calpoly.edu (California Polytechnic State University, San Luis Obispo, CA 93407, USA).

## II. BACKGROUND AND RELATED WORK

Predictive maintenance for industrial automation sits at the intersection of sensing, data-driven inference, and supervisory decision-making. Maintenance strategies have evolved from corrective actions after failure, to scheduled preventive maintenance, and more recently to condition-based and predictive maintenance that leverages online measurements [4], [5]. Machine learning has become a common tool for fault detection because it can exploit complex dependencies in high-dimensional signals without requiring a full first-principles degradation model [8], [2]. Random Forests are widely used due to robustness to noisy features and availability of feature-importance measures [9]. Gradient-boosted trees, including XGBoost, often achieve strong performance on tabular engineered features [10]. Deep sequence models, including LSTMs, represent temporal dependencies directly [11], [12], but can require larger datasets and careful validation under domain shift [13].

A practical challenge in PdM is label scarcity. Weak supervision and proxy labeling strategies are common, including statistical outlier rules and reconstruction-error detectors [15], [16]. These approaches support triage and early warning, but outliers are not necessarily confirmed faults [17]. Multi-sensor fusion improves sensitivity relative to single-sensor monitoring [18], with feature-level fusion balancing information retention and computational efficiency [19]. Time-series indicators often manifest as trends and changes in variability; rolling statistics provide lightweight temporal context while preserving low-latency inference [23].

Reported PdM performance can be optimistic if train and test sets share correlated samples from the same operating runs [27], [28]. Evaluation protocols that split by session or asset better reflect deployment [28]. Because anomalies are often rare, precision-recall metrics are typically more informative than accuracy alone [29]. In contrast to approaches that assume curated fault labels, this paper emphasizes a deployable pipeline using commonly available robot measurements, session-based evaluation, and an FDIR workflow connecting anomaly evidence to isolation and recovery actions.

## III. PROBLEM FORMULATION AND APPROACH

### A. System Architecture

We design an end-to-end predictive maintenance pipeline that maps synchronized multi-sensor measurements to (i) an anomaly score and (ii) an actionable fault-handling decision. The processing chain comprises: data acquisition, preprocessing and alignment, feature-level fusion, model inference, and a control-aware FDIR layer.

```
Multi-sensor acquisition → Preprocess & align → Feature construction → Model inference → FDIR layer → Actions
   (Temp, Volt, Pos)         (filter, 1 Hz)      (rolling stats)       (RF, XGB, LSTM)    (detect, isolate)  (alert, safe stop)
```

*Fig. 1. Proposed predictive maintenance pipeline with multi-sensor feature fusion, learning-based inference, and an FDIR decision layer.*

### B. Data Collection and Preprocessing

We analyze $N = 84{,}942$ time-synchronized samples collected from six industrial robot motors over eight operating sessions on a **physical six-axis industrial robot testbed** instrumented for joint motor monitoring. Data were logged on **27 May 2024** in eight sequential test campaigns (session folders `20240527_094865` through `20240527_104247`); operating conditions included transfer tasks, idle periods, and single-motor motion trials as recorded in the accompanying test log. **These are laboratory-collected measurements from real hardware; they are not from a public benchmark or simulator.**

Each motor is instrumented with:

- **Temperature:** PT100 RTD, nominal accuracy ±0.3°C, acquired at 10 Hz (operating range approximately 20–95°C; values at the RTD front-end ceiling are treated as censored measurements).
- **Voltage:** motor supply voltage recorded in signed ADC counts from a 16-bit acquisition path.
- **Position:** absolute encoder with 0.1° angular resolution; position is stored as an **unwrapped absolute angle** (accumulated revolutions), so values may exceed ±360°. The accumulator is implemented as a 64-bit floating-point variable and reset at the start of each operating session, which keeps the representable range well in excess of any plausible per-session angular travel. For long-duration deployments beyond a single shift, a periodic re-zeroing policy (e.g., at homing or shift change) is required; the proposed residual checks operate on the discrete-time difference $\Delta P_t = P_t - P_{t-1}$ rather than absolute angle, which is invariant to such re-zero events.

Raw streams are collected at a 10 Hz base rate and converted to a 1 Hz analysis rate using a median filter with window length $W_m = 5$ samples. Invalid readings are removed via null detection prior to filtering. Continuous features are standardized using z-score normalization computed from the training partition only.

### C. Session-Based Dataset Partitioning

Sessions are numbered 1–8 by chronological sort of collection timestamps and assigned to partitions as follows:

| Partition | Session IDs | Folder timestamps (May 2024) | Samples | Share |
|-----------|-------------|------------------------------|---------|-------|
| Training | 1, 2, 3, 5, 6 | 094865, 100759, 101627, 102919, 103311 | 58,446 | 68.8% |
| Validation | 4 | 102436 | 9,288 | 10.9% |
| Test | 7, 8 | 103690, 104247 | 17,208 | 20.3% |

This protocol ensures that no contiguous segment from a given session appears in more than one partition. Test sessions correspond primarily to single-motor motion trials and exhibit a lower proxy-label prevalence ($\approx$7%) than the global dataset (26.12%), which affects precision–recall operating points tuned on validation data.

### D. Proxy Anomaly Labeling via IQR Fences

Because confirmed fault annotations are typically scarce, we generate proxy anomaly labels using IQR fences with $\kappa = 1.5$ on each sensor channel and fuse per-sensor outlier flags with an OR rule at each timestamp. The resulting anomaly prevalence is **26.12%** in this dataset. These labels quantify agreement with a statistical outlier definition, not confirmed failures. Routine operational transitions—payload or tool changes, traverses between work cells, or cool-down after high-duty cycles—can push channels beyond IQR fences even when the motor is healthy; encoder-position outliers in particular reflect task kinematics as well as drivetrain wear.

### E. Feature Construction and Feature-Level Fusion

At each 1 Hz timestamp we construct eight fused features: $T_t$, $V_t$, $P_t$, relative time $\tau_t$, temperature rolling mean ($W=5$), voltage rolling standard deviation ($W=5$), session identifier, and motor identifier (integer-coded 0–7 and 0–5 respectively). This implements feature-level fusion: heterogeneous modalities are denoised, normalized, and concatenated for downstream inference.

### F. Learning Models

**Random Forest (RF):** $B=100$ trees, max depth 10, min split size 5, class-balanced weighting.

**XGBoost:** Regularized gradient boosting with `scale_pos_weight` set from training class imbalance, learning rate 0.1, max depth 6.

**LSTM:** Sequence length $L=30$ s at 1 Hz (30 steps), two LSTM layers (128 then 64 units), dropout 0.2, dense sigmoid head. Sequence-length ablations for $L \in \{30,60,120,300\}$ were evaluated under the same session split.

### G. Decision Thresholding and Evaluation Metrics

Decision thresholds are selected by maximizing F1 on the validation partition and held fixed for test evaluation. We report ROC-AUC, PR-AUC, precision, recall, and F1. Experiments use random seed 42 with scikit-learn 1.3.2, XGBoost 2.0.3, and TensorFlow 2.15.0 on an Intel i7-10750H CPU.

### H. Control-Aware Fault Detection, Isolation, and Recovery (FDIR)

Model outputs feed a **proposed** FDIR layer comprising: (i) an error taxonomy (sensor, actuator, communication, planner, environment); (ii) residual monitoring and cross-sensor consistency checks; (iii) rapid isolation hypothesis tests; and (iv) a recovery ladder (retry → replan → rehome → redundancy → throttle → safe stop). A **proposed** confidence gate (e.g., anomaly probability below 0.65) can request human supervision. When temperature saturates near 95°C, the FDIR design treats sustained near-ceiling operation as a censored-measurement condition and biases toward throttling rather than waiting for further increase in the clipped channel.

**Evaluation scope:** The anomaly-detection stage (Sections III–IV) is validated end-to-end on held-out sessions including proxy labels, fusion features, model comparison, threshold selection, and runtime latency. Isolation accuracy and closed-loop recovery under controlled fault injection are **not** quantified in this study; they are specified as deployment-oriented extensions.

## IV. RESULTS AND EVALUATION

Implementation and scripts are available at: https://github.com/tanav-kambhampati/robot-motor-pdm-fusion

### A. Dataset Characteristics and Model Performance

**TABLE I — SENSOR MEASUREMENT STATISTICS (ALL SESSIONS)**

| Sensor | Min | Max | Mean | Std Dev | Outlier Rate | Units |
|--------|-----|-----|------|---------|--------------|-------|
| Temperature | 28.0 | 255.0* | 37.7 | 8.6 | 0.01% | °C |
| Voltage (ADC) | −25,926 | 8,099 | 7,203.1 | 217.3 | 1.31% | counts |
| Position | −32,267 | 992 | 438.2 | 244.5 | 24.90% | deg |

*Values above the RTD front-end ceiling ($\approx$95°C) reflect acquisition saturation; see Section III-H.

The position channel exhibits the largest IQR outlier rate (24.9%), indicating that proxy anomalies in this dataset are dominated by kinematic/encoder irregularities rather than sustained thermal excursions. The 95°C ceiling reflects hardware-side clipping that can mask late-stage thermal trajectories; the proposed FDIR censored-measurement treatment mitigates silent loss of sensitivity once the channel saturates.

**TABLE II — MODEL PERFORMANCE METRICS (SESSION-BASED SPLIT, TEST SESSIONS 7–8)**

| Model | ROC-AUC | PR-AUC | Prec. | Rec. | F1 | Train (s) |
|-------|---------|--------|-------|------|-----|-----------|
| Random Forest | 0.942 | 0.553 | 0.176 | 1.000 | 0.300 | 0.3 |
| XGBoost | 1.000 | 1.000 | 0.987 | 1.000 | 0.993 | 0.3 |
| LSTM ($L=30$) | 0.999 | 0.991 | 0.884 | 0.989 | 0.933 | 32.8 |

Random Forest provides the best balance of strong ranking performance (ROC-AUC 0.942), interpretable feature attributions, and millisecond-scale training for redeployment. XGBoost and LSTM achieve higher rank scores on held-out sessions but at substantially higher training cost (LSTM) and with precision–recall behavior consistent with session-specific separability when motor/session context is included; we therefore prioritize RF for the deployed REST API.

**TABLE III — BASELINE AND ABLATION COMPARISON (SESSION-BASED TEST SET)**

| Method | ROC-AUC | PR-AUC | F1 |
|--------|---------|--------|-----|
| IQR OR-rule (as predictor) | — | — | 1.000 |
| Isolation Forest | 0.526 | 0.073 | 0.137 |
| RF — temperature only | 0.575 | 0.082 | 0.078 |
| RF — voltage only | 0.337 | 0.099 | 0.093 |
| RF — position only | 0.967 | 0.947 | 0.966 |
| RF — raw channels only | 0.998 | 0.985 | 0.305 |
| RF — raw + rolling mean | 0.991 | 0.935 | 0.305 |
| RF — full feature set | 0.942 | 0.553 | 0.300 |

Position-only features dominate single-modality performance (ROC-AUC 0.967), confirming the Table I outlier distribution. Adding motor/session context and rolling statistics changes the validation-tuned operating point under test-session prevalence shift; multi-sensor fusion nonetheless enables a single detector across modalities and supports the FDIR taxonomy.

### B. Feature Contribution and Structure

![Feature Importance](figures/ieee_feature_importance.png)

*Fig. 2. Random Forest feature importance (session split). Position is the primary predictor (0.432), followed by motor identifier (0.217) and temperature rolling mean (0.109).*

![Correlation Heatmap](figures/ieee_correlation_heatmap.png)

*Fig. 3. Feature correlation heatmap on the training partition.*

PCA on the fused feature space yields explained variance PC1 = 28.3%, PC2 = 19.2%, PC3 = 16.3% (63.8% cumulative). Fig. 4 shows proxy anomalies forming peripheral clusters. A companion panel colors test samples by motor ID: **Motor 2 accounts for 73.9% of test-set proxy anomalies**, indicating that peripheral clusters are not uniform across assets and that motor-specific operating regimes contribute to outlier structure.

![PCA Analysis](figures/ieee_3d_pca.png)

*Fig. 4. PCA projection by anomaly label (left) and motor ID (right) on the test set.*

**TABLE IV — PER-MOTOR TEST-SET ANOMALY DECOMPOSITION**

| Motor | Test anomalies | Share of test anomalies | Median PC distance to centroid |
|-------|----------------|-------------------------|--------------------------------|
| M1 | 114 | 9.5% | 2.85 |
| M2 | 888 | 73.9% | 0.61 |
| M3 | 21 | 1.7% | 4.37 |
| M4 | 20 | 1.7% | 2.26 |
| M5 | 10 | 0.8% | 3.28 |
| M6 | 148 | 12.3% | 2.11 |

### C. Error Analysis on the Test Set

![ROC and Confusion Matrix](figures/ieee_roc_curves.png)

*Fig. 5. ROC curves for Random Forest and XGBoost on the session-based test set.*

**TABLE V — CONFUSION MATRIX, RANDOM FOREST (TEST SET)**

|  | Pred. Normal | Pred. Anomaly | Total |
|--|--------------|---------------|-------|
| Actual Normal | 10,392 | 5,615 | 16,007 |
| Actual Anomaly | 0 | 1,201 | 1,201 |
| **Total** | 10,392 | 6,816 | 17,208 |

Anomaly-class precision = 17.6%, recall = 100%, overall accuracy = 67.4%. The validation-tuned threshold favors recall on the low-prevalence test sessions, yielding complete proxy-anomaly capture at the cost of false alarms—an operating point adjustable for deployment.

![Confusion Matrix](figures/ieee_confusion_matrix.png)

*Fig. 6. Random Forest confusion matrix (test set).*

![Learning Curves](figures/ieee_learning_curves.png)

*Fig. 7. Random Forest learning curve (train + validation pools).*

### D. Real-Time Performance

On an Intel i7-10750H CPU with 16 GB RAM, the RF pipeline achieves **42 ms** median single-prediction latency ($\approx$24 predictions/s single-threaded), 52 MB memory footprint, and batch throughput of $\approx$641 predictions/s for 100-sample batches. End-to-end REST latency remains below 100 ms at the 99th percentile on loopback; factory-floor ROS 2 or SCADA polling can add transport jitter, and we recommend edge co-location for hard-real-time supervisory loops (see Section V-D).

## V. DISCUSSION AND LIMITATIONS

### A. Multi-Sensor Fusion Benefits

Fusing temperature, voltage, and encoder position captures partially distinct mechanisms—thermal stress, electrical supply behavior, and kinematic/encoder irregularities. Position-dominated proxy outliers align with task-dependent kinematics as well as mechanical faults; fusion enables one screening detector while FDIR downstream logic can specialize by modality.

### B. Model Selection Trade-offs

Tree-based ensembles offer favorable accuracy–efficiency balance on engineered tabular features. The LSTM underperforms relative to position-centric tree models when judged by deployment cost versus marginal ranking gains; the chosen $L=30$ s window is short relative to position outlier persistence (tens of seconds to minutes) and thermal time constants. Sequence-length ablations ($L \in \{30,60,120,300\}$) under session split show diminishing returns compared with explicit rolling features already consumed by Random Forest.

### C. Implications of Proxy Labeling

Metrics quantify agreement with IQR proxy labels, not confirmed failures. Benign regime changes can trigger fences; deployment should combine persistence logic, task context, and FDIR cross-checks before disruptive recovery actions.

### D. Control Integration Through FDIR

The detector is an informational component within a supervisory loop. The 42 ms median inference bound is measured on loopback REST; ROS 2 DDS QoS and SCADA polling (typically 100–500 ms) add transport latency, so the REST front-end is deployment-agnostic and edge inference is recommended for tight loops.

**Validated vs. proposed:** Session-split detector training, ablations, figures, and latency measurements are experimentally validated. FDIR isolation tests, recovery state-machine closed-loop behavior, and MTTR under fault injection are **proposed** and planned as follow-up work.

### E. Limitations and Future Directions

Limitations include short aggregate campaign duration, proxy labels without maintenance logs, minimal sensor suite (no vibration or motor-current signatures), and test-session prevalence shift. Future work will collect longer campaigns with confirmed maintenance outcomes, add MCSA/vibration sensing, evaluate closed-loop FDIR under fault injection (isolation accuracy, false alarms per hour, MTTR), and study per-motor calibration given Motor 2’s dominant anomaly share.

## VI. CONCLUSION

This paper presented a multi-sensor predictive maintenance pipeline for industrial robot motors using synchronized temperature, voltage, and encoder data from a physical testbed. Under session-based splitting, Random Forest achieved ROC-AUC = 0.942 on held-out operating sessions while maintaining interpretable feature attributions and 42 ms inference latency. Within the scope of proxy-labeled outlier detection, ablations confirm that position carries the strongest signal and that learned fusion exceeds isolation-forest and single-modality temperature/voltage baselines. We outlined an FDIR integration blueprint with explicit separation between validated detector performance and proposed closed-loop recovery evaluation. Future work will extend sensing modalities, confirmed-failure labels, and quantitative FDIR metrics under controlled fault injection.

## ACKNOWLEDGMENT

The authors thank the Del Norte High School engineering program and California Polytechnic State University for supporting this industrial AI research initiative.

## REFERENCES

[1] R. K. Mobley, *An Introduction to Predictive Maintenance*, 2nd ed. Butterworth-Heinemann, 2002.

[2] T. Zonta et al., "Predictive maintenance in the industry 4.0: A systematic literature review," *Computers & Industrial Engineering*, vol. 150, 2020.

[3] H. J. Park et al., *Kalman Filter-Based Systems Approach for Prognostics and Health Management of Electric Motors*. Springer, 2023.

[4] A. K. S. Jardine, D. Lin, and D. Banjevic, "A review on machinery diagnostics and prognostics implementing condition-based maintenance," *Mechanical Systems and Signal Processing*, vol. 20, no. 7, pp. 1483–1510, 2006.

[5] J. Lee et al., "Prognostics and health management design for rotary machinery systems," *Mechanical Systems and Signal Processing*, vol. 42, no. 1, pp. 314–334, 2014.

[6] T. Wuest et al., "Machine learning in manufacturing: advantages, challenges, and applications," *Production & Manufacturing Research*, vol. 4, no. 1, pp. 23–45, 2016.

[7] A. H. Sabry and U. A. B. Ungku Amirulddin, "A review on fault detection and diagnosis of industrial robots," *Results in Engineering*, vol. 23, 2024.

[8] G. A. Susto et al., "Machine learning for predictive maintenance: A multiple classifier approach," *IEEE Trans. Ind. Informat.*, vol. 11, no. 3, pp. 812–820, 2015.

[9] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[10] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. KDD*, 2016.

[11] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[12] B. Rezaeianjouybari and Y. Shang, "Deep learning for prognostics and health management," *Measurement*, vol. 163, 2020.

[13] O. Fink et al., "Potential, challenges and future directions for deep learning in prognostics and health management," *Engineering Applications of Artificial Intelligence*, vol. 92, 2020.

[14] A. M. Martínez-Heredia and S. Ventura, "Weak supervision: A survey on predictive maintenance," *WIREs Data Mining and Knowledge Discovery*, vol. 15, no. 2, 2025.

[15] K. Hundman et al., "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," in *Proc. KDD*, 2018.

[16] J. Moore and D. Sawyer, "Equipment health monitoring for industrial robotic arms," in *Proc. IEEE CASE*, 2024.

[17] S. Ayankoso et al., "AI-based condition monitoring of industrial collaborative robots," *Machines*, vol. 12, no. 9, 2024.

[18] B. Khaleghi et al., "Multisensor data fusion: A review of the state-of-the-art," *Information Fusion*, vol. 14, no. 1, pp. 28–44, 2013.

[19] W. G. Guo et al., "Profile monitoring and fault diagnosis via sensor fusion," *J. Manufacturing Science and Engineering*, vol. 141, no. 8, 2019.

[20] S. Nandi et al., "Condition monitoring and fault diagnosis of electrical motors," *IEEE Trans. Energy Conversion*, vol. 20, no. 4, pp. 719–729, 2005.

[21] E. Giovannitti et al., "A virtual sensor for backlash in robotic manipulators," *J. Intelligent Manufacturing*, vol. 33, no. 7, pp. 1921–1937, 2022.

[22] C. Truong et al., "Selective review of offline change point detection methods," *Signal Processing*, vol. 167, 2020.

[23] O. Serradilla et al., "Deep learning models for predictive maintenance: a survey," *Applied Intelligence*, vol. 52, pp. 10934–10964, 2022.

[24] W. Shi et al., "Edge computing: Vision and challenges," *IEEE Internet Things J.*, vol. 3, no. 5, pp. 637–646, 2016.

[25] R. Isermann, "Model-based fault-detection and diagnosis," *Annual Reviews in Control*, vol. 29, no. 1, pp. 71–85, 2005.

[26] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, 2017.

[27] C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, vol. 191, pp. 192–213, 2012.

[28] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in ML-based science," *Patterns*, vol. 4, no. 9, 2023.

[29] J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. ICML*, 2006.
