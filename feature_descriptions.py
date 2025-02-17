htn_variable_dict = {
    "age": "(int) age of the patient",
    "Male": "(bool) sex of the patient (1 = male, 0 = female)",
    "BLACK": "(bool) ethnicity indicator (1 = black, 0 = non-black)",
    "OTHER": "(bool) ethnicity indicator (1 = asian, mixed, etc., 0 = black or white)",
    "WHITE": "(bool) ethnicity indicator (1 = white, 0 = non-white)",
    "weight_min": "(float) minimum recorded weight of the patient",
    "weight_max": "(float) maximum recorded weight of the patient",
    "weight_median": "(float) median weight of the patient",
    "weight_sd": "(float) standard deviation of recorded weights",
    "weight_skewness": "(float) skewness of recorded weights",
    "bmi_min": "(float) minimum body mass index (BMI) recorded",
    "bmi_max": "(float) maximum BMI recorded",
    "bmi_sd": "(float) standard deviation of BMI values",
    "bmi_skewness": "(float) skewness of BMI values",
    "bp_n": "(int) total number of blood pressure (BP) measurements",
    "min_systolic": "(float) minimum systolic blood pressure (SBP) measured",
    "min_diastolic": "(float) minimum diastolic blood pressure (DBP) measured",
    "max_systolic": "(float) maximum systolic blood pressure (SBP) measured",
    "max_diastolic": "(float) maximum diastolic blood pressure (DBP) measured",
    "mean_systolic": "(float) mean of systolic blood pressure (SBP) measured",
    "mean_diastolic": "(float) mean of diastolic blood pressure (DBP) measured",
    "median_systolic": "(float) median of systolic blood pressure (SBP) measured",
    "median_diastolic": "(float) median of diastolic blood pressure (DBP) measured",
    "sd_systolic": "(float) standard deviation of systolic blood pressure (SBP) measurements",
    "sd_diastolic": "(float) standard deviation of diastolic blood pressure (DBP) measurements",
    "skew_systolic": "(float) skewness of systolic blood pressure (SBP) measurements",
    "skew_diastolic": "(float) skewness of diastolic blood pressure (DBP) measurements",
    "high_bp_n": "(int) number of high blood pressure measurements (SBP >= 140 or DBP >= 90)",
    "mean_high_bp_systolic": "(float) mean of systolic BP for high blood pressure measurements",
    "mean_high_bp_diastolic": "(float) mean of diastolic BP for high blood pressure measurements",
    "median_high_bp_systolic": "(float) median of systolic BP for high blood pressure measurements",
    "median_high_bp_diastolic": "(float) median of diastolic BP for high blood pressure measurements",
    "sd_high_bp_systolic": "(float) standard deviation of systolic BP for high blood pressure measurements",
    "sd_high_bp_diastolic": "(float) standard deviation of diastolic BP for high blood pressure measurements",
    "skew_high_bp_systolic": "(float) skewness of systolic BP for high blood pressure measurements",
    "skew_high_bp_diastolic": "(float) skewness of diastolic BP for high blood pressure measurements",
    "median_high_bp_n_yr": "(float) median number of high BP measurements per year",
    "sd_high_bp_n_yr": "(float) standard deviation of high BP measurements per year",
    "skew_high_bp_n_yr": "(float) skewness of high BP measurements per year",
    "max.Pct.BASOPHILS": "(float) maximum percentage of basophils in lab results",
    "min.Pct.BASOPHILS": "(float) minimum percentage of basophils in lab results",
    "median.Pct.BASOPHILS": "(float) median percentage of basophils in lab results",
    "q1.Pct.BASOPHILS": "(float) first quartile percentage of basophils in lab results",
    "q3.Pct.BASOPHILS": "(float) third quartile percentage of basophils in lab results",
    "max.Pct.EOSINOPHILS": "(float) maximum percentage of eosinophils in lab results",
    "min.Pct.EOSINOPHILS": "(float) minimum percentage of eosinophils in lab results",
    "median.Pct.EOSINOPHILS": "(float) median percentage of eosinophils in lab results",
    "q1.Pct.EOSINOPHILS": "(float) first quartile percentage of eosinophils in lab results",
    "q3.Pct.EOSINOPHILS": "(float) third quartile percentage of eosinophils in lab results",
    "max.Pct.LYMPHOCYTES": "(float) maximum percentage of lymphocytes in lab results",
    "min.Pct.LYMPHOCYTES": "(float) minimum percentage of lymphocytes in lab results",
    "median.Pct.LYMPHOCYTES": "(float) median percentage of lymphocytes in lab results",
    "q1.Pct.LYMPHOCYTES": "(float) first quartile percentage of lymphocytes in lab results",
    "q3.Pct.LYMPHOCYTES": "(float) third quartile percentage of lymphocytes in lab results",
    "max.Pct.MONOCYTES": "(float) maximum percentage of monocytes in lab results",
    "min.Pct.MONOCYTES": "(float) minimum percentage of monocytes in lab results",
    "median.Pct.MONOCYTES": "(float) median percentage of monocytes in lab results",
    "q1.Pct.MONOCYTES": "(float) first quartile percentage of monocytes in lab results",
    "q3.Pct.MONOCYTES": "(float) third quartile percentage of monocytes in lab results",
    "max.Pct.NEUTROPHILS": "(float) maximum percentage of neutrophils in lab results",
    "min.Pct.NEUTROPHILS": "(float) minimum percentage of neutrophils in lab results",
    "median.Pct.NEUTROPHILS": "(float) median percentage of neutrophils in lab results",
    "q1.Pct.NEUTROPHILS": "(float) first quartile percentage of neutrophils in lab results",
    "q3.Pct.NEUTROPHILS": "(float) third quartile percentage of neutrophils in lab results",
    "max.ALBUMIN": "(float) maximum albumin level in lab results",
    "min.ALBUMIN": "(float) minimum albumin level in lab results",
    "median.ALBUMIN": "(float) median albumin level in lab results",
    "q1.ALBUMIN": "(float) first quartile albumin level in lab results",
    "q3.ALBUMIN": "(float) third quartile albumin level in lab results",
    "max.ALKALINE.PHOSPHATASE": "(float) maximum alkaline phosphatase level in lab results",
    "min.ALKALINE.PHOSPHATASE": "(float) minimum alkaline phosphatase level in lab results",
    "median.ALKALINE.PHOSPHATASE": "(float) median alkaline phosphatase level in lab results",
    "q1.ALKALINE.PHOSPHATASE": "(float) first quartile alkaline phosphatase level in lab results",
    "q3.ALKALINE.PHOSPHATASE": "(float) third quartile alkaline phosphatase level in lab results",
    "max.ALT": "(float) maximum ALT (alanine transaminase) level in lab results",
    "min.ALT": "(float) minimum ALT level in lab results",
    "median.ALT": "(float) median ALT level in lab results",
    "q1.ALT": "(float) first quartile ALT level in lab results",
    "q3.ALT": "(float) third quartile ALT level in lab results",
    "max.AST": "(float) maximum AST (aspartate transaminase) level in lab results",
    "min.AST": "(float) minimum AST level in lab results",
    "median.AST": "(float) median AST level in lab results",
    "q1.AST": "(float) first quartile AST level in lab results",
    "q3.AST": "(float) third quartile AST level in lab results",
    "max.BILIRUBIN.TOTAL": "(float) maximum total bilirubin level in lab results",
    "min.BILIRUBIN.TOTAL": "(float) minimum total bilirubin level in lab results",
    "median.BILIRUBIN.TOTAL": "(float) median total bilirubin level in lab results",
    "q1.BILIRUBIN.TOTAL": "(float) first quartile total bilirubin level in lab results",
    "q3.BILIRUBIN.TOTAL": "(float) third quartile total bilirubin level in lab results",
    "max.CALCIUM": "(float) maximum calcium level in lab results",
    "min.CALCIUM": "(float) minimum calcium level in lab results",
    "median.CALCIUM": "(float) median calcium level in lab results",
    "q1.CALCIUM": "(float) first quartile calcium level in lab results",
    "q3.CALCIUM": "(float) third quartile calcium level in lab results",
    "max.CARBON.DIOXIDE": "(float) maximum carbon dioxide level in lab results",
    "min.CARBON.DIOXIDE": "(float) minimum carbon dioxide level in lab results",
    "median.CARBON.DIOXIDE": "(float) median carbon dioxide level in lab results",
    "q1.CARBON.DIOXIDE": "(float) first quartile carbon dioxide level in lab results",
    "q3.CARBON.DIOXIDE": "(float) third quartile carbon dioxide level in lab results",
    "max.CHLORIDE": "(float) maximum chloride level in lab results",
    "min.CHLORIDE": "(float) minimum chloride level in lab results",
    "median.CHLORIDE": "(float) median chloride level in lab results",
    "q1.CHLORIDE": "(float) first quartile chloride level in lab results",
    "q3.CHLORIDE": "(float) third quartile chloride level in lab results",
    "max.CHOLESTEROL": "(float) maximum cholesterol level in lab results",
    "min.CHOLESTEROL": "(float) minimum cholesterol level in lab results",
    "median.CHOLESTEROL": "(float) median cholesterol level in lab results",
    "q1.CHOLESTEROL": "(float) first quartile cholesterol level in lab results",
    "q3.CHOLESTEROL": "(float) third quartile cholesterol level in lab results",
    "max.CHOLESTEROL.CALCULATED.LOW.DENSITY.LIPOPROTEIN": "(float) maximum LDL cholesterol level calculated",
    "min.CHOLESTEROL.CALCULATED.LOW.DENSITY.LIPOPROTEIN": "(float) minimum LDL cholesterol level calculated",
    "median.CHOLESTEROL.CALCULATED.LOW.DENSITY.LIPOPROTEIN": "(float) median LDL cholesterol level calculated",
    "q1.CHOLESTEROL.CALCULATED.LOW.DENSITY.LIPOPROTEIN": "(float) first quartile LDL cholesterol level calculated",
    "q3.CHOLESTEROL.CALCULATED.LOW.DENSITY.LIPOPROTEIN": "(float) third quartile LDL cholesterol level calculated",
    "max.CHOLESTEROL.HIGH.DENSITY.LIPOPROTEIN": "(float) maximum HDL cholesterol level calculated",
    "min.CHOLESTEROL.HIGH.DENSITY.LIPOPROTEIN": "(float) minimum HDL cholesterol level calculated",
    "median.CHOLESTEROL.HIGH.DENSITY.LIPOPROTEIN": "(float) median HDL cholesterol level calculated",
    "q1.CHOLESTEROL.HIGH.DENSITY.LIPOPROTEIN": "(float) first quartile HDL cholesterol level calculated",
    "q3.CHOLESTEROL.HIGH.DENSITY.LIPOPROTEIN": "(float) third quartile HDL cholesterol level calculated",
    "max.CREATININE": "(float) maximum creatinine level in lab results",
    "min.CREATININE": "(float) minimum creatinine level in lab results",
    "median.CREATININE": "(float) median creatinine level in lab results",
    "q1.CREATININE": "(float) first quartile creatinine level in lab results",
    "q3.CREATININE": "(float) third quartile creatinine level in lab results",
    "max.HEMATOCRIT": "(float) maximum hematocrit level in lab results",
    "min.HEMATOCRIT": "(float) minimum hematocrit level in lab results",
    "median.HEMATOCRIT": "(float) median hematocrit level in lab results",
    "q1.HEMATOCRIT": "(float) first quartile hematocrit level in lab results",
    "q3.HEMATOCRIT": "(float) third quartile hematocrit level in lab results",
    "max.HEMOGLOBIN": "(float) maximum hemoglobin level in lab results",
    "min.HEMOGLOBIN": "(float) minimum hemoglobin level in lab results",
    "median.HEMOGLOBIN": "(float) median hemoglobin level in lab results",
    "q1.HEMOGLOBIN": "(float) first quartile hemoglobin level in lab results",
    "q3.HEMOGLOBIN": "(float) third quartile hemoglobin level in lab results",
    "max.MEAN.CELLULAR.HEMOGLOBIN": "(float) maximum mean cellular hemoglobin in lab results",
    "min.MEAN.CELLULAR.HEMOGLOBIN": "(float) minimum mean cellular hemoglobin in lab results",
    "median.MEAN.CELLULAR.HEMOGLOBIN": "(float) median mean cellular hemoglobin in lab results",
    "q1.MEAN.CELLULAR.HEMOGLOBIN": "(float) first quartile mean cellular hemoglobin in lab results",
    "q3.MEAN.CELLULAR.HEMOGLOBIN": "(float) third quartile mean cellular hemoglobin in lab results",
    "max.MEAN.CELLULAR.HEMOGLOBIN.CONCENTRATION": "(float) maximum mean cellular hemoglobin concentration",
    "min.MEAN.CELLULAR.HEMOGLOBIN.CONCENTRATION": "(float) minimum mean cellular hemoglobin concentration",
    "median.MEAN.CELLULAR.HEMOGLOBIN.CONCENTRATION": "(float) median mean cellular hemoglobin concentration",
    "q1.MEAN.CELLULAR.HEMOGLOBIN.CONCENTRATION": "(float) first quartile mean cellular hemoglobin concentration",
    "q3.MEAN.CELLULAR.HEMOGLOBIN.CONCENTRATION": "(float) third quartile mean cellular hemoglobin concentration",
    "max.MEAN.CELLULAR.VOLUME": "(float) maximum mean cellular volume",
    "min.MEAN.CELLULAR.VOLUME": "(float) minimum mean cellular volume",
    "median.MEAN.CELLULAR.VOLUME": "(float) median mean cellular volume",
    "q1.MEAN.CELLULAR.VOLUME": "(float) first quartile mean cellular volume",
    "q3.MEAN.CELLULAR.VOLUME": "(float) third quartile mean cellular volume",
    "max.PLATELETS": "(float) maximum platelet count",
    "min.PLATELETS": "(float) minimum platelet count",
    "median.PLATELETS": "(float) median platelet count",
    "q1.PLATELETS": "(float) first quartile platelet count",
    "q3.PLATELETS": "(float) third quartile platelet count",
    "max.POTASSIUM": "(float) maximum potassium level in lab results",
    "min.POTASSIUM": "(float) minimum potassium level in lab results",
    "median.POTASSIUM": "(float) median potassium level in lab results",
    "q1.POTASSIUM": "(float) first quartile potassium level in lab results",
    "q3.POTASSIUM": "(float) third quartile potassium level in lab results",
    "max.PROTEIN.TOTAL": "(float) maximum total protein level in lab results",
    "min.PROTEIN.TOTAL": "(float) minimum total protein level in lab results",
    "median.PROTEIN.TOTAL": "(float) median total protein level in lab results",
    "q1.PROTEIN.TOTAL": "(float) first quartile total protein level in lab results",
    "q3.PROTEIN.TOTAL": "(float) third quartile total protein level in lab results",
    "max.RDW": "(float) maximum red cell distribution width",
    "min.RDW": "(float) minimum red cell distribution width",
    "median.RDW": "(float) median red cell distribution width",
    "q1.RDW": "(float) first quartile red cell distribution width",
    "q3.RDW": "(float) third quartile red cell distribution width",
    "max.RED.BLOOD.CELLS": "(float) maximum red blood cell count",
    "min.RED.BLOOD.CELLS": "(float) minimum red blood cell count",
    "median.RED.BLOOD.CELLS": "(float) median red blood cell count",
    "q1.RED.BLOOD.CELLS": "(float) first quartile red blood cell count",
    "q3.RED.BLOOD.CELLS": "(float) third quartile red blood cell count",
    "max.SODIUM": "(float) maximum sodium level in lab results",
    "min.SODIUM": "(float) minimum sodium level in lab results",
    "median.SODIUM": "(float) median sodium level in lab results",
    "q1.SODIUM": "(float) first quartile sodium level in lab results",
    "q3.SODIUM": "(float) third quartile sodium level in lab results",
    "max.THYROID.STIMULATING.HORMONE": "(float) maximum TSH level in lab results",
    "min.THYROID.STIMULATING.HORMONE": "(float) minimum TSH level in lab results",
    "median.THYROID.STIMULATING.HORMONE": "(float) median TSH level in lab results",
    "q1.THYROID.STIMULATING.HORMONE": "(float) first quartile TSH level in lab results",
    "q3.THYROID.STIMULATING.HORMONE": "(float) third quartile TSH level in lab results",
    "max.TRIGLYCERIDES": "(float) maximum triglyceride level in lab results",
    "min.TRIGLYCERIDES": "(float) minimum triglyceride level in lab results",
    "median.TRIGLYCERIDES": "(float) median triglyceride level in lab results",
    "q1.TRIGLYCERIDES": "(float) first quartile triglyceride level in lab results",
    "q3.TRIGLYCERIDES": "(float) third quartile triglyceride level in lab results",
    "max.UREA.NITROGEN": "(float) maximum urea nitrogen level in lab results",
    "min.UREA.NITROGEN": "(float) minimum urea nitrogen level in lab results",
    "median.UREA.NITROGEN": "(float) median urea nitrogen level in lab results",
    "q1.UREA.NITROGEN": "(float) first quartile urea nitrogen level in lab results",
    "q3.UREA.NITROGEN": "(float) third quartile urea nitrogen level in lab results",
    "max.WBC": "(float) maximum white blood cell count",
    "min.WBC": "(float) minimum white blood cell count",
    "median.WBC": "(float) median white blood cell count",
    "q1.WBC": "(float) first quartile white blood cell count",
    "q3.WBC": "(float) third quartile white blood cell count",
    "median_E03_9": "(float) median count of ICD code E03.9 (hypothyroidism)",
    "median_E11_9": "(float) median count of ICD code E11.9 (type 2 diabetes)",
    "median_E78_00": "(float) median count of ICD code E78.00 (pure hypercholesterolemia)",
    "median_E78_01": "(float) median count of ICD code E78.01 (familial hypercholesterolemia)",
    "median_E78_2": "(float) median count of ICD code E78.2 (mixed hyperlipidemia)",
    "median_E78_5": "(float) median count of ICD code E78.5 (hyperlipidemia unspecified)",
    "median_I10": "(float) median count of ICD code I10 (essential hypertension)",
    "median_I16_0": "(float) median count of ICD code I16.0 (hypertensive urgency)",
    "median_I16_1": "(float) median count of ICD code I16.1 (hypertensive emergency)",
    "median_I16_9": "(float) median count of ICD code I16.9 (unspecified hypertension crisis)",
    "sum_E03_8": "(int) total count of ICD code E03.8 (other hypothyroidism)",
    "sum_E03_9": "(int) total count of ICD code E03.9 (hypothyroidism unspecified)",
    "sum_E11_65": "(int) total count of ICD code E11.65 (diabetes with hyperglycemia)",
    "sum_E11_9": "(int) total count of ICD code E11.9 (type 2 diabetes with unspecified complications)",
    "sum_E66_01": "(int) total count of ICD code E66.01 (morbid obesity due to excess calories)",
    "sum_E66_09": "(int) total count of ICD code E66.09 (other obesity due to excess calories)",
    "sum_E66_1": "(int) total count of ICD code E66.1 (drug-induced obesity)",
    "sum_E66_8": "(int) total count of ICD code E66.8 (other obesity)",
    "sum_E66_9": "(int) total count of ICD code E66.9 (obesity unspecified)",
    "sum_E78_00": "(int) total count of ICD code E78.00 (pure hypercholesterolemia unspecified)",
    "sum_E78_01": "(int) total count of ICD code E78.01 (familial hypercholesterolemia)",
    "sum_E78_2": "(int) total count of ICD code E78.2 (mixed hyperlipidemia)",
    "sum_E78_5": "(int) total count of ICD code E78.5 (hyperlipidemia unspecified)",
    "sum_E87_6": "(int) total count of ICD code E87.6 (hypokalemia)",
    "sum_G47_30": "(int) total count of ICD code G47.30 (sleep apnea unspecified)",
    "sum_G47_33": "(int) total count of ICD code G47.33 (obstructive sleep apnea)",
    "sum_I10": "(int) total count of ICD code I10 (essential hypertension)",
    "sum_I16_0": "(int) total count of ICD code I16.0 (hypertensive urgency)",
    "sum_I16_1": "(int) total count of ICD code I16.1 (hypertensive emergency)",
    "sum_I16_9": "(int) total count of ICD code I16.9 (hypertensive crisis unspecified)",
    "sum_I25_10": "(int) total count of ICD code I25.10 (atherosclerotic heart disease of native coronary artery)",
    "sum_I48_0": "(int) total count of ICD code I48.0 (paroxysmal atrial fibrillation)",
    "sum_I48_1": "(int) total count of ICD code I48.1 (persistent atrial fibrillation)",
    "sum_I48_2": "(int) total count of ICD code I48.2 (chronic atrial fibrillation)",
    "sum_I48_91": "(int) total count of ICD code I48.91 (unspecified atrial fibrillation)",
    "sum_L70_8": "(int) total count of ICD code L70.8 (other acne)",
    "sum_N18_3": "(int) total count of ICD code N18.3 (chronic kidney disease stage 3)",
    "median_Diabetes_type_1": "(float) median count of type 1 diabetes diagnosis per year",
    "median_Dyslipidemias": "(float) median count of dyslipidemia diagnosis per year",
    "median_Essential_HTN": "(float) median count of essential hypertension diagnosis per year",
    "median_HTN_Emergency": "(float) median count of hypertension emergencies per year",
    "median_Hypothyroidism": "(float) median count of hypothyroidism diagnosis per year",
    "sum_ACNE": "(int) total count of acne diagnoses",
    "sum_Arrythmias": "(int) total count of arrhythmia diagnoses",
    "sum_Atrial_fibrillation": "(int) total count of atrial fibrillation diagnoses",
    "sum_CAD_native": "(int) total count of coronary artery disease of native artery diagnoses",
    "sum_CKD": "(int) total count of chronic kidney disease diagnoses",
    "sum_Diabetes_type_2": "(int) total count of type 2 diabetes diagnoses",
    "sum_Dyslipidemias": "(int) total count of dyslipidemia diagnoses",
    "sum_Essential_HTN": "(int) total count of essential hypertension diagnoses",
    "sum_Heart_Failure": "(int) total count of heart failure diagnoses",
    "sum_HTN_Emergency": "(int) total count of hypertensive emergencies",
    "sum_Hypokalemia": "(int) total count of hypokalemia diagnoses",
    "sum_Hypothyroidism": "(int) total count of hypothyroidism diagnoses",
    "sum_Obesity": "(int) total count of obesity diagnoses",
    "sum_Obstructive_Sleep_Apnea": "(int) total count of obstructive sleep apnea diagnoses",
    "Dx_N": "(int) total number of ICD codes recorded for the patient",
    "enc_N": "(int) total number of outpatient encounters",
    "dx_days": "(int) days between the first and last diagnosis recorded in the system",
    "HTN_MED_days_ACEI_ARB": "(int) total days prescribed ACE inhibitors or angiotensin receptor blockers",
    "HTN_MED_days_ALDOSTERONE_ANTAGONISTS": "(int) total days prescribed aldosterone antagonists",
    "HTN_MED_days_ALPHA_ANTAGONISTS": "(int) total days prescribed alpha antagonists",
    "HTN_MED_days_BETA_BLOCKERS": "(int) total days prescribed beta blockers",
    "HTN_MED_days_CENTRAL_ALPHA_AGONISTS": "(int) total days prescribed central alpha agonists",
    "HTN_MED_days_DIHYDRO_CCBS": "(int) total days prescribed dihydropyridine calcium channel blockers",
    "HTN_MED_days_HYDRALAZINE": "(int) total days prescribed hydralazine",
    "HTN_MED_days_K_SPARING_DIURETICS": "(int) total days prescribed potassium-sparing diuretics",
    "HTN_MED_days_LOOP_DIURETICS": "(int) total days prescribed loop diuretics",
    "HTN_MED_days_MINOXIDIL": "(int) total days prescribed minoxidil",
    "HTN_MED_days_NON_DIHYDRO_CCBS": "(int) total days prescribed non-dihydropyridine calcium channel blockers",
    "HTN_MED_days_RENIN_ANTAGONIST": "(int) total days prescribed renin antagonists",
    "HTN_MED_days_THIAZIDE": "(int) total days prescribed thiazide diuretics",
    "HTN_MED_days_POTASSIUM_CHLORIDE": "(int) total days prescribed potassium chloride",
    "MED_N": "(int) total number of medication prescriptions",
    "high_BP_during_htn_meds_1": "(int) number of high BP measurements (SBP >=140 or DBP >=90) while prescribed 1 hypertension medication",
    "high_BP_during_htn_meds_2": "(int) number of high BP measurements (SBP >=140 or DBP >=90) while prescribed 2 hypertension medications",
    "high_BP_during_htn_meds_3": "(int) number of high BP measurements (SBP >=140 or DBP >=90) while prescribed 3 hypertension medications",
    "high_BP_during_htn_meds_4_plus": "(int) number of high BP measurements (SBP >=140 or DBP >=90) while prescribed 4 or more hypertension medications",
    "sum_enc_during_htn_meds_1": "(int) total encounters while prescribed 1 hypertension medication",
    "sum_enc_during_htn_meds_2": "(int) total encounters while prescribed 2 hypertension medications",
    "sum_enc_during_htn_meds_3": "(int) total encounters while prescribed 3 hypertension medications",
    "sum_enc_during_htn_meds_4_plus": "(int) total encounters while prescribed 4+ hypertension medications",
    "median_enc_during_htn_meds_1": "(float) median encounters per year while prescribed 1 hypertension medication",
    "median_enc_during_htn_meds_2": "(float) median encounters per year while prescribed 2 hypertension medications",
    "median_enc_during_htn_meds_3": "(float) median encounters per year while prescribed 3 hypertension medications",
    "median_enc_during_htn_meds_4_plus": "(float) median encounters per year while prescribed 4+ hypertension medications",
    "sd_enc_during_htn_meds_1": "(float) standard deviation of encounters per year while prescribed 1 hypertension medication",
    "sd_enc_during_htn_meds_2": "(float) standard deviation of encounters per year while prescribed 2 hypertension medications",
    "sd_enc_during_htn_meds_3": "(float) standard deviation of encounters per year while prescribed 3 hypertension medications",
    "sd_enc_during_htn_meds_4_plus": "(float) standard deviation of encounters per year while prescribed 4+ hypertension medications",
    "skewness_enc_during_htn_meds_1": "(float) skewness of encounters per year while prescribed 1 hypertension medication",
    "skewness_enc_during_htn_meds_2": "(float) skewness of encounters per year while prescribed 2 hypertension medications",
    "skewness_enc_during_htn_meds_3": "(float) skewness of encounters per year while prescribed 3 hypertension medications",
    "skewness_enc_during_htn_meds_4_plus": "(float) skewness of encounters per year while prescribed 4+ hypertension medications",
    "N_med_K_chlo_enc": "(int) number of encounters on potassium chloride medication",
    "sd_med_K_chlo_enc": "(float) standard deviation of encounters on potassium chloride medication per year",
    "skewness_med_K_chlo_enc": "(float) skewness of encounters on potassium chloride medication per year",
    "low_K_N": "(int) total number of low potassium test results",
    "test_K_N": "(int) total number of potassium test results",
    "Med_Potassium_N": "(int) total number of potassium supplement prescriptions",
    "Dx_HypoK_N": "(int) total number of hypokalemia diagnoses",
    "ICD_hyp_sum": "(int) total count of hypertension-related ICD codes",
    "MED_HTN_N": "(int) total number of hypertension medication prescriptions",
    "bp_hyp_norm": "(float) ratio of high BP count to total BP measurements",
    "ICD_hyp_sum_norm": "(float) ratio of hypertension-related ICD codes to total ICD codes",
    "MED_HTN_N_norm": "(float) ratio of hypertension medication prescriptions to total prescriptions",
    "re_hyp_spe_norm": "(float) ratio of specific hypertension mentions to word count in clinical notes",
    "re_htn_sum": "(int) sum of regex counts for hypertension in clinical notes",
    "re_htn_spec_sum": "(int) sum of specific regex counts for hypertension in clinical notes",
    "re_htn_teixera_sum": "(int) sum of regex counts for hypertension (Teixeira method) in clinical notes",
    "re_word_count_sum": "(int) total word count in clinical notes",
    "re_htn_max": "(int) maximum regex count for hypertension in clinical notes",
    "re_htn_spec_max": "(int) maximum specific regex count for hypertension in clinical notes",
    "re_htn_teixera_max": "(int) maximum regex count for hypertension (Teixeira method) in clinical notes",
    "re_word_count_max": "(int) maximum word count in clinical notes",
    "re_htn_mean": "(float) mean regex count for hypertension in clinical notes",
    "re_htn_spec_mean": "(float) mean specific regex count for hypertension in clinical notes",
    "re_htn_teixera_mean": "(float) mean regex count for hypertension (Teixeira method) in clinical notes",
    "re_word_count_mean": "(float) mean word count in clinical notes",
    "re_htn_median": "(float) median regex count for hypertension in clinical notes",
    "re_htn_spec_median": "(float) median specific regex count for hypertension in clinical notes",
    "re_htn_teixera_median": "(float) median regex count for hypertension (Teixeira method) in clinical notes",
    "re_word_count_median": "(float) median word count in clinical notes",
    "re_htn_sd": "(float) standard deviation of regex counts for hypertension in clinical notes",
    "re_htn_spec_sd": "(float) standard deviation of specific regex counts for hypertension in clinical notes",
    "re_htn_teixera_sd": "(float) standard deviation of regex counts for hypertension (Teixeira method) in clinical notes",
    "re_word_count_sd": "(float) standard deviation of word count in clinical notes",
    "re_htn_skewness": "(float) skewness of regex counts for hypertension in clinical notes",
    "re_htn_spec_skewness": "(float) skewness of specific regex counts for hypertension in clinical notes",
    "re_htn_teixera_skewness": "(float) skewness of regex counts for hypertension (Teixeira method) in clinical notes",
    "re_word_count_skewness": "(float) skewness of word count in clinical notes",
    "htn_dx_ia": "(bool) heuristic-based prediction for hypertension diagnosis",
    "res_htn_dx_ia": "(bool) heuristic-based prediction for resistant hypertension diagnosis",
    "htn_hypok_dx_ia": "(bool) heuristic-based prediction for hypertension with hypokalemia diagnosis",
    "HTN_heuristic": "(bool) heuristic-based indicator for hypertension",
    "hypoK_heuristic_v4": "(bool) heuristic-based indicator for hypokalemia (version 4)",
    "res_HTN_heuristic": "(bool) heuristic-based indicator for resistant hypertension",
}
