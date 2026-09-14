"""Fixed-budget matched risk heads and feature registry."""
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEXT = ['sequence_nll', 'token_entropy', 'beam_margin_risk', 'output_length']
ASR = ['asr_sequence_nll', 'asr_token_entropy']
WAVE = ['audio_duration', 'audio_rms_db', 'audio_zero_fraction', 'audio_spectral_entropy', 'audio_clipping_fraction']
PROBES = ['perturb_mean_distance', 'perturb_max_distance']
SAMPLES = ['sample_mean_distance', 'sample_number_disagreement', 'sample_negation_disagreement']
EVIDENCE = ['cascade_sequence_nll', 'evidence_text_distance', 'evidence_number_mismatch', 'evidence_negation_mismatch', 'evidence_coverage_gap']
FULL = TEXT + ASR + WAVE + PROBES + SAMPLES + EVIDENCE
FEATURES = {
    'et_full': FULL,
    'et_evidence': EVIDENCE,
    'et_asr_evidence': ASR + EVIDENCE,
    'et_without_acoustic': TEXT + PROBES + SAMPLES + EVIDENCE,
    'et_without_stability': TEXT + ASR + WAVE + EVIDENCE,
    'et_without_evidence': TEXT + ASR + WAVE + PROBES + SAMPLES,
    'et_without_waveform': TEXT + ASR + PROBES + SAMPLES + EVIDENCE,
    'et_without_sampling': TEXT + ASR + WAVE + PROBES + EVIDENCE,
    'et_without_probes': TEXT + ASR + WAVE + SAMPLES + EVIDENCE,
    'lr_full': FULL,
    'hgb_full': FULL,
    'lr_sequence_probability': ['sequence_nll'],
    'lr_token_entropy': ['token_entropy'],
    'lr_asr': ASR,
    'lr_qe': ['qe_risk'],
    'lr_self_consistency': SAMPLES,
    'lr_perturbation': PROBES,
    'lr_evidence': EVIDENCE,
}


def build_model(name, seed=20260906):
    if name.startswith('et_'):
        clf = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=20, max_features=.7,
                                   class_weight='balanced', n_jobs=8, random_state=seed)
    elif name.startswith('hgb_'):
        clf = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
                min_samples_leaf=30, learning_rate=.05, l2_regularization=2., random_state=seed)
    else:
        clf = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=seed)
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), clf)
