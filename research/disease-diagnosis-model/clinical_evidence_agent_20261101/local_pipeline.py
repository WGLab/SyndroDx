"""Wire frozen face/HPO ranking to offline evidence acquisition.

Private case inputs and returned traces must stay on HPC. Optional, unverified
genetic records do not block the face/HPO path or consume its question budget.
"""
from copy import deepcopy
from engine import run
from hpo_questions import selector


class LocalPipeline:
    def __init__(self, adapter, question_selector=None, variant_review_index=None):
        self.adapter = adapter
        if question_selector is None:
            from common import ROOT
            if (ROOT / 'agent_releases/CURRENT.json').exists():
                from learned_questions import current_selector
                question_selector = current_selector()
            else:
                question_selector = selector()
        self.question_selector = question_selector
        self.variant_review_index = variant_review_index

    def assess(self, case, native512_feature=None, visual_encoder_sha256=None,
               resolver=None, mode='face_hpo', max_actions=5, purpose='development'):
        if native512_feature is not None:
            expected = self.adapter.manifest['artifacts']['visual_encoder']['sha256']
            if visual_encoder_sha256 != expected:
                raise ValueError('unverified_visual_feature_encoder')
        core = deepcopy(case)
        if not core.get('patient_key') or any(e.get('patient_key') != core['patient_key'] for e in core.get('evidence', [])):
            raise ValueError('mismatched_patient_evidence')
        optional = [e for e in core.get('evidence', []) if e.get('kind') == 'variant']
        cards = []
        if optional:
            if self.variant_review_index is None:
                from variant_review import VariantReviewIndex
                self.variant_review_index = VariantReviewIndex()
            cards = self.variant_review_index.cards(core['patient_key'], optional, core.get('decision_time'))
        core['evidence'] = [e for e in core.get('evidence', []) if e.get('kind') != 'variant']
        result = run(core, ranker=self.adapter.bind(native512_feature, mode), resolver=resolver,
                     max_actions=max_actions, purpose=purpose, question_selector=self.question_selector)
        result.update(modality_policy='available_face_and_eligible_hpo_only',
                      optional_variant_records_held_for_review=len(optional),
                      variant_evidence_cards=cards,
                      variants_used_for_ranking=False, missing_genetics_blocks_core=False,
                      public_knowledge_supplied_patient_answers=False,
                      LLM_controller_used=False)
        return result
