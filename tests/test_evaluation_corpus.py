"""scripts/evaluation_corpus.py（評価スクリプト共通のドキュメント本文）のテスト。"""

import scripts.evaluate_model_accuracy as evaluate_model_accuracy
import scripts.evaluate_relevance_grading as evaluate_relevance_grading
import scripts.evaluate_retrieval as evaluate_retrieval
from scripts.evaluation_corpus import DOCUMENT_TEXTS, build_documents


def test_build_documents_defaults_to_all_ids():
    docs = build_documents()

    assert [d.metadata["doc_id"] for d in docs] == list(DOCUMENT_TEXTS)
    assert [d.page_content for d in docs] == list(DOCUMENT_TEXTS.values())


def test_build_documents_selects_ids_and_adds_extra_metadata():
    docs = build_documents(["return_policy"], thread_id="t")

    assert len(docs) == 1
    assert docs[0].page_content == DOCUMENT_TEXTS["return_policy"]
    assert docs[0].metadata == {"doc_id": "return_policy", "thread_id": "t"}


def test_retrieval_corpus_covers_all_shared_documents():
    assert {d.metadata["doc_id"] for d in evaluate_retrieval.CORPUS} == set(DOCUMENT_TEXTS)


def test_model_accuracy_corpus_uses_shared_texts():
    for doc in evaluate_model_accuracy.CORPUS:
        assert doc.page_content == DOCUMENT_TEXTS[doc.metadata["doc_id"]]


def test_relevance_grading_candidates_use_shared_texts():
    texts = set(DOCUMENT_TEXTS.values())
    for case in evaluate_relevance_grading.EVAL_CASES:
        for candidate in case["candidates"]:
            assert candidate["text"] in texts
