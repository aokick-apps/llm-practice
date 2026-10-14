"""
LLM採点（rag_chain.py の `_grade_relevance`、reranking相当）の適合率・再現率を評価するスクリプト。

rag_chain.py の retrieve_context ツールは
  1) 一次検索: ベクトル類似度で候補を広めに集める（scripts/evaluate_retrieval.py が評価対象）
  2) 二次検索: LLM（Ollama等）に候補を採点させ、本当に関連するものだけに絞り込む（_grade_relevance）
の2段階で構成されている。

このスクリプトは 2) の LLM採点だけを対象に、質問・候補文書・正解ラベル（関連/非関連）を
組にした評価セットを使って適合率(precision)・再現率(recall)・F1を計算する。
_grade_relevance は実際にOllama等のローカルLLM（setup.py の model）を呼び出すため、
Ollamaが起動していない環境（CI含む）では実行できない、手動実行専用のスクリプトである。
一次検索は使わず実行主体・実行コストが異なるため、evaluate_retrieval.py とは別スクリプトにしている。

本番の chroma_db/ や data/ には一切アクセスしない。

使い方:
    python scripts/evaluate_relevance_grading.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from rag_chain import _grade_relevance
from scripts.evaluation_corpus import DOCUMENT_TEXTS


def _candidate(text: str, relevant: bool) -> dict:
    return {"text": text, "relevant": relevant}


# 質問ごとに、候補文書（本文テキスト）と正解ラベル（関連=True/非関連=False）の組を用意する。
# 語彙は近いが意味的には無関係な「ノイズ候補」を混ぜ、一次検索の粗いフィルタでは弾けない
# 誤検出をLLM採点（_grade_relevance）が正しく除外できるかを検証する。
EVAL_CASES = [
    {
        "query": "電化製品はいつまで返品できますか？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["return_policy"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["exchange_policy"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["shipping_domestic"],
                False,
            ),
        ],
    },
    {
        "query": "海外へ配送する場合、届くまでどれくらいかかりますか？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["shipping_international"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["shipping_domestic"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["return_policy"],
                False,
            ),
        ],
    },
    {
        "query": "Pythonでリストを1行で作る書き方を教えて",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["python_list_comprehension"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["python_dict_comprehension"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["git_commit_convention"],
                False,
            ),
        ],
    },
    {
        "query": "コミットメッセージの書き方のルールは？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["git_commit_convention"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["git_branch_naming"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["vacation_request"],
                False,
            ),
        ],
    },
    {
        "query": "体調不良で休むときはどう連絡すればいい？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["sick_leave_policy"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["vacation_request"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["expense_report"],
                False,
            ),
        ],
    },
    {
        "query": "経費の領収書はいつまでに提出すればいいですか？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["expense_report"],
                True,
            ),
            _candidate(DOCUMENT_TEXTS["travel_reimbursement"], False),
            _candidate(
                DOCUMENT_TEXTS["password_policy"],
                False,
            ),
        ],
    },
    {
        "query": "パスワードやAPIキーのセキュリティルールを教えて",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["password_policy"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["api_key_rotation"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["db_backup_schedule"],
                False,
            ),
        ],
    },
    {
        "query": "データベースのバックアップはどのくらいの頻度で行われますか？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["db_backup_schedule"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["two_factor_auth"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["onboarding_checklist"],
                False,
            ),
        ],
    },
    {
        "query": "新しく入社した人に初日にやることは？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["onboarding_checklist"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["offboarding_checklist"],
                False,
            ),
            _candidate(DOCUMENT_TEXTS["meeting_room_booking"], False),
        ],
    },
    {
        "query": "この検索の仕組みは何段階になっていますか？",
        "candidates": [
            _candidate(
                DOCUMENT_TEXTS["rag_architecture"],
                True,
            ),
            _candidate(
                DOCUMENT_TEXTS["chroma_choice"],
                False,
            ),
            _candidate(
                DOCUMENT_TEXTS["printer_troubleshooting"],
                False,
            ),
        ],
    },
]


def evaluate_case(case: dict) -> tuple[float, float]:
    """1件の評価ケースについて、_grade_relevanceの適合率・再現率を返す。"""
    docs = [Document(candidate["text"]) for candidate in case["candidates"]]
    predicted_idx = set(_grade_relevance(case["query"], docs))
    relevant_idx = {i for i, candidate in enumerate(case["candidates"]) if candidate["relevant"]}

    true_positives = predicted_idx & relevant_idx
    precision = len(true_positives) / len(predicted_idx) if predicted_idx else 0.0
    recall = len(true_positives) / len(relevant_idx) if relevant_idx else 0.0
    return precision, recall


def main() -> None:
    print(f"評価ケース数: {len(EVAL_CASES)}件（Ollama等のローカルLLMを実際に呼び出すため実行に時間がかかります）\n")
    print(f"{'質問':40} | {'適合率(P)':>9} | {'再現率(R)':>9} | {'F1':>6}")
    print("-" * 76)

    precisions = []
    recalls = []
    for case in EVAL_CASES:
        precision, recall = evaluate_case(case)
        precisions.append(precision)
        recalls.append(recall)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        print(f"{case['query'][:40]:40} | {precision:>9.3f} | {recall:>9.3f} | {f1:>6.3f}")

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = (
        (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0
    )
    print("-" * 76)
    print(f"{'平均':40} | {avg_precision:>9.3f} | {avg_recall:>9.3f} | {avg_f1:>6.3f}")


if __name__ == "__main__":
    main()
