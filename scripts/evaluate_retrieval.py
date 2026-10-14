"""
検索精度（一次検索のベクトル類似度）の評価スクリプト。

rag_chain.py の retrieve_context ツールは
  1) 一次検索: ベクトル類似度で CANDIDATE_K 件を広めに取得し、
     L2距離が RECALL_DISTANCE_THRESHOLD 未満のものだけを候補として残す（粗いフィルタ）
  2) 二次検索: LLM（Ollama等）に採点させ、本当に関連するものだけに絞り込む（_grade_relevance）
の2段階で構成されている。

このスクリプトは 1) の一次検索だけを対象に、質問と正解ドキュメントのペア（評価セット）を使って
適合率(precision)・再現率(recall)を計算する。LLM採点（2）は使わない
（この評価はLLM起動の有無に依存せず、1) 単体の妥当性を検証したいため）。

2種類の評価を行う:
  - run_threshold_grid_search(): 現行モデル(EMBEDDING_MODEL_NAME)を固定し、
    CANDIDATE_K / RECALL_DISTANCE_THRESHOLD のグリッドサーチを行う（従来のチューニング用途）。
  - run_model_comparison(): CANDIDATE_K / RECALL_DISTANCE_THRESHOLD は本番採用値に固定し、
    埋め込みモデル自体（MODEL_CONFIGS）を比較する。全モデル×全グリッドの総当たりは
    時間がかかりすぎるため、モデル比較は本番値のみで行う。

本番の chroma_db/ や data/ は一切変更しない。評価用コーパスは一時ディレクトリに
専用コレクションとして作成し、実行後に破棄する。

使い方:
    python scripts/evaluate_retrieval.py
"""

import itertools
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chain import CANDIDATE_K, EMBEDDING_MODEL_NAME, RECALL_DISTANCE_THRESHOLD, get_embeddings
from scripts.evaluation_corpus import build_documents

# 評価専用コーパス（本番のdata/とは無関係）。
# 閾値・候補数の違いが見えるよう、各トピックに「正解ドキュメント」と、
# 語彙は近いが意味的には無関係な「ノイズドキュメント」を対にして混ぜている
# （rag_chain.py のコメントにある「単語は近いが意味的には無関係な文書」を再現するため）。
CORPUS = build_documents()

# 質問と正解ドキュメント（doc_id）のペア。
EVAL_SET = [
    {"query": "電化製品はいつまで返品できますか？", "relevant_ids": {"return_policy"}},
    {"query": "海外へ配送する場合、届くまでどれくらいかかりますか？", "relevant_ids": {"shipping_international"}},
    {"query": "Pythonでリストを1行で作る書き方を教えて", "relevant_ids": {"python_list_comprehension"}},
    {"query": "コミットメッセージの書き方のルールは？", "relevant_ids": {"git_commit_convention"}},
    {"query": "体調不良で休むときはどう連絡すればいい？", "relevant_ids": {"sick_leave_policy"}},
    {"query": "経費の領収書はいつまでに提出すればいいですか？", "relevant_ids": {"expense_report"}},
    {"query": "二要素認証はどうやって設定しますか？", "relevant_ids": {"two_factor_auth"}},
    {"query": "データベースのバックアップはどのくらいの頻度で行われますか？", "relevant_ids": {"db_backup_schedule"}},
    {"query": "新しく入社した人に初日にやることは？", "relevant_ids": {"onboarding_checklist"}},
    {"query": "この検索の仕組みは何段階になっていますか？", "relevant_ids": {"rag_architecture"}},
]

CANDIDATE_K_VALUES = [4, 8, 12]
DISTANCE_THRESHOLD_VALUES = [1.0, 1.3, 1.5]

# 埋め込みモデル比較の対象一覧。
# intfloat/multilingual-e5-* はモデルカード（HuggingFace README、config_sentence_transformers.json
# には未定義）上、検索タスクでは埋め込み対象のテキスト先頭に "query: " / "passage: " を
# 付けることが前提の設計であり、付けないと本来の性能が出ず不公平な比較になる。
# sentence-transformers/all-mpnet-base-v2 はプレフィックス不要。
# EMBEDDING_MODEL_NAME（本番採用中モデル）のエントリはquery_prefix/passage_prefixを空文字にする
# （get_embeddings()が内部の_PrefixedEmbeddingsで必要なプレフィックスを自動付与するため、
# ここで重ねて付けると二重付与になる）。
MODEL_CONFIGS = [
    {"name": EMBEDDING_MODEL_NAME, "query_prefix": "", "passage_prefix": ""},
    {"name": "sentence-transformers/all-mpnet-base-v2", "query_prefix": "", "passage_prefix": ""},
    {"name": "intfloat/multilingual-e5-large", "query_prefix": "query: ", "passage_prefix": "passage: "},
]


def build_eval_vectorstore(model_name: str = EMBEDDING_MODEL_NAME, passage_prefix: str = "") -> tuple[Chroma, str]:
    """評価専用の一時Chromaコレクションを作り、CORPUSを投入して返す。

    model_nameが現行モデル(EMBEDDING_MODEL_NAME)と一致する場合はget_embeddings()の
    lru_cacheキャッシュを再利用し、それ以外の比較対象モデルは都度ロードする。
    passage_prefixはmultilingual-e5系のように文書側テキストへのプレフィックス付与が
    前提のモデルを公平に評価するためのもの（CORPUSのpage_content自体は変更しない）。
    """
    tmp_dir = tempfile.mkdtemp(prefix="llm_practice_eval_retrieval_")
    embeddings = (
        get_embeddings()
        if model_name == EMBEDDING_MODEL_NAME
        else HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
    )
    vector_store = Chroma(
        collection_name="eval_retrieval",
        embedding_function=embeddings,
        persist_directory=tmp_dir,
    )
    corpus = (
        [Document(passage_prefix + doc.page_content, metadata=doc.metadata) for doc in CORPUS]
        if passage_prefix
        else CORPUS
    )
    vector_store.add_documents(corpus)
    return vector_store, tmp_dir


def evaluate(
    vector_store: Chroma, candidate_k: int, distance_threshold: float, query_prefix: str = ""
) -> tuple[float, float]:
    """指定したCANDIDATE_K / RECALL_DISTANCE_THRESHOLDでの平均適合率・平均再現率を返す。

    query_prefixはmultilingual-e5系のようにクエリ側テキストへのプレフィックス付与が
    前提のモデルを公平に評価するためのもの。
    """
    precisions = []
    recalls = []
    for case in EVAL_SET:
        candidates = vector_store.similarity_search_with_score(query_prefix + case["query"], k=candidate_k)
        retrieved_ids = {doc.metadata["doc_id"] for doc, score in candidates if score < distance_threshold}
        relevant_ids = case["relevant_ids"]
        true_positives = retrieved_ids & relevant_ids

        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(true_positives) / len(relevant_ids) if relevant_ids else 0.0
        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    return avg_precision, avg_recall


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def run_threshold_grid_search() -> None:
    """現行モデル(EMBEDDING_MODEL_NAME)を固定し、CANDIDATE_K / RECALL_DISTANCE_THRESHOLDの
    グリッドサーチ結果を表示する（既存のパラメータチューニング用評価）。
    """
    vector_store, tmp_dir = build_eval_vectorstore()
    try:
        print(f"[CANDIDATE_K / THRESHOLD グリッドサーチ] モデル: {EMBEDDING_MODEL_NAME}")
        print(f"評価用コーパス: {len(CORPUS)}件 / 評価クエリ: {len(EVAL_SET)}件\n")
        print(f"{'CANDIDATE_K':>12} | {'THRESHOLD':>9} | {'適合率(P)':>9} | {'再現率(R)':>9} | {'F1':>6}")
        print("-" * 62)
        for candidate_k, threshold in itertools.product(CANDIDATE_K_VALUES, DISTANCE_THRESHOLD_VALUES):
            precision, recall = evaluate(vector_store, candidate_k, threshold)
            f1 = _f1(precision, recall)
            print(f"{candidate_k:>12} | {threshold:>9.2f} | {precision:>9.3f} | {recall:>9.3f} | {f1:>6.3f}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_model_comparison() -> None:
    """CANDIDATE_K / RECALL_DISTANCE_THRESHOLDは本番採用値に固定し、埋め込みモデル自体を比較する。

    全モデル×全(CANDIDATE_K, THRESHOLD)の総当たりは時間がかかりすぎるため、
    モデル比較では本番採用中の値のみを使う。
    """
    print(f"\n[埋め込みモデル比較] CANDIDATE_K={CANDIDATE_K} / RECALL_DISTANCE_THRESHOLD={RECALL_DISTANCE_THRESHOLD}\n")
    print(f"{'モデル':<40} | {'適合率(P)':>9} | {'再現率(R)':>9} | {'F1':>6}")
    print("-" * 76)
    for config in MODEL_CONFIGS:
        vector_store, tmp_dir = build_eval_vectorstore(config["name"], config["passage_prefix"])
        try:
            precision, recall = evaluate(vector_store, CANDIDATE_K, RECALL_DISTANCE_THRESHOLD, config["query_prefix"])
            print(f"{config['name']:<40} | {precision:>9.3f} | {recall:>9.3f} | {_f1(precision, recall):>6.3f}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    run_threshold_grid_search()
    run_model_comparison()


if __name__ == "__main__":
    main()
