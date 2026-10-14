"""評価スクリプト間で共有する評価用ドキュメント本文。

質問と期待値は評価軸ごとに異なるため各スクリプトで持ち、本文のみをここで一元管理する。
"""

from langchain_core.documents import Document

DOCUMENT_TEXTS = {
    "return_policy": (
        "返品ポリシー: 電化製品は購入から30日以内であれば未使用品に限り返品を受け付けます。"
        "返金は元の支払い方法に対して7営業日以内に行われます。"
    ),
    "exchange_policy": (
        "交換ポリシー: セール品・アウトレット品は返品不可ですが、サイズ違いによる交換のみ"
        "購入後14日以内であれば承ります。"
    ),
    "shipping_domestic": (
        "国内配送: 国内への配送は通常3〜5営業日で到着します。送料は5000円以上のご注文で無料になります。"
    ),
    "shipping_international": (
        "海外配送: 海外への配送には通常10〜15営業日かかります。関税はお客様負担となる場合があります。"
    ),
    "python_list_comprehension": (
        "Pythonのリスト内包表記は `[式 for 変数 in イテラブル]` の形式で、ループを1行で簡潔に書ける構文です。"
    ),
    "python_dict_comprehension": (
        "Pythonの辞書内包表記は `{キー: 値 for 変数 in イテラブル}` の形式で、辞書を1行で構築できます。"
    ),
    "git_branch_naming": (
        "このプロジェクトのブランチ命名規則は `feature/<内容>` や `fix/<内容>` のように"
        "プレフィックスを付けることになっています。"
    ),
    "git_commit_convention": (
        "コミットメッセージは Conventional Commits（feat:, fix:, test: など）の形式に従う必要があります。"
    ),
    "vacation_request": ("有給休暇の申請は、取得希望日の2週間前までに人事ポータルから申請してください。"),
    "sick_leave_policy": ("体調不良で欠勤する場合は、当日の始業前までに上長にチャットで連絡してください。"),
    "expense_report": ("経費精算は、領収書を経費システムにアップロードし、支出から30日以内に申請する必要があります。"),
    "travel_reimbursement": ("出張旅費の払い戻しには、出張前に上長の事前承認が必要です。"),
    "password_policy": ("パスワードポリシー: パスワードは12文字以上とし、90日ごとに変更する必要があります。"),
    "two_factor_auth": ("二要素認証（2FA）は、設定ページの「セキュリティ」タブから認証アプリを使って有効化できます。"),
    "api_key_rotation": ("APIキーは90日ごとにローテーションし、Vaultで安全に管理する必要があります。"),
    "db_backup_schedule": ("データベースのバックアップは毎晩深夜2時に自動実行され、30日分保存されます。"),
    "onboarding_checklist": (
        "新入社員のオンボーディングでは、初日にPCの貸与とアカウント発行、社内システムの説明を行います。"
    ),
    "offboarding_checklist": (
        "退職者のオフボーディングでは、最終出社日にアクセス権限を全て失効させ、貸与物を回収します。"
    ),
    "rag_architecture": (
        "このアプリの検索は2段階構成です。まずベクトル類似度で候補を広めに集め、その後LLMに採点させて関連する文書だけに絞り込みます。"
    ),
    "chroma_choice": (
        "ベクトルDBにはChromaを採用しています。ローカルに永続化でき、追加のサーバー起動が不要なためです。"
    ),
    "meeting_room_booking": ("会議室の予約は社内カレンダーアプリから行えます。30分単位で予約可能です。"),
    "printer_troubleshooting": (
        "プリンターの紙詰まりが起きた場合は、電源を切ってから用紙トレイをゆっくり引き出してください。"
    ),
}


def build_documents(doc_ids: list[str] | None = None, **extra_metadata) -> list[Document]:
    """DOCUMENT_TEXTSからDocumentリストを作る。doc_ids省略時は全件、extra_metadataは各Documentのmetadataに追加する。"""
    ids = doc_ids if doc_ids is not None else list(DOCUMENT_TEXTS)
    return [Document(DOCUMENT_TEXTS[i], metadata={"doc_id": i, **extra_metadata}) for i in ids]
