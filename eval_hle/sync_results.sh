#!/bin/bash
#
# サーバーから推論結果をダウンロードするスクリプト。
# 実行前に、ローカルの既存の結果をタイムスタンプ付きでバックアップする。
#

# --- 設定 ---
REMOTE_USER="P10U029"
REMOTE_HOST="10.255.255.101"
SSH_KEY_PATH="~/.ssh/id_ed25519"

# サーバー上の結果ディレクトリのパス
REMOTE_PREDICTIONS_PATH="~/llm_bridge_prod/eval_hle/predictions/"
REMOTE_LEADERBOARD_PATH="~/llm_bridge_prod/eval_hle/leaderboard/"
REMOTE_JUDGED_PATH="~/llm_bridge_prod/eval_hle/judged/"

# ローカルの保存先ディレクトリ
LOCAL_TARGET_DIR="./eval_hle_results/"
BACKUP_DIR="./backup/"


# --- メイン処理 ---
echo "古い結果のバックアップを開始します..."

# バックアップディレクトリと保存先ディレクトリがなければ作成
mkdir -p "$BACKUP_DIR"
mkdir -p "$LOCAL_TARGET_DIR"

# ローカルに前回の結果ディレクトリが存在する場合のみバックアップ処理を行う
if [ "$(ls -A $LOCAL_TARGET_DIR 2>/dev/null)" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_NAME="results_${TIMESTAMP}"
    
    echo "  -> '${LOCAL_TARGET_DIR}' を '${BACKUP_DIR}${BACKUP_NAME}' に移動します。"
    mv "$LOCAL_TARGET_DIR" "${BACKUP_DIR}${BACKUP_NAME}"
    mkdir -p "$LOCAL_TARGET_DIR" # 次のrsyncのために空のディレクトリを再作成
else
    echo "  -> バックアップ対象が見つからないため、スキップします。"
fi

echo ""
echo "サーバーから最新の結果を同期します..."

# predictions ディレクトリを同期
rsync -avz -e "ssh -i $SSH_KEY_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PREDICTIONS_PATH}" "${LOCAL_TARGET_DIR}predictions/"

# leaderboard ディレクトリを同期
rsync -avz -e "ssh -i $SSH_KEY_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LEADERBOARD_PATH}" "${LOCAL_TARGET_DIR}leaderboard/"

# judged ディレクトリを同期
rsync -avz -e "ssh -i $SSH_KEY_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_JUDGED_PATH}" "${LOCAL_TARGET_DIR}judged/"

echo ""
echo "同期が完了しました。"
echo "最新の結果は '${LOCAL_TARGET_DIR}' にあります。"
ls -lR "$LOCAL_TARGET_DIR"