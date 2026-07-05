# RunPod セットアップ手順

## 1. GitHub Actions のビルドを待つ

main へ push すると `Build GHCR image` が走る。

```
https://github.com/grawthings-beep/claude-LTX/actions
```

緑になったら `ghcr.io/grawthings-beep/claude-ltx:cuda12.8` が使える。
初回はGHCRパッケージがprivateになるので、リポジトリの Packages 設定で
public にするか、RunPod側にレジストリ認証を設定すること。

## 2. RunPod テンプレート

Container image:

```
ghcr.io/grawthings-beep/claude-ltx:cuda12.8
```

HTTP port:

```
ComfyUI 8188
```

ディスク:

```
Container disk: 40 GB+
Volume / Network Volume: 150 GB+（モデル一式が数十GBあるため）
Volume mount path: /workspace
```

環境変数:

```
PORT=8188
LISTEN=0.0.0.0
RUN_DEP_CHECK=0
DOWNLOAD_MODELS=1
MODEL_DOWNLOAD_MODE=background
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
MODEL_MANIFEST_URL=https://raw.githubusercontent.com/grawthings-beep/claude-LTX/main/config/ltx-video-models.json
HF_XET_HIGH_PERFORMANCE=1
HF_HUB_DOWNLOAD_TIMEOUT=120
ARIA2_CONNECTIONS=8
ARIA2_SPLITS=8
DOWNLOAD_JOBS=1
VERIFY_MODEL_HASHES=once
COMFYUI_ARGS=--reserve-vram 5
```

対象モデルはすべて公開リポジトリなので `HF_TOKEN` が無くても落ちるはずだが、
レート制限回避のため設定を推奨。トークンは RunPod Secrets を使うこと。

## 3. 初回起動

ComfyUIはダウンロード完了前に開く。進捗確認:

```bash
cat /workspace/comfyui/logs/model-download.status
tail -f /workspace/comfyui/logs/model-download.log
```

`complete` になったらComfyUIをリロード。2回目以降の起動は
`/workspace/comfyui/models` を再利用してスキップされる。
全モデル完了までポートを開けたくない場合は `MODEL_DOWNLOAD_MODE=blocking`。

## 4. 生成

Workflows リストから選択:

```
ltx23_official_single_stage.json   反復・seed探し用（速い）
ltx23_official_two_stage_hq.json   本番用（二段HQ、遅いが高画質）
```

`LoadImage` に自分の画像を入れ、プロンプトを書いて生成。

## 5. 長尺動画

README の「長尺動画（20秒超）」を参照。two_stage_hq を API形式で
エクスポートしてから:

```bash
python3 /opt/claude-ltx/scripts/long_video.py \
  --workflow /workspace/comfyui/exports/two_stage_api.json \
  --image /workspace/comfyui/input/start.png \
  --segments 3 --frames 241 --seed 42 \
  --prompt "..." \
  --output /workspace/comfyui/output/long.mp4
```

まず `--segments 2` で境界の品質（動きの連続性・色）を確認してから
本数を増やすこと。
