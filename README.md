# claude-LTX

RunPod上のComfyUIで、LTX 2.3（公式 `ltx-2.3-22b-dev-fp8`）を使い「高画質」かつ「20秒超の長尺」動画を生成するためのテンプレート。

2つの問題をそれぞれ次の方法で解決する。

1. **画質**: 公式の二段パイプライン（低解像度で生成 → latent空間でx2スパシャルアップスケール → distilled LoRAで再refine）。ComfyUI-LTXVideo同梱の公式example workflowをそのまま使う。
2. **長さ**: LTX 2.3の1回の生成上限は20秒（24/25fps時）。それを超える長さは `scripts/long_video.py` が I2Vセグメントを連結して作る。各セグメントの最終フレームを次のセグメントの入力画像にし、最後にffmpegで結合する。

## 構成

```
Dockerfile                  runpod/comfyui ベースのGHCRイメージ
.github/workflows/          push時にGHCRへ自動ビルド
custom_nodes.txt            カスタムノードのピン留め（ComfyUI-LTXVideo等）
config/ltx-video-models.json  モデルmanifest（公式Lightricksモデルのみ）
scripts/start.sh            起動・manifest取得・ワークフロー設置・DL開始
scripts/download_models.py  hf_xet/aria2による再開可能ダウンローダ
scripts/long_video.py       長尺動画パイプライン（セグメント連結）
scripts/check_env.py        必須ノード・モデルの存在チェック
workflows/                  追加ワークフロー置き場（公式WFは起動時に自動設置）
```

## ダウンロードされるモデル（すべて公式）

| ファイル | 配置先 |
|---|---|
| `ltx-2.3-22b-dev-fp8.safetensors` (Lightricks/LTX-2.3-fp8) | `models/checkpoints` |
| `gemma_3_12B_it_fp4_mixed.safetensors` (Comfy-Org/ltx-2) | `models/text_encoders` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `models/latent_upscale_models` |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `models/loras/ltx23` |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors`（無効化済み、必要なら有効化） | `models/latent_upscale_models` |

## ワークフロー

起動時、ピン留めされたComfyUI-LTXVideoの公式exampleを、このイメージが実際に
ダウンロードするモデル名（fp8チェックポイント、fp4-mixed Gemma、`loras/ltx23`）に
パッチした上で Workflows リストへ設置する。

```
ltx23_official_two_stage_hq.json    二段HQ（低解像度生成 → latent x2 → refine）
ltx23_official_single_stage.json    単段（速度優先の反復用）
```

推奨手順: `single_stage` でseed・プロンプトを試行し、当たりが出たら
`two_stage_hq` で本番生成する。

## 長尺動画（20秒超）

`scripts/long_video.py` はComfyUI APIを叩いてI2Vセグメントを順番に生成し、
前セグメントの最終フレームを次の入力画像として連結する。

### 事前準備（1回だけ）

1. ComfyUIで `ltx23_official_two_stage_hq.json` を開き、解像度・fps等を決める
2. 設定 → Dev mode を有効化し、**Export (API)** でJSONを保存
   （例: `/workspace/comfyui/exports/two_stage_api.json`）

### 実行

RunPodのターミナルで:

```bash
python3 /opt/claude-ltx/scripts/long_video.py \
  --server http://127.0.0.1:8188 \
  --workflow /workspace/comfyui/exports/two_stage_api.json \
  --image /workspace/comfyui/input/start.png \
  --segments 4 --frames 241 --seed 42 \
  --prompt "segment 1 description..." \
  --prompt "segment 2 description..." \
  --prompt "segment 3 description..." \
  --prompt "segment 4 description..." \
  --output /workspace/comfyui/output/long_video.mp4
```

- `--frames` は `8*k+1`（121, 241, 361, ...）。241フレーム@24fpsで約10秒/セグメント
- `--prompt` はセグメント数より少なければ最後のものが使い回される
- セグメントの中間ファイルは `--keep-segments` で保持できる
- パッチ対象はノードタイトルで特定する。公式WF以外を使う場合は
  `--prompt-title` / `--frames-title` で合わせること

### この方式の既知の限界（重要）

- **ピクセル空間での連結**であり、latent空間のvideo extensionではない。
  境界フレームの静止画だけを引き継ぐため、**動きの速度・方向は境界で不連続**に
  なりうる。境界に来るカットは動きの少ない絵にするのが現実的な対策
- **音声はセグメントごとに独立生成**されるため、境界で音が途切れる・変わる。
  音が重要な用途では、無音で生成して後からBGM/SEを付ける方が安定する
- **色ドリフト**: セグメントを重ねるほど色味が初期画像からずれる可能性がある
- 上記がどの程度実害になるかは素材依存。まず `--segments 2` で境界品質を
  確認してから本数を増やすこと

latent空間でのextend（Kijai氏の "Extend Any Video" ワークフロー等、
https://huggingface.co/Kijai/LTX2.3_comfy ）の方が境界品質は原理的に有利。
必要ならそちらのワークフローをComfyUIに読み込んで手動運用し、本スクリプトは
自動化・一括生成用として使い分けるとよい。

## VRAM目安

Lightricks公式はbf16フルモデルに32GB+ VRAMを要求している。本イメージはfp8を
使うためそれより軽いが、二段HQ・高解像度・長フレームはVRAM消費が増える。
RunPodでは48GB級（L40S / RTX 6000 Ada / A6000）以上を推奨。OOMが出る場合は
`COMFYUI_ARGS=--reserve-vram 5` を維持しつつ、まず解像度かフレーム数を下げる。

## セットアップ

[RUNPOD_STEPS.md](RUNPOD_STEPS.md) を参照。

## 出典

- 二段パイプラインと公式workflowは [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)（コミット `229437c` にピン留め）
- インフラ構成（Docker/GHCR/manifest/ダウンローダ）は [ltx-video-runpod](https://github.com/grawthings-beep/ltx-video-runpod) をベースに公式モデル構成へ改修
