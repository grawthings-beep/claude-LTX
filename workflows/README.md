# workflows

このディレクトリのJSONは起動時に ComfyUI の Workflows リストへコピーされる。

公式のLTX 2.3ワークフロー（`ltx23_official_two_stage_hq.json` /
`ltx23_official_single_stage.json`）はここには置いていない。イメージに
ピン留めされた ComfyUI-LTXVideo の `example_workflows/2.3/` から起動時に
コピーし、モデル名（fp8チェックポイント、fp4-mixed Gemma、`loras/ltx23`）を
パッチして設置する（`scripts/start.sh` の `install_official_workflows`）。

独自ワークフローを追加したい場合はこのディレクトリにJSONを置けばよい。
