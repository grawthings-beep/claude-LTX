import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ContainerContractTests(unittest.TestCase):
    def test_ultralytics_is_pinned_installed_and_verified(self):
        requirements = (
            ROOT / "custom_nodepacks" / "ComfyUI-LTXLoop" / "requirements.txt"
        ).read_text(encoding="utf-8")
        installer = (ROOT / "scripts" / "install_custom_nodes.sh").read_text(
            encoding="utf-8"
        )
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertEqual(requirements.strip(), "ultralytics==8.4.104")
        self.assertIn('"${target}/requirements.txt"', installer)
        self.assertIn("import ultralytics", dockerfile)
        self.assertIn("ultralytics.__version__ == '8.4.104'", dockerfile)

    def test_startup_blocks_only_on_the_small_required_mosaic_model(self):
        start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
        env_example = (ROOT / "runpod-template.env.example").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${MODEL_ROOT}/models/auto_mosaic"', start)
        self.assertIn("export MODEL_ROOT", start)
        self.assertIn("YOLO_CONFIG_DIR", start)
        self.assertIn('YOLO_AUTOINSTALL="${YOLO_AUTOINSTALL:-false}"', start)
        self.assertIn('YOLO_OFFLINE="${YOLO_OFFLINE:-true}"', start)
        self.assertIn("--only-group auto-mosaic", start)
        self.assertIn("--exclude-group auto-mosaic", start)
        self.assertIn("CIVITAI_API_TOKEN as a RunPod Secret", start)
        self.assertIn(
            "CIVITAI_API_TOKEN={{ RUNPOD_SECRET_CIVITAI_API_TOKEN }}",
            env_example,
        )
        self.assertLess(start.index("ensure_auto_mosaic_model"), start.index("case \"${MODEL_DOWNLOAD_MODE}\""))
        self.assertLess(start.index("ensure_auto_mosaic_model"), start.index('exec "${PYTHON_BIN}" main.py'))

    def test_downloader_preserves_resume_and_strict_archive_contracts(self):
        downloader = (ROOT / "scripts" / "download_models.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('output.with_name(output.name + ".part")', downloader)
        self.assertIn("--continue=true", downloader)
        self.assertIn("size_bytes", downloader)
        self.assertIn("provides", downloader)
        self.assertIn("auth_query_env", downloader)
        self.assertIn("add_auth_query", downloader)
        self.assertIn("redact_text", downloader)

    def test_real_entrypoint_smoke_runs_during_docker_build(self):
        smoke = (ROOT / "scripts" / "container_smoke.sh").read_text(
            encoding="utf-8"
        )
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("/opt/claude-ltx/scripts/start.sh", smoke)
        self.assertIn("--quick-test-for-ci", smoke)
        self.assertIn("DOWNLOAD_MODELS=0", smoke)
        self.assertIn("ntd11_anime_nsfw_segm_v5.pt", smoke)
        self.assertIn("container_smoke.sh", dockerfile)

    def test_ci_validates_generated_workflow_and_mosaic_node(self):
        workflow = (ROOT / ".github" / "workflows" / "build-ghcr.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/prepare_workflows.py --check", workflow)
        self.assertIn("mosaic_nodes.py", workflow)


if __name__ == "__main__":
    unittest.main()
