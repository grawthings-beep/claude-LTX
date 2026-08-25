import hashlib
import importlib.util
import io
import os
import pathlib
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock


SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "download_models.py"
SPEC = importlib.util.spec_from_file_location("download_models", SCRIPT)
download_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_models)


class DownloadModelsTest(unittest.TestCase):
    def test_civitai_auth_query_comes_from_environment(self):
        with mock.patch.dict(os.environ, {"CIVITAI_API_TOKEN": "top secret"}):
            url = download_models.add_auth_query(
                "https://civitai.com/api/download/models/2266294?download=1",
                "CIVITAI_API_TOKEN",
            )

        self.assertIn("download=1", url)
        self.assertIn("token=top+secret", url)
        self.assertNotIn("top secret", download_models.redact_text(url))

    def test_exact_size_validation_rejects_both_short_and_long_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "model.zip"
            path.write_bytes(b"1234")
            self.assertFalse(download_models.validate_size(path, 5, 0))
            self.assertFalse(download_models.validate_size(path, 3, 0))
            self.assertTrue(download_models.validate_size(path, 4, 0))

    def test_extraction_cache_requires_every_provided_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            sentinel = root / "models" / "auto_mosaic" / ".model.zip.extracted"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("model.pt", encoding="utf-8")
            entry = {
                "provides": ["models/auto_mosaic/model.pt"],
            }
            self.assertFalse(
                download_models.extracted_ready(entry, root, sentinel)
            )
            (sentinel.parent / "model.pt").write_bytes(b"weights")
            self.assertTrue(download_models.extracted_ready(entry, root, sentinel))

    def test_verified_zip_extracts_the_exact_provided_model(self):
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("nested/ntd11_anime_nsfw_segm_v5.pt", b"weights")
            archive.writestr("ignored/readme.txt", b"ignored")
        archive_bytes = archive_buffer.getvalue()
        entry = {
            "name": "mosaic model",
            "url": "https://example.invalid/model.zip",
            "path": "models/auto_mosaic/model.zip",
            "size_bytes": len(archive_bytes),
            "sha256": hashlib.sha256(archive_bytes).hexdigest(),
            "extract": "pt",
            "provides": [
                "models/auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)

            def fake_download(url, output, headers):
                output.parent.mkdir(parents=True)
                output.write_bytes(archive_bytes)

            with mock.patch.object(
                download_models, "run_urllib", side_effect=fake_download
            ):
                download_models.download(entry, root, False, 8, 8, "always")

            extracted = root / entry["provides"][0]
            self.assertEqual(extracted.read_bytes(), b"weights")
            self.assertFalse((root / entry["path"]).exists())
            sentinel = extracted.parent / ".model.zip.extracted"
            self.assertTrue(sentinel.is_file())

            with mock.patch.object(
                download_models,
                "run_urllib",
                side_effect=AssertionError("verified extraction should be reused"),
            ):
                download_models.download(entry, root, False, 8, 8, "always")

    def test_authenticated_download_errors_redact_the_secret(self):
        entry = {
            "name": "private model",
            "url": "https://example.invalid/model.zip",
            "path": "models/model.zip",
            "required": True,
            "requires_env": ["CIVITAI_API_TOKEN"],
            "auth_query_env": "CIVITAI_API_TOKEN",
        }
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ, {"CIVITAI_API_TOKEN": "top secret"}
        ), mock.patch.object(
            download_models,
            "run_urllib",
            side_effect=RuntimeError("failed URL token=top+secret"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                download_models.download(
                    entry, pathlib.Path(temp_dir), False, 8, 8, "once"
                )

        self.assertNotIn("top secret", str(raised.exception))
        self.assertNotIn("top+secret", str(raised.exception))
        self.assertIn("[REDACTED]", str(raised.exception))

    def test_parse_huggingface_resolve_url(self):
        parsed = download_models.parse_huggingface_url(
            "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/"
            "split_files/text_encoders/model.safetensors?download=true"
        )

        self.assertEqual(
            parsed,
            (
                "Comfy-Org/ltx-2",
                "main",
                "split_files/text_encoders/model.safetensors",
            ),
        )

    def test_hf_hub_download_moves_nested_file_to_requested_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "model.safetensors"

            def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "org/repo")
                self.assertEqual(kwargs["revision"], "main")
                self.assertEqual(kwargs["filename"], "nested/model.safetensors")
                nested = pathlib.Path(kwargs["local_dir"]) / kwargs["filename"]
                nested.parent.mkdir(parents=True)
                nested.write_bytes(b"xet data")
                return str(nested)

            fake_module = types.SimpleNamespace(
                get_hf_file_metadata=lambda url, token: types.SimpleNamespace(
                    etag='"abc123"'
                ),
                hf_hub_download=fake_download,
                hf_hub_url=lambda repo_id, filename, revision: (
                    f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
                ),
            )
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_module}):
                remote_etag = download_models.run_hf_hub(
                    "https://huggingface.co/org/repo/resolve/main/nested/model.safetensors",
                    output,
                    {"Authorization": "Bearer token"},
                )

            self.assertEqual(output.read_bytes(), b"xet data")
            self.assertEqual(remote_etag, "ABC123")

    def test_aria2_uses_partial_file_then_renames_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "model.safetensors"

            def fake_run(command, check):
                self.assertTrue(check)
                directory = pathlib.Path(command[command.index("-d") + 1])
                filename = command[command.index("-o") + 1]
                self.assertEqual(filename, "model.safetensors.part")
                self.assertFalse(output.exists())
                (directory / filename).write_bytes(b"complete")

            with mock.patch.object(download_models.subprocess, "run", side_effect=fake_run):
                download_models.run_aria2("https://example.invalid/model", output, 8, 8)

            self.assertEqual(output.read_bytes(), b"complete")
            self.assertFalse(download_models.partial_path(output).exists())

    def test_legacy_aria2_download_is_moved_to_partial_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "model.safetensors"
            legacy_control = pathlib.Path(str(output) + ".aria2")
            output.write_bytes(b"incomplete")
            legacy_control.write_bytes(b"aria state")

            download_models.migrate_legacy_aria_download(output)

            partial = download_models.partial_path(output)
            self.assertFalse(output.exists())
            self.assertEqual(partial.read_bytes(), b"incomplete")
            self.assertEqual(pathlib.Path(str(partial) + ".aria2").read_bytes(), b"aria state")

    def test_once_verification_uses_marker_on_later_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "model.safetensors"
            output.write_bytes(b"model data")
            expected = hashlib.sha256(b"model data").hexdigest().upper()

            self.assertTrue(download_models.verify_sha256(output, expected, "model", "once"))

            with mock.patch.object(
                download_models,
                "sha256_file",
                side_effect=AssertionError("file should not be rehashed"),
            ):
                self.assertTrue(download_models.verify_sha256(output, expected, "model", "once"))

    def test_fresh_hf_download_trusts_matching_content_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            content = b"model data"
            expected = hashlib.sha256(content).hexdigest().upper()
            entry = {
                "name": "model",
                "url": "https://huggingface.co/org/repo/resolve/main/model.safetensors",
                "path": "models/model.safetensors",
                "sha256": expected,
            }

            def fake_hf_download(url, output, headers):
                output.parent.mkdir(parents=True)
                output.write_bytes(content)
                return expected

            with mock.patch.object(
                download_models,
                "run_hf_hub",
                side_effect=fake_hf_download,
            ), mock.patch.object(
                download_models,
                "sha256_file",
                side_effect=AssertionError("fresh Xet file should not be rehashed"),
            ):
                download_models.download(entry, root, True, 8, 8, "once")

            output = root / entry["path"]
            self.assertTrue(download_models.has_cached_verification(output, expected))


if __name__ == "__main__":
    unittest.main()
