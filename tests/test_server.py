import tempfile
from unittest import mock
from pathlib import Path
import time
import unittest
from types import SimpleNamespace

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
from dataclasses import replace

from translategemma.workers.engine import TranslateGemmaEngine
from translategemma.workers.generation import plan_generation
from translategemma.workers.manager import TranslationManager

from server import (
    Config,
    Job,
    JobStore,
    Runtime,
    StoreFullError,
    ValidationError,
    WorkerNotReadyError,
    create_app,
    parse_image_translation_payload,
    parse_translation_payload,
)


def make_config(**overrides):
    base = Config(
        model_path="/tmp/model",
        host="127.0.0.1",
        port=7860,
        max_gpu_workers=2,
        gpu_ids="0,1",
        allow_cpu_fallback=False,
        stagger_worker_start=True,
        worker_start_mode="auto",
        worker_parallel_cache_min_bytes=1_000_000,
        worker_parallel_min_available_ram_mb=24_576,
        worker_load_timeout=10,
        max_worker_restarts=1,
        max_queue_size=4,
        max_store_size=10,
        result_ttl_seconds=60,
        max_input_chars=100,
        vision_enabled=False,
        max_image_bytes=524288,
        max_image_width=8192,
        max_image_height=8192,
        max_image_pixels=20_000_000,
        default_output_tokens=32,
        max_output_tokens=64,
        request_timeout=0.01,
        shutdown_timeout=1,
        max_request_bytes=4096,
        model_dtype="bfloat16",
        jax_preallocate=True,
        jax_mem_fraction=0.9,
        jax_compilation_cache_dir=None,
        jax_persistent_cache_min_compile_time_secs=1.0,
        jax_persistent_cache_min_entry_size_bytes=-1,
        generation_bucketing=True,
        generation_length_buckets=(256, 512, 1024, 1536, 2048),
        generation_bucket_step=512,
        warmup_enabled=False,
        warmup_output_tokens=32,
        warmup_text_buckets=(256,),
        warmup_vision_buckets=(512,),
        api_auth_required=True,
        api_key="test-api-key",
        restart_secret="test-restart-secret",
    )
    return replace(base, **overrides)


class FakeManager:
    def __init__(self, config):
        self.config = config
        self.store = JobStore(config.max_store_size, config.result_ttl_seconds)
        self.next_job = None

    def health(self):
        return {
            "state": "ready",
            "ready": True,
            "ready_workers": 2,
            "expected_workers": 2,
            "accepting_jobs": True,
            "jobs": self.store.stats(),
            "workers": [],
            "detected_gpus": [],
        }

    def submit(self, payload):
        if self.next_job is not None:
            job = self.next_job
        else:
            job = Job(
                id="job-test",
                text=payload["text"],
                src=payload["src"],
                tgt=payload["tgt"],
                max_tokens=payload["max_tokens"],
            )
        self.store.put(job)
        return job

    def shutdown(self, wait_for_jobs, timeout):
        return True


class GenerationPlanningTests(unittest.TestCase):
    def test_text_default_quantizes_to_256(self):
        plan = plan_generation(
            prompt_tokens=48,
            requested_new_tokens=128,
            buckets=(256, 512, 1024),
            fallback_step=512,
        )
        self.assertEqual(plan.exact_max_length, 184)
        self.assertEqual(plan.compiled_max_length, 256)
        self.assertTrue(plan.bucketed)

    def test_vision_default_quantizes_to_512(self):
        plan = plan_generation(
            prompt_tokens=64,
            vision_tokens=256,
            requested_new_tokens=128,
            buckets=(256, 512, 1024),
            fallback_step=512,
        )
        self.assertEqual(plan.exact_max_length, 456)
        self.assertEqual(plan.compiled_max_length, 512)

    def test_large_request_uses_fallback_step(self):
        plan = plan_generation(
            prompt_tokens=2100,
            requested_new_tokens=128,
            buckets=(256, 512, 1024, 1536, 2048),
            fallback_step=512,
        )
        self.assertEqual(plan.compiled_max_length, 2560)

    def test_bucketing_can_be_disabled(self):
        plan = plan_generation(
            prompt_tokens=48,
            requested_new_tokens=128,
            bucketing_enabled=False,
            buckets=(256, 512),
            fallback_step=512,
        )
        self.assertEqual(plan.compiled_max_length, plan.exact_max_length)
        self.assertFalse(plan.bucketed)


class EngineOutputBudgetTests(unittest.TestCase):
    class FakeTokenizer:
        def tokenize(self, text):
            return text.split()

        def detokenize(self, tokens):
            return " ".join(tokens)

    def test_bucket_slack_is_trimmed_to_requested_new_tokens(self):
        engine = TranslateGemmaEngine(
            preset_path="/tmp/not-loaded",
            dtype="bfloat16",
        )
        engine.preprocessor = SimpleNamespace(tokenizer=self.FakeTokenizer())
        result = engine._finalize_output(
            "one two three four five",
            max_tokens=3,
            bucket_has_slack=True,
        )
        self.assertEqual(result, "one two three")

    def test_exact_length_path_does_not_retokenize(self):
        engine = TranslateGemmaEngine(
            preset_path="/tmp/not-loaded",
            dtype="bfloat16",
        )
        engine.preprocessor = SimpleNamespace(tokenizer=self.FakeTokenizer())
        result = engine._finalize_output(
            "one two three four five",
            max_tokens=3,
            bucket_has_slack=False,
        )
        self.assertEqual(result, "one two three four five")


class StartupStrategyTests(unittest.TestCase):
    def test_auto_uses_stagger_for_cold_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            config = make_config(
                jax_compilation_cache_dir=cache,
                worker_start_mode="auto",
                worker_parallel_cache_min_bytes=10,
                worker_parallel_min_available_ram_mb=100,
            )
            manager = TranslationManager(config)
            with mock.patch.object(manager, "_available_ram_mb", return_value=1000):
                self.assertEqual(manager._resolve_startup_mode(2), "stagger")
            self.assertEqual(manager._startup_strategy["reason"], "cold-cache")

    def test_auto_uses_parallel_for_warm_cache_and_ram(self):
        with tempfile.TemporaryDirectory() as cache:
            Path(cache, "compiled.bin").write_bytes(b"x" * 32)
            config = make_config(
                jax_compilation_cache_dir=cache,
                worker_start_mode="auto",
                worker_parallel_cache_min_bytes=10,
                worker_parallel_min_available_ram_mb=100,
            )
            manager = TranslationManager(config)
            with mock.patch.object(manager, "_available_ram_mb", return_value=1000):
                self.assertEqual(manager._resolve_startup_mode(2), "parallel")
            self.assertEqual(
                manager._startup_strategy["reason"],
                "warm-cache-and-sufficient-ram",
            )

    def test_auto_falls_back_to_stagger_when_ram_is_low(self):
        with tempfile.TemporaryDirectory() as cache:
            Path(cache, "compiled.bin").write_bytes(b"x" * 32)
            config = make_config(
                jax_compilation_cache_dir=cache,
                worker_start_mode="auto",
                worker_parallel_cache_min_bytes=10,
                worker_parallel_min_available_ram_mb=1000,
            )
            manager = TranslationManager(config)
            with mock.patch.object(manager, "_available_ram_mb", return_value=500):
                self.assertEqual(manager._resolve_startup_mode(2), "stagger")
            self.assertIn("low-available-ram", manager._startup_strategy["reason"])

    def test_health_reports_live_free_vram_and_startup_snapshot(self):
        config = make_config()
        manager = TranslationManager(config)
        manager._selected_gpus = [
            {"id": "0", "name": "Tesla T4", "memory_total_mb": 15360, "memory_free_mb": 14912}
        ]
        manager._target_worker_count = 0
        with mock.patch(
            "translategemma.workers.manager.gpu_metrics_by_id",
            return_value={
                "0": {"id": "0", "name": "Tesla T4", "memory_total_mb": 15360, "memory_free_mb": 287}
            },
        ):
            health = manager.health()
        gpu = health["detected_gpus"][0]
        self.assertEqual(gpu["memory_free_mb"], 287)
        self.assertEqual(gpu["startup_memory_free_mb"], 14912)





class ValidationTests(unittest.TestCase):
    def test_valid_payload(self):
        config = make_config()
        parsed = parse_translation_payload(
            {
                "text": " Hello ",
                "source_lang": "English",
                "target_lang": "Vietnamese",
                "max_new_tokens": 16,
            },
            config,
        )
        self.assertEqual(parsed["text"], "Hello")
        self.assertEqual(parsed["max_tokens"], 16)

    def test_rejects_oversized_text(self):
        config = make_config(max_input_chars=3)
        with self.assertRaises(ValidationError):
            parse_translation_payload({"text": "four"}, config)

    def test_rejects_bad_tokens(self):
        config = make_config()
        with self.assertRaises(ValidationError):
            parse_translation_payload({"text": "hello", "max_new_tokens": 999}, config)

    def test_rejects_non_object(self):
        with self.assertRaises(ValidationError):
            parse_translation_payload([], make_config())


class JobStoreTests(unittest.TestCase):
    def test_store_never_exceeds_limit_when_all_jobs_pending(self):
        store = JobStore(max_size=1, result_ttl_seconds=60)
        store.put(Job("job-1", "a", "en", "vi", 8))
        with self.assertRaises(StoreFullError):
            store.put(Job("job-2", "b", "en", "vi", 8))

    def test_completed_job_can_be_evicted(self):
        store = JobStore(max_size=1, result_ttl_seconds=60)
        first = Job("job-1", "a", "en", "vi", 8)
        store.put(first)
        store.mark_completed("job-1", "x", 0.1)
        store.put(Job("job-2", "b", "en", "vi", 8))
        self.assertIsNone(store.get("job-1"))
        self.assertIsNotNone(store.get("job-2"))

    def test_ttl_cleanup(self):
        store = JobStore(max_size=5, result_ttl_seconds=0.01)
        store.put(Job("job-1", "a", "en", "vi", 8))
        store.mark_completed("job-1", "x", 0.1)
        time.sleep(0.02)
        self.assertIsNone(store.get("job-1"))

    def test_completed_job_releases_image_reference(self):
        store = JobStore(max_size=5, result_ttl_seconds=60)
        job = Job("job-image", "", "en", "vi", 8, image=object())
        store.put(job)
        store.mark_completed("job-image", "x", 0.1)
        self.assertIsNone(store.get("job-image").image)


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this test environment")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.manager = FakeManager(self.config)
        self.runtime = Runtime(config=self.config, manager=self.manager)
        self.client = create_app(self.runtime).test_client()
        self.headers = {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        }

    def test_authentication_required(self):
        response = self.client.post("/translate", json={"text": "hello"})
        self.assertEqual(response.status_code, 401)

    def test_health_summary_hides_worker_details(self):
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertNotIn("workers", body)
        self.assertNotIn("detected_gpus", body)

    def test_health_details_require_api_key(self):
        response = self.client.get("/health/ready?details=1")
        self.assertEqual(response.status_code, 401)
        response = self.client.get(
            "/health/ready?details=1",
            headers={"Authorization": "Bearer test-api-key"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("workers", response.get_json())

    def test_sync_timeout_returns_202(self):
        response = self.client.post(
            "/translate",
            json={"text": "hello"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["status"], "processing")
        self.assertIn("result_url", response.get_json())

    def test_completed_sync_result(self):
        job = Job("job-done", "hello", "English", "Vietnamese", 16)
        job.status = "completed"
        job.result = "xin chào"
        job.completed_at = time.time()
        job.done.set()
        self.manager.next_job = job
        response = self.client.post(
            "/translate",
            json={"text": "hello", "max_new_tokens": 16},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["translation"], "xin chào")

    def test_invalid_json_shape(self):
        response = self.client.post(
            "/translate",
            json=["not", "an", "object"],
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this test environment")
class NotReadyApiTests(unittest.TestCase):
    class NotReadyManager:
        def __init__(self, config):
            self.config = config

        def health(self):
            return {
                "state": "loading",
                "ready": False,
                "ready_workers": 0,
                "expected_workers": 2,
                "accepting_jobs": True,
                "jobs": {
                    "total": 0,
                    "queued": 0,
                    "processing": 0,
                    "completed": 0,
                    "failed": 0,
                },
                "workers": [],
                "detected_gpus": [],
            }

        def submit(self, payload):
            raise WorkerNotReadyError("No GPU worker is ready yet", health=self.health())

    def setUp(self):
        self.config = make_config()
        self.manager = self.NotReadyManager(self.config)
        self.runtime = Runtime(config=self.config, manager=self.manager)
        self.client = create_app(self.runtime).test_client()
        self.headers = {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        }

    def test_sync_returns_friendly_503(self):
        response = self.client.post(
            "/translate",
            json={"text": "hello"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("Retry-After"), "30")
        body = response.get_json()
        self.assertEqual(body["ready_workers"], 0)
        self.assertEqual(body["expected_workers"], 2)
        self.assertEqual(body["health_url"], "/health/ready")
        self.assertIn("still loading", body["error"])

    def test_async_returns_friendly_503(self):
        response = self.client.post(
            "/translate/async",
            json={"text": "hello"},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("still loading", response.get_json()["error"])


def _png_base64():
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    return __import__("base64").b64encode(buffer.getvalue()).decode("ascii")


class ImageValidationTests(unittest.TestCase):
    def test_image_payload_parses(self):
        config = make_config(vision_enabled=True)
        parsed = parse_image_translation_payload(
            {
                "image": _png_base64(),
                "source_lang": "English",
                "target_lang": "Japanese",
                "max_new_tokens": 32,
            },
            config,
        )
        self.assertEqual(parsed["src"], "English")
        self.assertEqual(parsed["tgt"], "Japanese")
        self.assertEqual(parsed["max_tokens"], 32)
        self.assertEqual(parsed["image"].shape, (16, 16, 3))
        self.assertEqual(parsed["image"].dtype.name, "uint8")

    def test_image_rejected_when_disabled(self):
        config = make_config(vision_enabled=False)
        with self.assertRaises(ValidationError):
            parse_image_translation_payload({"image": _png_base64()}, config)

    def test_image_rejects_invalid_base64(self):
        config = make_config(vision_enabled=True)
        with self.assertRaises(ValidationError):
            parse_image_translation_payload({"image": "not@base64!!"}, config)

    def test_image_rejects_oversized(self):
        config = make_config(vision_enabled=True, max_image_bytes=64)
        with self.assertRaises(ValidationError):
            parse_image_translation_payload({"image": _png_base64()}, config)

    def test_image_rejects_non_image(self):
        import base64

        config = make_config(vision_enabled=True)
        junk = base64.b64encode(b"not an image").decode("ascii")
        with self.assertRaises(ValidationError):
            parse_image_translation_payload({"image": junk}, config)

    def test_image_rejects_over_pixel_limit(self):
        config = make_config(vision_enabled=True, max_image_pixels=100)
        with self.assertRaises(ValidationError) as ctx:
            parse_image_translation_payload({"image": _png_base64()}, config)
        self.assertIn("too large", str(ctx.exception))

    def test_image_rejects_over_dimension_limit(self):
        config = make_config(vision_enabled=True, max_image_width=8)
        with self.assertRaises(ValidationError) as ctx:
            parse_image_translation_payload({"image": _png_base64()}, config)
        self.assertIn("dimensions", str(ctx.exception))

    def test_image_pixel_limit_checked_before_decode(self):
        from PIL import Image

        encoded = _png_base64()
        config = make_config(vision_enabled=True, max_image_pixels=100)
        load_calls = []
        with mock.patch.object(
            Image.Image, "load", side_effect=lambda *a: load_calls.append(1)
        ):
            with self.assertRaises(ValidationError):
                parse_image_translation_payload({"image": encoded}, config)
        self.assertEqual(
            load_calls,
            [],
            "image.load() must not run before the pixel limit is validated",
        )

    def test_image_rejects_unsupported_format(self):
        from io import BytesIO
        from PIL import Image
        import base64 as b64

        buffer = BytesIO()
        Image.new("RGB", (16, 16), "white").save(buffer, format="ICO")
        encoded = b64.b64encode(buffer.getvalue()).decode("ascii")
        config = make_config(vision_enabled=True)
        with self.assertRaises(ValidationError) as ctx:
            parse_image_translation_payload({"image": encoded}, config)
        self.assertIn("Unsupported image format", str(ctx.exception))

    def test_image_rejects_pillow_decompression_bomb_warning(self):
        from PIL import Image

        encoded = _png_base64()
        config = make_config(vision_enabled=True)
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 200):
            with self.assertRaises(ValidationError) as ctx:
                parse_image_translation_payload({"image": encoded}, config)
        self.assertIn("decompression bomb", str(ctx.exception))

    def test_image_rejects_pillow_decompression_bomb_error(self):
        from PIL import Image

        encoded = _png_base64()
        config = make_config(vision_enabled=True)
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 64):
            with self.assertRaises(ValidationError) as ctx:
                parse_image_translation_payload({"image": encoded}, config)
        self.assertIn("decompression bomb", str(ctx.exception))


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this test environment")
class ImageApiTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config(vision_enabled=True)
        self.manager = FakeManager(self.config)
        self.runtime = Runtime(config=self.config, manager=self.manager)
        self.client = create_app(self.runtime).test_client()
        self.headers = {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        }

    def test_image_endpoint_when_disabled_returns_400(self):
        config = make_config(vision_enabled=False)
        runtime = Runtime(config=config, manager=FakeManager(config))
        client = create_app(runtime).test_client()
        response = client.post(
            "/translate/image",
            json={"image": _png_base64()},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 400)

    def test_image_endpoint_sync_completed(self):
        job = Job("job-img", "", "English", "Vietnamese", 32)
        job.image = None
        job.status = "completed"
        job.result = "xin chào"
        job.completed_at = time.time()
        job.done.set()
        self.manager.next_job = job
        response = self.client.post(
            "/translate/image",
            json={"image": _png_base64()},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["translation"], "xin chào")

    def test_image_async_returns_202(self):
        response = self.client.post(
            "/translate/image/async",
            json={"image": _png_base64()},
            headers=self.headers,
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()["status"], "queued")


if __name__ == "__main__":
    unittest.main()
