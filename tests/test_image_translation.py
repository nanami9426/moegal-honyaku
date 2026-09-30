import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi import BackgroundTasks

from app.core.custom_conf import custom_conf
from app.services import image_translation as pipeline
from app.api.routes.manga_translate import _translate_image_bytes


class ImageTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        pipeline._cache.clear()
        pipeline._inflight.clear()
        pipeline._cache_bytes = 0
        self.image = np.full((80, 100, 3), 255, np.uint8)
        self.data = cv2.imencode(".png", self.image)[1].tobytes()
        self.patches = [
            patch.object(pipeline, "IMAGE_JOBS", asyncio.Semaphore(3)),
            patch.object(pipeline, "MODEL_JOBS", asyncio.Semaphore(1)),
            patch.object(pipeline, "IMAGE_WORKERS", asyncio.Semaphore(2)),
            patch("app.services.ocr.detect_text_bubbles", return_value=np.array([[20, 20, 60, 60]])),
            patch("app.services.pic_process.recognize_text_regions", new_callable=AsyncMock, return_value=["原文"]),
            patch("app.services.text_erasure.erase_text_regions", return_value=(self.image, np.zeros((80, 100), np.uint8))),
            patch.object(pipeline, "translate_req", new_callable=AsyncMock, return_value=(["译文"], 0)),
        ]
        values = [item.start() for item in self.patches]
        self.detect, self.ocr, self.erase, self.translate = values[3:]
        self.original_mode = custom_conf.translate_mode

    async def asyncTearDown(self):
        custom_conf.translate_mode = self.original_mode
        if pipeline._inflight:
            await asyncio.gather(*pipeline._inflight.values(), return_exceptions=True)
        for item in reversed(self.patches):
            item.stop()

    async def test_no_bubbles_skips_expensive_work_and_has_explicit_api_code(self):
        self.detect.return_value = np.empty((0, 4))
        result = await _translate_image_bytes(self.data, True, BackgroundTasks())
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["code"], "NO_TEXT_BUBBLES")
        self.ocr.assert_not_called()
        self.erase.assert_not_called()
        self.translate.assert_not_called()
        cached = await pipeline.translate_image(self.data, "horizontal")
        self.assertTrue(cached["cache_hit"])
        self.detect.assert_called_once()
        await pipeline.translate_image(self.data, "horizontal", force_refresh=True)
        self.assertEqual(self.detect.call_count, 2)

    async def test_ocr_and_translation_proceed_while_erasure_is_running(self):
        started, release = threading.Event(), threading.Event()

        def erase(*_):
            started.set()
            if not release.wait(3):
                raise TimeoutError("翻译未与擦除并行")
            return self.image, np.zeros((80, 100), np.uint8)

        async def translate(*_, **__):
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            release.set()
            return ["译文"], 0

        self.erase.side_effect = erase
        self.translate.side_effect = translate
        result = await asyncio.wait_for(pipeline.translate_image(self.data, "horizontal"), 5)
        self.assertGreater(len(result["image_bytes"]), 0)
        self.assertTrue({"detect", "ocr", "erase", "translate", "render"} <= result["timings"].keys())

    async def test_next_image_uses_model_while_first_image_waits_for_translation(self):
        both = asyncio.Event()
        calls = 0

        async def translate(*_, **__):
            nonlocal calls
            calls += 1
            if calls == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 2)
            return ["译文"], 0

        self.translate.side_effect = translate
        other = cv2.imencode(".png", self.image - 1)[1].tobytes()
        await asyncio.gather(pipeline.translate_image(self.data, "horizontal"),
                             pipeline.translate_image(other, "horizontal"))
        self.assertEqual(calls, 2)

    async def test_duplicates_share_work_and_cache_is_separated_by_settings(self):
        one, two = await asyncio.gather(*(pipeline.translate_image(self.data, "horizontal") for _ in range(2)))
        self.detect.assert_called_once()
        self.assertEqual(one["image_bytes"], two["image_bytes"])
        self.assertTrue(two["coalesced"])
        cached = await pipeline.translate_image(self.data, "horizontal")
        self.assertTrue(cached["cache_hit"])
        await pipeline.translate_image(self.data, "vertical")
        custom_conf.translate_mode = "structured"
        await pipeline.translate_image(self.data, "horizontal")
        self.assertEqual(self.translate.await_count, 3)

    async def test_failed_request_is_not_cached_and_pending_slot_is_released(self):
        self.translate.side_effect = RuntimeError("临时错误")
        with self.assertRaisesRegex(RuntimeError, "临时错误"):
            await pipeline.translate_image(self.data, "horizontal")
        self.assertFalse(pipeline._inflight)
        self.assertFalse(pipeline._cache)
        self.translate.side_effect = None
        await pipeline.translate_image(self.data, "horizontal")
        self.assertEqual(self.translate.await_count, 2)

    async def test_cancelled_waiter_does_not_cancel_shared_work(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        async def translate(*_, **__):
            entered.set()
            await finish.wait()
            return ["译文"], 0

        self.translate.side_effect = translate
        first = asyncio.create_task(pipeline.translate_image(self.data, "horizontal"))
        await entered.wait()
        second = asyncio.create_task(pipeline.translate_image(self.data, "horizontal"))
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        finish.set()
        self.assertTrue((await second)["image_bytes"])
        self.translate.assert_awaited_once()

    async def test_cache_is_bounded_and_expired_entries_are_recomputed(self):
        with patch.object(pipeline, "CACHE_MAX_ITEMS", 1):
            await pipeline.translate_image(self.data, "horizontal")
            await pipeline.translate_image(self.data, "vertical")
            self.assertEqual(len(pipeline._cache), 1)
            await pipeline.translate_image(self.data, "horizontal")
            self.assertEqual(self.detect.call_count, 3)
        key, (_, size, result) = next(iter(pipeline._cache.items()))
        pipeline._cache[key] = (0, size, result)
        await pipeline.translate_image(self.data, "horizontal")
        self.assertEqual(self.detect.call_count, 4)

    async def test_queue_overload_rejects_new_work(self):
        with patch.object(pipeline, "MAX_PENDING", 0):
            with self.assertRaises(pipeline.TranslationBusyError):
                await pipeline.translate_image(self.data, "horizontal")
        self.detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
