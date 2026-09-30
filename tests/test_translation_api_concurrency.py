import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.services import translate_api


class TranslationApiConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_sentences_from_multiple_images_share_one_api_limit(self):
        active = peak = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def create(**kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
            await release.wait()
            active -= 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="译文"))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(translate_api, "TRANSLATE_API_SEMAPHORE", asyncio.Semaphore(2)), \
                patch.object(translate_api, "_provider_options", return_value=(client, "test", {})):
            tasks = [asyncio.create_task(translate_api.translate_req(["一", "二", "三"])) for _ in range(2)]
            try:
                await asyncio.wait_for(started.wait(), 2)
                self.assertEqual(peak, 2)
            finally:
                release.set()
            results = await asyncio.gather(*tasks)
            self.assertEqual(peak, 2)
            self.assertTrue(all(result[0] == ["译文"] * 3 for result in results))


if __name__ == "__main__":
    unittest.main()
