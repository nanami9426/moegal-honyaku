import base64
import unittest
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi import FastAPI

from app.api.routes.manga_translate import manga_translate_router
from app.services.image_translation import TranslationBusyError
from app.services.translate_api import MissingTranslateProviderConfigError


class TranslateRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        application = FastAPI()
        application.include_router(manga_translate_router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test")
        self.data = cv2.imencode(".png", np.full((20, 20, 3), 255, np.uint8))[1].tobytes()
        self.payload = {"image_base64": "data:image/png;base64," + base64.b64encode(self.data).decode(),
                        "referer": "https://manga.example", "force_refresh": True}

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_web_and_upload_return_explicit_skip_and_forward_manual_retry(self):
        result = {"code": "NO_TEXT_BUBBLES", "timings": {}, "cache_hit": False, "coalesced": False}
        with patch("app.api.routes.manga_translate.translate_image", return_value=result) as translate:
            web = await self.client.post("/api/v1/translate/web", json=self.payload)
            upload = await self.client.post("/api/v1/translate/upload?force_refresh=true",
                                            files={"img": ("image.png", self.data, "image/png")})
        for response in (web, upload):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "skipped")
            self.assertEqual(response.json()["code"], "NO_TEXT_BUBBLES")
            self.assertNotIn("res_img", response.json())
        self.assertTrue(all(call.kwargs["force_refresh"] for call in translate.await_args_list))

    async def test_success_keeps_existing_fields_and_optional_image_response(self):
        result = {"image_bytes": self.data, "raw_text": ["原文"], "cn_text": ["译文"], "price": 0,
                  "timings": {}, "cache_hit": True, "coalesced": False}
        with patch("app.api.routes.manga_translate.translate_image", return_value=result):
            response = await self.client.post("/api/v1/translate/web", json=self.payload)
            no_image = await self.client.post("/api/v1/translate/web", json={**self.payload, "include_res_img": False})
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["cn_text"], ["译文"])
        self.assertEqual(base64.b64decode(payload["res_img"]), self.data)
        self.assertTrue(payload["cache_hit"])
        self.assertIsNone(no_image.json()["res_img"])

    async def test_queue_and_configuration_errors_have_machine_readable_codes(self):
        for error, status, code in ((TranslationBusyError("忙碌"), 429, "QUEUE_FULL"),
                                    (MissingTranslateProviderConfigError("缺少配置"), 400, "MISSING_TRANSLATE_CONFIG")):
            with self.subTest(code=code), patch("app.api.routes.manga_translate.translate_image", side_effect=error):
                response = await self.client.post("/api/v1/translate/web", json=self.payload)
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["code"], code)


if __name__ == "__main__":
    unittest.main()
