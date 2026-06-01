import hashlib
import json
import logging
import os
import shutil
from urllib.parse import urljoin

import requests
from django.conf import settings

from options.options import SysOptions
from utils.shortcuts import rand_str
from .dispatcher import ChooseJudgeServer

logger = logging.getLogger(__name__)

EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"


class SelfTestRunner:
    def __init__(self, code, language, input_data):
        self.code = code
        self.language = language
        self.input_data = input_data
        self.token = hashlib.sha256(
            SysOptions.judge_server_token.encode("utf-8")
        ).hexdigest()

    def run(self):
        sub_config = list(filter(
            lambda item: self.language == item["name"],
            SysOptions.languages
        ))
        if not sub_config:
            return {
                "success": False,
                "error": f"Unsupported language: {self.language}"
            }
        sub_config = sub_config[0]

        test_case_id = rand_str()
        test_case_dir = os.path.join(settings.TEST_CASE_DIR, test_case_id)
        os.mkdir(test_case_dir)
        os.chmod(test_case_dir, 0o710)

        try:
            input_bytes = self.input_data.encode("utf-8")
            with open(os.path.join(test_case_dir, "1.in"), "wb") as f:
                f.write(input_bytes)

            with open(os.path.join(test_case_dir, "1.out"), "wb") as f:
                f.write(b"")

            info = {
                "spj": False,
                "test_cases": {
                    "1": {
                        "stripped_output_md5": EMPTY_MD5,
                        "input_size": len(input_bytes),
                        "output_size": 0,
                        "input_name": "1.in",
                        "output_name": "1.out",
                    }
                }
            }
            with open(os.path.join(test_case_dir, "info"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4)

            for item in os.listdir(test_case_dir):
                os.chmod(os.path.join(test_case_dir, item), 0o640)

            data = {
                "language_config": sub_config["config"],
                "src": self.code,
                "max_cpu_time": 5000,
                "max_memory": 256 * 1024 * 1024,
                "test_case_id": test_case_id,
                "output": True,
                "spj_version": "",
                "spj_config": {},
                "spj_compile_config": "",
                "spj_src": "",
                "io_mode": {"io_mode": "Standard IO", "input": "1.in", "output": "1.out"},
            }

            with ChooseJudgeServer() as server:
                if not server:
                    return {"success": False, "error": "No available judge server"}
                resp = self._request(
                    urljoin(server.service_url, "/judge"), data=data
                )

            if not resp:
                return {"success": False, "error": "Judge server not responding"}

            if resp.get("err"):
                return {
                    "success": False,
                    "error": resp.get("data", "Compile error")
                }

            test_result = resp["data"][0] if resp.get("data") else {}
            return {
                "success": True,
                "output": test_result.get("output", ""),
                "time_cost": test_result.get("cpu_time", 0),
                "memory_cost": test_result.get("memory", 0),
                "result": test_result.get("result", 0),
            }

        finally:
            shutil.rmtree(test_case_dir, ignore_errors=True)

    def _request(self, url, data=None):
        kwargs = {"headers": {"X-Judge-Server-Token": self.token}}
        if data:
            kwargs["json"] = data
        try:
            return requests.post(url, **kwargs).json()
        except Exception:
            logger.exception("Self-test request failed")
            return None
