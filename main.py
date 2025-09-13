import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent

@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "一个强大的图片风格化插件",
    "v1.1.0",
class FigurineProPlugin(Star):
    
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url: logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url
        async def _download_image(self, url: str) -> bytes | None:
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as e: logger.error(f"图片下载失败: {e}"); return None
        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit(): logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。"); return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)
        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        logger.info("检测到动图, 将抽取第一帧进行生成")
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception as e: logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw
        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file(): raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"): raw = await self._download_image(src)
            elif src.startswith("base64://"): raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw: return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)
        async def get_first_image(self, event: AstrMessageEvent) -> bytes | None:
            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)): return img
                            if s_chain.file and (img := await self._load_bytes(s_chain.file)): return img
            at_user_id = None
            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url and (img := await self._load_bytes(seg.url)): return img
                    if seg.file and (img := await self._load_bytes(seg.file)): return img
                elif isinstance(seg, At): at_user_id = str(seg.qq)
            if at_user_id: return await self._get_avatar(at_user_id)
            return await self._get_avatar(event.get_sender_id())
        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig): 
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None
        self.default_prompts: Dict[str, str] = {}

    async def initialize(self):
        prompts_file = Path(__file__).parent / "prompts.json"
        if prompts_file.exists():
            try:
                content = prompts_file.read_text("utf-8")
                self.default_prompts = json.loads(content)
                logger.info("默认 prompts.json 文件已加载")
            except Exception as e:
                logger.error(f"加载默认 prompts.json 文件失败: {e}", exc_info=True)
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_user_counts()
        await self._load_group_counts()
        logger.info("FigurinePro 插件已加载")
        if not self.conf.get("api_keys"):
            logger.warning("FigurinePro: 未配置任何 API 密钥，插件可能无法工作")

    # --- 【修改】仅全局管理员拥有最高权限 ---
    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        """检查用户是否为机器人的全局管理员"""
        admin_ids = self.context.get_config().get("admins_id", [])
        return event.get_sender_id() in admin_ids
    
    # --- 数据读写辅助函数 ---
    async def _load_user_counts(self):
        if not self.user_counts_file.exists(): self.user_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_counts = {str(k): v for k, v in data.items()}
        except Exception as e: logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True); self.user_counts = {}

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e: logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        count = self._get_user_count(user_id_str)
        if count > 0: self.user_counts[user_id_str] = count - 1; await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists(): self.group_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.group_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.group_counts = {str(k): v for k, v in data.items()}
        except Exception as e: logger.error(f"加载群组次数文件时发生错误: {e}", exc_info=True); self.group_counts = {}

    async def _save_group_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.group_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.group_counts_file.write_text, json_data, "utf-8")
        except Exception as e: logger.error(f"保存群组次数文件时发生错误: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        group_id_str = str(group_id)
        count = self._get_group_count(group_id_str)
        if count > 0: self.group_counts[group_id_str] = count - 1; await self._save_group_counts()

    # --- 管理指令 (仅限全局管理员) ---
    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return # 权限检查
        cmd_text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target_qq, count = None, 0
        if at_seg:
            target_qq = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", cmd_text)
            if match: count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", cmd_text)
            if match: target_qq, count = match.group(1), int(match.group(2))
        if not target_qq or count <= 0:
            yield event.plain_result('格式错误:\n#手办化增加用户次数 @用户 <次数>\n或 #手办化增加用户次数 <QQ号> <次数>')
            return
        current_count = self._get_user_count(target_qq)
        self.user_counts[str(target_qq)] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return # 权限检查
        cmd_text = event.message_str.strip()
        match = re.search(r"(\d+)\s+(\d+)", cmd_text)
        if not match:
            yield event.plain_result('格式错误: #手办化增加群组次数 <群号> <次数>')
            return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[str(target_group)] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id_to_query = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg: user_id_to_query = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match: user_id_to_query = match.group(1)
        
        user_count = self._get_user_count(user_id_to_query)
        reply_msg = ""
        if user_id_to_query == event.get_sender_id():
            reply_msg = f"您好，您当前个人剩余次数为: {user_count}"
        else:
            reply_msg = f"用户 {user_id_to_query} 个人剩余次数为: {user_count}"
        
        group_id = event.get_group_id()
        if group_id:
            group_count = self._get_group_count(group_id)
            reply_msg += f"\n本群共享剩余次数为: {group_count}"
        yield event.plain_result(reply_msg)


    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return # 权限检查
        new_keys = event.message_str.strip().split()
        if not new_keys: yield event.plain_result("格式错误，请提供要添加的Key。"); return
        api_keys = self.conf.get("api_keys", [])
        added_keys = [key for key in new_keys if key not in api_keys]
        api_keys.extend(added_keys)
        await self.conf.set("api_keys", api_keys)
        yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return # 权限检查
        api_keys = self.conf.get("api_keys", [])
        if not api_keys: yield event.plain_result("📝 暂未配置任何 API Key。"); return
        key_list_str = "\n".join(f"{i+1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
        yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return # 权限检查
        param = event.message_str.strip()
        api_keys = self.conf.get("api_keys", [])
        if param.lower() == "all":
            count = len(api_keys)
            await self.conf.set("api_keys", [])
            yield event.plain_result(f"✅ 已删除全部 {count} 个 Key。")
        elif param.isdigit() and 1 <= int(param) <= len(api_keys):
            idx = int(param) - 1
            removed_key = api_keys.pop(idx)
            await self.conf.set("api_keys", api_keys)
            yield event.plain_result(f"✅ 已删除 Key: {removed_key[:8]}...")
        else:
            yield event.plain_result("格式错误，请使用 #手办化删除key <序号|all>")
    
    # --- 图片生成指令 (一个指令一个函数) ---
    @filter.command("手办化", prefix_optional=True)
    async def on_cmd_figurine(self, event: AstrMessageEvent):
        cmd = "手办化"
        async for result in self._process_figurine_request(event, cmd): yield result
    
    @filter.command("手办化2", prefix_optional=True)
    async def on_cmd_figurine2(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "手办化2"): yield result
        
    @filter.command("手办化3", prefix_optional=True)
    async def on_cmd_figurine3(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "手办化3"): yield result

    @filter.command("手办化4", prefix_optional=True)
    async def on_cmd_figurine4(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "手办化4"): yield result
        
    @filter.command("手办化5", prefix_optional=True)
    async def on_cmd_figurine5(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "手办化5"): yield result

    @filter.command("手办化6", prefix_optional=True)
    async def on_cmd_figurine6(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "手办化6"): yield result
        
    @filter.command("Q版化", prefix_optional=True)
    async def on_cmd_qversion(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "Q版化"): yield result

    @filter.command("痛屋化", prefix_optional=True)
    async def on_cmd_painroom(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "痛屋化"): yield result
        
    @filter.command("痛屋化2", prefix_optional=True)
    async def on_cmd_painroom2(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "痛屋化2"): yield result

    @filter.command("痛车化", prefix_optional=True)
    async def on_cmd_paincar(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "痛车化"): yield result
        
    @filter.command("cos化", prefix_optional=True)
    async def on_cmd_cos(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "cos化"): yield result
        
    @filter.command("cos自拍", prefix_optional=True)
    async def on_cmd_cos_selfie(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "cos自拍"): yield result
        
    @filter.command("bnn", prefix_optional=True)
    async def on_cmd_bnn(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "bnn"): yield result
        
    @filter.command("孤独的我", prefix_optional=True)
    async def on_cmd_clown(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "孤独的我"): yield result
        
    @filter.command("第三视角", prefix_optional=True)
    async def on_cmd_view3(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "第三视角"): yield result
        
    @filter.command("鬼图", prefix_optional=True)
    async def on_cmd_ghost(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "鬼图"): yield result

    @filter.command("第一视角", prefix_optional=True)
    async def on_cmd_view1(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "第一视角"): yield result
        
    @filter.command("贴纸化", prefix_optional=True)
    async def on_cmd_sticker(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "贴纸化"): yield result

    @filter.command("玉足", prefix_optional=True)
    async def on_cmd_foot_jade(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "玉足"): yield result
        
    @filter.command("fumo化", prefix_optional=True)
    async def on_cmd_fumo(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "fumo化"): yield result

    @filter.command("手办化帮助", prefix_optional=True)
    async def on_cmd_help(self, event: AstrMessageEvent):
        async for result in self._process_figurine_request(event, "手办化帮助"): yield result

    # --- 核心图片生成处理器 ---
    async def _process_figurine_request(self, event: AstrMessageEvent, cmd: str):
        cmd_text = event.message_str

        cmd_map = { "手办化": "figurine_1", "手办化2": "figurine_2", "手办化3": "figurine_3", "手办化4": "figurine_4", "手办化5": "figurine_5", "手办化6": "figurine_6", "Q版化": "q_version", "痛屋化": "pain_room_1", "痛屋化2": "pain_room_2", "痛车化": "pain_car", "cos化": "cos", "cos自拍": "cos_selfie", "孤独的我": "clown", "第三视角": "view_3", "鬼图": "ghost", "第一视角": "view_1", "贴纸化": "sticker", "玉足": "foot_jade", "手办化帮助": "help", "fumo化": "fumo" }
        prompt_key = cmd_map.get(cmd) if cmd != "bnn" else "bnn_custom"
        
        user_prompt = None
        if cmd == "bnn":
            user_prompt = cmd_text.strip()
        elif prompt_key == "help":
            user_prompt = self.conf.get("help_text", "帮助信息未配置")
        else:
            user_prompts = self.conf.get("prompts", {})
            user_prompt = user_prompts.get(prompt_key) or self.default_prompts.get(prompt_key, "")
        
        if cmd == "手办化帮助":
            yield event.plain_result(user_prompt)
            return
        if not user_prompt and cmd != "bnn":
            yield event.plain_result(f"❌ 预设 '{cmd}' 未在配置中找到或prompt为空。")
            return
        if cmd == "bnn" and not user_prompt:
            yield event.plain_result("❌ 命令格式错误: bnn <提示词> [图片]")
            return
            
        # --- 【最终版】权限和次数检查逻辑 ---
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)

        if not is_master:
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0

            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0

            if group_id: # 在群聊中，群组或个人有次数即可
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽，请联系管理员补充。")
                    return
            else: # 在私聊中，仅判断个人次数
                if not has_user_count:
                    yield event.plain_result("❌ 您的使用次数已用完，请联系管理员补充。")
                    return

        if not self.iwf or not (img_bytes := await self.iwf.get_first_image(event)):
            yield event.plain_result("请发送或引用一张图片，或@一个用户再试。")
            return
        
        yield event.plain_result(f"🎨 收到请求，正在生成 [{cmd}] 风格图片...")
        start_time = datetime.now()
        res = await self._call_api(img_bytes, user_prompt)
        elapsed = (datetime.now() - start_time).total_seconds()

        if isinstance(res, bytes):
            # --- 【最终版】次数扣除逻辑：优先扣群组，再扣个人 ---
            if not is_master:
                # 优先扣除群组次数
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                # 群组次数用完后，扣除个人次数
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)

            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"预设: {cmd}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")

    # --- API 调用辅助函数 ---
    async def _get_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        try: return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except (IndexError, TypeError, KeyError): pass
        try: return data["choices"][0]["message"]["images"][0]["url"]
        except (IndexError, TypeError, KeyError): pass
        try:
            content_text = data["choices"][0]["message"]["content"]
            url_match = re.search(r'https?://[^\s<>")\]]+', content_text)
            if url_match: return url_match.group(0).rstrip(")>,'\"")
        except (IndexError, TypeError, KeyError): pass
        return None

    async def _call_api(self, image_bytes: bytes, prompt: str) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url: return "API URL 未配置"
        api_key = await self._get_api_key()
        if not api_key: return "无可用的 API Key"
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = { "model": "nano-banana", "max_tokens": 1500, "stream": False, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}]}
        try:
            if not self.iwf: return "ImageWorkflow 未初始化"
            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=120) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_text}")
                    return f"API请求失败 (HTTP {resp.status}): {error_text[:200]}"
                data = await resp.json()
                if "error" in data: return data["error"].get("message", json.dumps(data["error"]))
                gen_image_url = self._extract_image_url_from_response(data)
                if not gen_image_url:
                    error_msg = f"API响应中未找到图片数据。原始响应 (部分): {str(data)[:500]}..."
                    logger.error(f"API响应中未找到图片数据: {data}")
                    return error_msg
                if gen_image_url.startswith("data:image/"):
                    b64_data = gen_image_url.split(",", 1)[1]
                    return base64.b64decode(b64_data)
                else:
                    return await self.iwf._download_image(gen_image_url) or "下载生成的图片失败"
        except asyncio.TimeoutError: logger.error("API 请求超时"); return "请求超时"
        except Exception as e: logger.error(f"调用 API 时发生未知错误: {e}", exc_info=True); return f"发生未知错误: {e}"

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")


