# --- START OF MODIFIED main.py ---

import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import aiohttp
from PIL import Image as PILImage

import astrbot.core.message.components as Comp
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image, At, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


# --- 图像处理工作流 ---
class ImageWorkflow:
    def __init__(self, proxy_url: str | None = None):
        if proxy_url:
            logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
        # 【修复】让 aiohttp.ClientSession 自己管理 connector 的生命周期，这是最简单和安全的方式，避免资源泄漏。
        self.session = aiohttp.ClientSession()
        self.proxy = proxy_url

    async def _download_image(self, url: str) -> bytes | None:
        try:
            async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def _get_avatar(self, user_id: str) -> bytes | None:
        # 【修复】对于非 QQ 平台或无效 ID，返回 None 而不是一个随机 QQ 头像，避免产生误导性结果。
        if not user_id.isdigit():
            logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。")
            return None
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
        except Exception as e:
            # 【修复】增加异常日志记录
            logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw
        return raw

    async def _load_bytes(self, src: str) -> bytes | None:
        raw: bytes | None = None
        loop = asyncio.get_running_loop()

        if Path(src).is_file():
            raw = await loop.run_in_executor(None, Path(src).read_bytes)
        elif src.startswith("http"):
            raw = await self._download_image(src)
        elif src.startswith("base64://"):
            raw = await loop.run_in_executor(None, base64.b64decode, src[9:])

        if not raw:
            return None
        return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

    async def get_first_image(self, event: AstrMessageEvent) -> bytes | None:
        # 优先处理回复中的图片
        for seg in event.message_obj.message:
            if isinstance(seg, Reply) and seg.chain:
                for s_chain in seg.chain:
                    if isinstance(s_chain, Image):
                        if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                            return img
                        if s_chain.file and (img := await self._load_bytes(s_chain.file)):
                            return img
        
        # 处理当前消息中的图片和@
        at_user_id = None
        for seg in event.message_obj.message:
            if isinstance(seg, Image):
                if seg.url and (img := await self._load_bytes(seg.url)):
                    return img
                if seg.file and (img := await self._load_bytes(seg.file)):
                    return img
            elif isinstance(seg, At):
                at_user_id = str(seg.qq)
        
        # 【修复】如果 @ 了用户，则优先使用其头像。如果获取失败，则返回 None，
        # 而不是错误地回退到发送者头像，以避免逻辑混乱。
        if at_user_id:
            return await self._get_avatar(at_user_id)
            
        # 兜底使用发送者头像
        return await self._get_avatar(event.get_sender_id())

    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()


# 【修复】移除在新版框架中不再需要的 @register 装饰器
class FigurineProPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir("astrbot_plugin_figurine_pro")
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        
        self.iwf: ImageWorkflow

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = ImageWorkflow(proxy_url)
        await self._load_user_counts()

        logger.info("FigurinePro 插件已加载")
        if not self.conf.get("api_keys"):
            logger.warning("FigurinePro: 未配置任何 API 密钥，插件可能无法工作")

    async def _load_user_counts(self):
        if not self.user_counts_file.exists():
            self.user_counts = {}
            return
        
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict):
                self.user_counts = {str(k): v for k, v in data.items()}
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"加载用户次数文件时发生错误，文件可能已损坏: {e}", exc_info=True)
            self.user_counts = {}

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        count = self._get_user_count(user_id_str)
        if count > 0:
            self.user_counts[user_id_str] = count - 1
            await self._save_user_counts()
    
    # --- 指令处理 ---
    @filter.regex(r"^#手办化(增加|查询)次数", is_admin=True)
    async def on_manage_counts(self, event: AstrMessageEvent):
        cmd_text = event.message_obj.message_str.strip()
        
        if "增加次数" in cmd_text:
            target_qq, count = None, 0
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            
            if at_seg: # 模式1: @用户 + 次数 (例如: #手办化增加次数 @张三 10)
                target_qq = str(at_seg.qq)
                match = re.search(r"(\d+)\s*$", cmd_text)
                if match:
                    count = int(match.group(1))
            else: # 模式2: QQ号 + 次数 (例如: #手办化增加次数 12345 10)
                # 【修复】使用更严格的正则确保格式正确性
                match = re.search(r"增加次数\s+(\d+)\s+(\d+)\s*$", cmd_text)
                if match:
                    target_qq = match.group(1)
                    count = int(match.group(2))

            if not target_qq or count <= 0:
                yield event.plain_result('格式错误，请使用：#手办化增加次数 @用户 <次数> 或 #手办化增加次数 <QQ号> <次数>')
                return
            
            current_count = self._get_user_count(target_qq)
            self.user_counts[str(target_qq)] = current_count + count
            await self._save_user_counts()
            yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，当前剩余：{current_count + count} 次")

        elif "查询次数" in cmd_text:
            target_qq = None
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)

            if at_seg:
                target_qq = str(at_seg.qq)
            else:
                 # 【修复】使用更严格的正则提取QQ号
                match = re.search(r"查询次数\s+(\d+)", cmd_text)
                if match:
                    target_qq = match.group(1)
            
            if not target_qq: # 如果未指定用户，则查询发送者自己
                target_qq = event.get_sender_id()
            
            count = self._get_user_count(target_qq)
            yield event.plain_result(f"用户 {target_qq} 剩余次数: {count}")

    @filter.regex(r"^#手办化查询次数$")
    async def on_query_my_counts(self, event: AstrMessageEvent):
        count = self._get_user_count(event.get_sender_id())
        yield event.plain_result(f"您好，您当前剩余次数为: {count}")

    @filter.regex(r"^#手办化(添加key|key列表|删除key)", is_admin=True)
    async def on_manage_keys(self, event: AstrMessageEvent):
        cmd_text = event.message_obj.message_str
        api_keys = self.conf.get("api_keys", [])

        if "添加key" in cmd_text:
            new_keys = cmd_text.replace("#手办化添加key", "").strip().split()
            if not new_keys:
                yield event.plain_result("格式错误，请提供要添加的Key。")
                return
            
            added_keys = [key for key in new_keys if key not in api_keys]
            api_keys.extend(added_keys)
            await self.conf.set("api_keys", api_keys)
            yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

        elif "key列表" in cmd_text:
            if not api_keys:
                yield event.plain_result("📝 暂未配置任何 API Key。")
                return
            
            key_list_str = "\n".join(f"{i+1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
            yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

        elif "删除key" in cmd_text:
            param = cmd_text.replace("#手办化删除key", "").strip()
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

    def _parse_figurine_command(self, event: AstrMessageEvent) -> Tuple[str | None, str | None]:
        """【新增】辅助函数，用于解析指令和获取对应的 prompt"""
        cmd_match = re.match(r"^#?([\w\d]+)", event.message_obj.message_str)
        if not cmd_match:
            return None, None
        cmd = cmd_match.group(1)

        cmd_map = {
            "手办化": "figurine_1", "手办化2": "figurine_2", "手办化3": "figurine_3", "手办化4": "figurine_4",
            "手办化5": "figurine_5", "手办化6": "figurine_6", "Q版化": "q_version", "痛屋化": "pain_room_1",
            "痛屋化2": "pain_room_2", "痛车化": "pain_car", "cos化": "cos", "cos自拍": "cos_selfie",
            "孤独的我": "clown", "第三视角": "view_3", "鬼图": "ghost", "第一视角": "view_1", "手办化帮助": "help"
        }
        
        prompt_key = cmd_map.get(cmd) if cmd != "bnn" else "bnn_custom"
        if not prompt_key and cmd != "手办化帮助": # `手办化帮助` a special case handled in caller
            return cmd, None
        
        user_prompt = None
        if cmd == "bnn":
            user_prompt = re.sub(r"^#?bnn\s*", "", event.message_obj.message_str, count=1).strip()
        elif prompt_key == "help":
            user_prompt = self.conf.get("help_text", "帮助信息未配置")
        elif prompt_key:
            user_prompt = self.conf.get("prompts", {}).get(prompt_key, "")

        return cmd, user_prompt

    @filter.regex(r"^#?(手办化[2-6]?|Q版化|痛屋化2?|痛车化|cos化|cos自拍|bnn|孤独的我|第三视角|鬼图|第一视角|手办化帮助)")
    async def on_figurine(self, event: AstrMessageEvent):
        # 【重构】将指令解析和 prompt 获取逻辑移至辅助函数，使主函数更清晰。
        cmd, user_prompt = self._parse_figurine_command(event)

        if not cmd:
            return

        if cmd == "手办化帮助":
            yield event.plain_result(user_prompt)
            return

        if not user_prompt:
            if cmd == "bnn":
                yield event.plain_result("❌ 命令格式错误，请使用：#bnn <提示词> [图片]")
            elif self.conf.get("prompts", {}).get(cmd) is None: # prompt key not found
                 yield event.plain_result(f"❌ 预设 '{cmd}' 未在配置中找到，请检查插件配置。")
            else: # Known command, but prompt is empty in config
                 yield event.plain_result(f"未知的指令: {cmd}")
            return
            
        if not event.is_admin() and self._get_user_count(event.get_sender_id()) <= 0:
            yield event.plain_result("❌ 您的使用次数已用完，请联系管理员补充。")
            return
        
        img_bytes = await self.iwf.get_first_image(event)
        if not img_bytes:
            yield event.plain_result("请发送或引用一张图片，或@一个用户再试。")
            return

        yield event.plain_result(f"🎨 收到请求，正在生成 [{cmd}] 风格图片...")
        start_time = datetime.now()

        res = await self._call_api(img_bytes, user_prompt)

        elapsed = (datetime.now() - start_time).total_seconds()
        if isinstance(res, bytes):
            if not event.is_admin():
                await self._decrease_user_count(event.get_sender_id())
            
            remaining_count = "∞" if event.is_admin() else self._get_user_count(event.get_sender_id())
            caption = f"✅ 生成成功 ({elapsed:.2f}s)\n预设: {cmd} | 剩余次数: {remaining_count}"
            yield event.chain_result([Image.fromBytes(res), Plain(caption)])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")

    async def _get_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        """
        【新增】辅助函数，用于从复杂的 API 响应中提取图片 URL。
        由于第三方 API 的响应结构可能不统一或发生变更，这里采用多种方式尝试提取，以提高插件的健壮性。
        """
        try:
            # 方式1: 尝试从标准路径 `choices[0].message.images[0].image_url.url` 获取
            return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except (IndexError, TypeError, KeyError):
            pass # 如果失败，静默处理并尝试下一种方式

        try:
            # 方式2: 尝试备用路径 `choices[0].message.images[0].url`
            return data["choices"][0]["message"]["images"][0]["url"]
        except (IndexError, TypeError, KeyError):
            pass

        try:
            # 方式3: 如果直接路径查找失败，尝试从文本内容中用正则表达式匹配URL
            content_text = data["choices"][0]["message"]["content"]
            # 匹配 http/https 开头的 URL，并处理结尾可能存在的干扰字符
            url_match = re.search(r'https?://[^\s<>")\]]+', content_text)
            if url_match:
                return url_match.group(0).rstrip(")>,'\"")
        except (IndexError, TypeError, KeyError):
            pass

        return None

    async def _call_api(self, image_bytes: bytes, prompt: str) -> bytes | str:
        api_url = self.conf.get("api_url")
        if not api_url: return "API URL 未配置"
            
        api_key = await self._get_api_key()
        if not api_key: return "无可用的 API Key"

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "nano-banana", "max_tokens": 1500, "stream": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}]}

        try:
            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy, timeout=120) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_text}")
                    return f"API请求失败 (HTTP {resp.status}): {error_text[:200]}"
                
                data = await resp.json()
                if "error" in data:
                    return data["error"].get("message", json.dumps(data["error"]))
                
                # 【重构】调用新的辅助函数来解析响应
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

        except asyncio.TimeoutError:
            logger.error("API 请求超时")
            return "请求超时"
        except Exception as e:
            logger.error(f"调用 API 时发生未知错误: {e}", exc_info=True)
            return f"发生未知错误: {e}"

    async def terminate(self):
        if hasattr(self, 'iwf') and self.iwf:
            await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
