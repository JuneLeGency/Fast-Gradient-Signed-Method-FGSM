import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import shutil
import os
import base64
from io import BytesIO
import asyncio
import ssl
import sys
import logging
import uuid
from typing import Optional

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# --- SSL 证书问题修复 ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- 标准包导入 ---
# 假设项目从根目录运行，可以直接从 'scripts' 包导入
from scripts.utils import get_all_classes_with_cn_names

# 导入我们的核心逻辑
import attack_core

# --- 临时文件目录配置 ---
TEMP_UPLOAD_DIR = "temp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# --- FastAPI 应用初始化 ---
app = FastAPI()

# --- 中间件配置 ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 辅助函数 ---
def pil_to_base64(img):
    """将 PIL Image 对象转换为 Base64 编码的字符串。"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- API & WebSocket 端点 ---

@app.get("/api")
def read_root():
    logging.info("根路径被访问。")
    return {"message": "AI 对抗攻击演示后端服务已启动"}

@app.post("/api/upload")
async def upload_for_attack(file: UploadFile = File(...)):
    """上传用于攻击的图片并返回一个唯一ID。"""
    try:
        # Generate a unique filename to avoid conflicts
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"文件 '{file.filename}' 已上传并保存为 '{unique_filename}'")
        return {"image_id": unique_filename}
    except Exception:
        logging.error("上传文件时发生错误。", exc_info=True)
        return {"error": "文件上传失败。"}


@app.post("/api/predict/")
async def predict_normal_image(file: UploadFile = File(...)):
    # Construct the temporary file path inside the dedicated directory
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"temp_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"正在对文件进行预测: {file.filename}")
        _, result_text = await run_in_threadpool(attack_core.predict_image, image_path=temp_file_path)
        
        if result_text and "错误" in result_text:
            logging.warning(f"预测过程中发生错误: {result_text}")
            return {"error": result_text}
        return {"predictions": result_text}
    except Exception:
        logging.error("对文件 %s 进行正常预测时发生错误。", file.filename, exc_info=True)
        return {"error": "服务器内部错误，请查看后端日志。"}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/attack/fgsm/")
async def perform_fgsm_attack(epsilon: float = Form(0.05), file: Optional[UploadFile] = File(None)):
    temp_file_path = None
    try:
        image_path_for_attack = None
        if file:
            # Save uploaded file temporarily
            temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"attack_{file.filename}")
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_path_for_attack = temp_file_path
            logging.info(f"正在使用上传的图片执行 FGSM 攻击: {file.filename}, Epsilon: {epsilon}")
        else:
            logging.info(f"正在使用默认图片执行 FGSM 攻击, Epsilon: {epsilon}")

        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = await run_in_threadpool(
            attack_core.generate_fgsm_attack, 
            epsilon=epsilon, 
            image_path=image_path_for_attack
        )
        return {
            "original_image": pil_to_base64(orig_pil),
            "perturbation_image": pil_to_base64(pert_pil),
            "adversarial_image": pil_to_base64(adv_pil),
            "original_text": orig_txt,
            "adversarial_text": adv_txt
        }
    except Exception:
        logging.error("执行 FGSM 攻击时发生错误。", exc_info=True)
        return {"error": "服务器内部错误，请查看后端日志。"}
    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/api/classes")
def get_classes():
    """返回所有可用的 ImageNet 类别列表。"""
    try:
        logging.info("正在获取 ImageNet 类别列表。")
        return get_all_classes_with_cn_names(attack_core.RESNET50_WEIGHTS)
    except Exception:
        logging.error("获取 ImageNet 类别列表时发生错误。", exc_info=True)
        return {"error": "无法获取类别列表，请查看后端日志。"}

@app.websocket("/api/attack/targeted_ws")
async def perform_targeted_attack_ws(websocket: WebSocket, target_class_id: int, image_id: Optional[str] = None):
    """通过 WebSocket 执行定向攻击，并实时发送进度和结果。"""
    await websocket.accept()
    loop = asyncio.get_running_loop()

    def progress_callback(progress):
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({"type": "progress", "value": progress}),
            loop
        )
    
    image_path_for_attack = None
    if image_id:
        image_path_for_attack = os.path.join(TEMP_UPLOAD_DIR, image_id)
        if not os.path.exists(image_path_for_attack):
            logging.error(f"请求的图片ID不存在: {image_id}")
            await websocket.send_json({"type": "error", "message": f"图片文件 '{image_id}' 未找到。"})
            await websocket.close()
            return
        logging.info(f"正在通过 WebSocket 使用图片 '{image_id}' 执行目标攻击，目标类别ID: {target_class_id}")
    else:
        logging.info(f"正在通过 WebSocket 使用默认图片执行目标攻击，目标类别ID: {target_class_id}")

    try:
        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = await run_in_threadpool(
            attack_core.generate_targeted_attack, 
            target_class_id=target_class_id, 
            progress_callback=progress_callback,
            image_path=image_path_for_attack
        )
        
        await websocket.send_json({
            "type": "result",
            "data": {
                "original_image": pil_to_base64(orig_pil),
                "perturbation_image": pil_to_base64(pert_pil),
                "adversarial_image": pil_to_base64(adv_pil),
                "original_text": orig_txt,
                "adversarial_text": adv_txt
            }
        })
    except Exception:
        logging.error("通过 WebSocket 执行目标攻击时发生错误。", exc_info=True)
        await websocket.send_json({"type": "error", "message": "服务器内部错误，请查看后端日志。"})
    finally:
        await websocket.close()
        # Note: We are not deleting the uploaded image here as it's part of a separate request.
        # A cleanup mechanism for old files in temp_uploads might be needed for a production system.

# --- 挂载静态文件 (必须在所有API路由之后) ---
app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")

# --- 启动服务 ---
if __name__ == "__main__":
    print("正在以开发模式启动服务器...")
    print("请在浏览器中打开 http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)