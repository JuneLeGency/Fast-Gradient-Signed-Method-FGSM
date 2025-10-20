import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import shutil
import os
import base64
from io import BytesIO

# 导入我们的核心逻辑
import attack_core

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

# --- API 端点 (Endpoints) ---

@app.get("/")
def read_root():
    return {"message": "AI 对抗攻击演示后端服务已启动"}

@app.post("/predict/")
async def predict_normal_image(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        _, result_text = await run_in_threadpool(attack_core.predict_image, image_path=temp_file_path)
        
        if "错误" in result_text:
            return {"error": result_text}
        return {"predictions": result_text}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/attack/fgsm/")
async def perform_fgsm_attack(epsilon: float = 0.05):
    try:
        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = await run_in_threadpool(attack_core.generate_fgsm_attack, epsilon=epsilon)
        return {
            "original_image": pil_to_base64(orig_pil),
            "perturbation_image": pil_to_base64(pert_pil),
            "adversarial_image": pil_to_base64(adv_pil),
            "original_text": orig_txt,
            "adversarial_text": adv_txt
        }
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/attack/targeted_ws")
async def perform_targeted_attack_ws(websocket: WebSocket):
    await websocket.accept()
    
    async def progress_callback(progress):
        await websocket.send_json({"type": "progress", "value": progress})

    try:
        # 在一个单独的线程中运行耗时的攻击函数
        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = await run_in_threadpool(
            attack_core.generate_targeted_attack, progress_callback=progress_callback
        )
        
        # 发送最终结果
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
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()

# --- 启动服务 ---
if __name__ == "__main__":
    print("后端服务代码已更新。")
    print("请在终端中，进入 backend 目录，然后手动运行以下命令来启动服务：")
    print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
