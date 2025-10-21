import customtkinter
import tkinter
from tkinter import filedialog
from PIL import Image
import os
import sys
import threading

from scripts.utils import get_all_classes_with_cn_names

# 导入我们的核心逻辑
from backend import attack_core

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # ---- 窗口基础设置 ----
        self.title("AI 对抗攻击演示 (原生GUI版)")
        self.geometry("1100x750")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- 创建选项卡视图 ----
        self.tab_view = customtkinter.CTkTabview(self, width=250)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.tab_view.add("正常识别")
        self.tab_view.add("非定向攻击 (FGSM)")
        self.tab_view.add("定向攻击")

        # ---- 配置每个选项卡 ----
        self.setup_normal_prediction_tab()
        self.setup_fgsm_attack_tab()
        self.setup_targeted_attack_tab()

    def setup_normal_prediction_tab(self):
        tab = self.tab_view.tab("正常识别")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        self.select_image_button = customtkinter.CTkButton(tab, text="选择图片", command=self.select_and_predict_image)
        self.select_image_button.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.image_path_label = customtkinter.CTkLabel(tab, text="尚未选择图片", anchor="w")
        self.image_path_label.grid(row=0, column=1, padx=20, pady=(20, 10))

        self.image_display_label = customtkinter.CTkLabel(tab, text="")
        self.image_display_label.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        self.prediction_textbox = customtkinter.CTkTextbox(tab, height=120, wrap="none")
        self.prediction_textbox.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="ew")
        self.prediction_textbox.insert("0.0", "预测结果将显示在这里...")

    def setup_fgsm_attack_tab(self):
        tab = self.tab_view.tab("非定向攻击 (FGSM)")
        tab.grid_columnconfigure((0, 1, 2), weight=1)
        tab.grid_rowconfigure(2, weight=1)

        info_label = customtkinter.CTkLabel(tab, text="此功能将固定攻击 backend/images/panda.jpg 图片", anchor="center")
        info_label.grid(row=0, column=0, columnspan=3, padx=20, pady=(10,0))

        epsilon_frame = customtkinter.CTkFrame(tab)
        epsilon_frame.grid(row=1, column=0, padx=20, pady=10)
        epsilon_label = customtkinter.CTkLabel(epsilon_frame, text="扰动强度 (Epsilon): ")
        epsilon_label.pack(side="left", padx=(10,5), pady=10)
        self.epsilon_slider = customtkinter.CTkSlider(epsilon_frame, from_=0, to=0.2, number_of_steps=20)
        self.epsilon_slider.pack(side="left", padx=(5,10), pady=10)
        self.epsilon_slider.set(0.05)

        self.fgsm_attack_button = customtkinter.CTkButton(tab, text="开始攻击", command=self.run_fgsm_attack)
        self.fgsm_attack_button.grid(row=1, column=1, padx=20, pady=10)

        fgsm_results_frame = customtkinter.CTkFrame(tab)
        fgsm_results_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")
        fgsm_results_frame.grid_columnconfigure((0,1,2), weight=1)
        fgsm_results_frame.grid_rowconfigure(1, weight=1)

        self.fgsm_orig_image_label = customtkinter.CTkLabel(fgsm_results_frame, text="原始图像")
        self.fgsm_orig_image_label.grid(row=0, column=0, pady=5)
        self.fgsm_orig_image_display = customtkinter.CTkLabel(fgsm_results_frame, text="")
        self.fgsm_orig_image_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.fgsm_pert_image_label = customtkinter.CTkLabel(fgsm_results_frame, text="扰动")
        self.fgsm_pert_image_label.grid(row=0, column=1, pady=5)
        self.fgsm_pert_image_display = customtkinter.CTkLabel(fgsm_results_frame, text="")
        self.fgsm_pert_image_display.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.fgsm_adv_image_label = customtkinter.CTkLabel(fgsm_results_frame, text="对抗样本")
        self.fgsm_adv_image_label.grid(row=0, column=2, pady=5)
        self.fgsm_adv_image_display = customtkinter.CTkLabel(fgsm_results_frame, text="")
        self.fgsm_adv_image_display.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        fgsm_text_frame = customtkinter.CTkFrame(tab)
        fgsm_text_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        fgsm_text_frame.grid_columnconfigure((0,1), weight=1)
        self.fgsm_orig_text = customtkinter.CTkTextbox(fgsm_text_frame, height=80)
        self.fgsm_orig_text.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.fgsm_adv_text = customtkinter.CTkTextbox(fgsm_text_frame, height=80)
        self.fgsm_adv_text.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.fgsm_orig_text.insert("0.0", "攻击前结果...")
        self.fgsm_adv_text.insert("0.0", "攻击后结果...")

    def setup_targeted_attack_tab(self):
        tab = self.tab_view.tab("定向攻击")
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        # --- Widgets ---
        # 使用新的工具函数准备类别列表
        all_classes = get_all_classes_with_cn_names(attack_core.RESNET50_WEIGHTS)
        self.class_list_for_gui = [item["label"] for item in all_classes]
        self.class_map_for_gui = {item["label"]: item["value"] for item in all_classes}

        info_label = customtkinter.CTkLabel(tab, text="此功能将固定攻击 backend/images/panda.jpg 图片", anchor="center")
        info_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(10,0))

        controls_frame = customtkinter.CTkFrame(tab)
        controls_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

        self.target_class_combobox = customtkinter.CTkComboBox(controls_frame, values=self.class_list_for_gui, width=250)
        # 查找并设置默认值
        default_label = next((item["label"] for item in all_classes if item["value"] == 504), "")
        self.target_class_combobox.set(default_label)
        self.target_class_combobox.pack(side="left", padx=10, pady=10)

        self.targeted_attack_button = customtkinter.CTkButton(controls_frame, text="开始攻击", command=self.run_targeted_attack)
        self.targeted_attack_button.pack(side="left", padx=10, pady=10)

        self.progress_bar = customtkinter.CTkProgressBar(tab, orientation="horizontal", mode="determinate")
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        targeted_results_frame = customtkinter.CTkFrame(tab)
        targeted_results_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        targeted_results_frame.grid_columnconfigure((0,1,2), weight=1)
        targeted_results_frame.grid_rowconfigure(1, weight=1)

        self.targeted_orig_image_label = customtkinter.CTkLabel(targeted_results_frame, text="原始图像")
        self.targeted_orig_image_label.grid(row=0, column=0, pady=5)
        self.targeted_orig_image_display = customtkinter.CTkLabel(targeted_results_frame, text="")
        self.targeted_orig_image_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.targeted_pert_image_label = customtkinter.CTkLabel(targeted_results_frame, text="扰动")
        self.targeted_pert_image_label.grid(row=0, column=1, pady=5)
        self.targeted_pert_image_display = customtkinter.CTkLabel(targeted_results_frame, text="")
        self.targeted_pert_image_display.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.targeted_adv_image_label = customtkinter.CTkLabel(targeted_results_frame, text="对抗样本")
        self.targeted_adv_image_label.grid(row=0, column=2, pady=5)
        self.targeted_adv_image_display = customtkinter.CTkLabel(targeted_results_frame, text="")
        self.targeted_adv_image_display.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        targeted_text_frame = customtkinter.CTkFrame(tab)
        targeted_text_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        targeted_text_frame.grid_columnconfigure((0,1), weight=1)
        self.targeted_orig_text = customtkinter.CTkTextbox(targeted_text_frame, height=100)
        self.targeted_orig_text.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.targeted_adv_text = customtkinter.CTkTextbox(targeted_text_frame, height=100)
        self.targeted_adv_text.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.targeted_orig_text.insert("0.0", "攻击前结果...")
        self.targeted_adv_text.insert("0.0", "攻击后结果...")

    # ---- Callback Functions ----
    def select_and_predict_image(self):
        initial_dir = os.path.join(os.path.dirname(__file__), "backend", "images")
        file_path = filedialog.askopenfilename(initialdir=initial_dir,
                                               title="选择一个图片文件",
                                               filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("all files", "*.*")))
        if not file_path:
            return

        self.image_path_label.configure(text=file_path)
        self.prediction_textbox.delete("0.0", "end")
        self.prediction_textbox.insert("0.0", "正在识别中，请稍候...")
        self.update()

        pil_image, result_text = attack_core.predict_image(file_path)
        
        if pil_image is None:
            self.prediction_textbox.delete("0.0", "end")
            self.prediction_textbox.insert("0.0", result_text)
            return

        self.prediction_textbox.delete("0.0", "end")
        self.prediction_textbox.insert("0.0", result_text)

        self._update_image_display(self.image_display_label, pil_image)

    def run_fgsm_attack(self):
        epsilon = self.epsilon_slider.get()
        self.fgsm_orig_text.delete("0.0", "end")
        self.fgsm_adv_text.delete("0.0", "end")
        self.fgsm_orig_text.insert("0.0", f"正在以 Epsilon={epsilon:.3f} 进行攻击...")
        self.fgsm_adv_text.insert("0.0", "请稍候...")
        self.update()

        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = attack_core.generate_fgsm_attack(epsilon)

        self.fgsm_orig_text.delete("0.0", "end")
        self.fgsm_orig_text.insert("0.0", orig_txt)
        self.fgsm_adv_text.delete("0.0", "end")
        self.fgsm_adv_text.insert("0.0", adv_txt)

        self._update_image_display(self.fgsm_orig_image_display, orig_pil)
        self._update_image_display(self.fgsm_pert_image_display, pert_pil)
        self._update_image_display(self.fgsm_adv_image_display, adv_pil)

    def run_targeted_attack(self):
        selected_class_str = self.target_class_combobox.get()
        target_id = self.class_map_for_gui.get(selected_class_str)

        if target_id is None:
            tkinter.messagebox.showerror("错误", "无效的选择，请从列表中选择一个目标。")
            return

        self.targeted_attack_button.configure(state="disabled")
        self.target_class_combobox.configure(state="disabled")
        self.targeted_orig_text.delete("0.0", "end")
        self.targeted_adv_text.delete("0.0", "end")
        self.targeted_orig_text.insert("0.0", "正在进行迭代式攻击...")
        self.targeted_adv_text.insert("0.0", "请稍候，此过程约需1-2分钟...")
        self.progress_bar.set(0)
        self.update()

        attack_thread = threading.Thread(target=self._execute_targeted_attack, args=(target_id,))
        attack_thread.start()

    def _execute_targeted_attack(self, target_id):
        results = attack_core.generate_targeted_attack(self.update_progress_bar, target_class_id=target_id)
        self.after(0, self.update_targeted_ui, results)

    def update_progress_bar(self, value):
        self.progress_bar.set(value)

    def update_targeted_ui(self, results):
        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = results

        self.targeted_orig_text.delete("0.0", "end")
        self.targeted_orig_text.insert("0.0", orig_txt)
        self.targeted_adv_text.delete("0.0", "end")
        self.targeted_adv_text.insert("0.0", adv_txt)

        self._update_image_display(self.targeted_orig_image_display, orig_pil)
        self._update_image_display(self.targeted_pert_image_display, pert_pil)
        self._update_image_display(self.targeted_adv_image_display, adv_pil)
        
        self.targeted_attack_button.configure(state="normal")
        self.target_class_combobox.configure(state="normal")

    def get_resized_ctk_image(self, pil_image, max_w, max_h):
        max_w = max(max_w, 100)
        max_h = max(max_h, 100)
        
        img_copy = pil_image.copy()
        img_copy.thumbnail((max_w - 20, max_h - 20), Image.Resampling.LANCZOS)
        return customtkinter.CTkImage(light_image=img_copy, dark_image=img_copy, size=img_copy.size)

    def _update_image_display(self, label_widget, pil_image):
        """Helper to update a CTkLabel with a new PIL image, ensuring reference is kept."""
        w, h = label_widget.winfo_width(), label_widget.winfo_height()
        ctk_img = self.get_resized_ctk_image(pil_image, w, h)
        label_widget.configure(image=ctk_img)
        label_widget.image = ctk_img # Keep a reference to prevent garbage collection

if __name__ == "__main__":
    # 解决SSL证书问题，以便下载模型
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    app = App()
    app.mainloop()