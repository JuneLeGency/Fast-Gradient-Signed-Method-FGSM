import customtkinter
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# 导入我们重构的核心逻辑
import attack_core

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # ---- 窗口基础设置 ----
        self.title("AI 对抗攻击演示")
        self.geometry("1100x700")
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

        # -- Widgets --
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

        # -- Widgets --
        info_label = customtkinter.CTkLabel(tab, text="此功能将固定攻击 images/panda.jpg 图片", anchor="center")
        info_label.grid(row=0, column=0, columnspan=3, padx=20, pady=(10,0))

        epsilon_frame = customtkinter.CTkFrame(tab)
        epsilon_frame.grid(row=1, column=0, padx=20, pady=10)
        epsilon_label = customtkinter.CTkLabel(epsilon_frame, text="扰动强度 (Epsilon): ")
        epsilon_label.pack(side="left", padx=(10,5), pady=10)
        self.epsilon_slider = customtkinter.CTkSlider(epsilon_frame, from_=0, to=0.2, number_of_steps=20)
        self.epsilon_slider.pack(side="left", padx=(5,10), pady=10)
        self.epsilon_slider.set(0.05)

        self.attack_button = customtkinter.CTkButton(tab, text="开始攻击", command=self.run_fgsm_attack)
        self.attack_button.grid(row=1, column=1, padx=20, pady=10)

        # Image display frames
        self.fgsm_results_frame = customtkinter.CTkFrame(tab)
        self.fgsm_results_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=10, sticky="nsew")
        self.fgsm_results_frame.grid_columnconfigure((0,1,2), weight=1)
        self.fgsm_results_frame.grid_rowconfigure(1, weight=1)

        self.orig_image_label = customtkinter.CTkLabel(self.fgsm_results_frame, text="原始图像")
        self.orig_image_label.grid(row=0, column=0, pady=5)
        self.orig_image_display = customtkinter.CTkLabel(self.fgsm_results_frame, text="")
        self.orig_image_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.pert_image_label = customtkinter.CTkLabel(self.fgsm_results_frame, text="扰动")
        self.pert_image_label.grid(row=0, column=1, pady=5)
        self.pert_image_display = customtkinter.CTkLabel(self.fgsm_results_frame, text="")
        self.pert_image_display.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.adv_image_label = customtkinter.CTkLabel(self.fgsm_results_frame, text="对抗样本")
        self.adv_image_label.grid(row=0, column=2, pady=5)
        self.adv_image_display = customtkinter.CTkLabel(self.fgsm_results_frame, text="")
        self.adv_image_display.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        # Text results
        self.fgsm_text_frame = customtkinter.CTkFrame(tab)
        self.fgsm_text_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        self.fgsm_text_frame.grid_columnconfigure((0,1), weight=1)
        self.orig_text = customtkinter.CTkTextbox(self.fgsm_text_frame, height=80)
        self.orig_text.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.adv_text = customtkinter.CTkTextbox(self.fgsm_text_frame, height=80)
        self.adv_text.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.orig_text.insert("0.0", "攻击前结果...")
        self.adv_text.insert("0.0", "攻击后结果...")

    def setup_targeted_attack_tab(self):
        tab = self.tab_view.tab("定向攻击")
        placeholder_label = customtkinter.CTkLabel(tab, text="此功能正在开发中...", font=("", 20))
        placeholder_label.pack(expand=True, padx=20, pady=20)

    # ---- Callback Functions ----
    def select_and_predict_image(self):
        file_path = filedialog.askopenfilename(initialdir=os.path.join(os.path.dirname(__file__), "images"),
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
            self.prediction_textbox.insert("0.0", result_text) # Show error message
            return

        # Update text
        self.prediction_textbox.delete("0.0", "end")
        self.prediction_textbox.insert("0.0", result_text)

        # Update image
        w, h = self.image_display_label.winfo_width(), self.image_display_label.winfo_height()
        resized_img = self.get_resized_ctk_image(pil_image, w, h)
        self.image_display_label.configure(image=resized_img)

    def run_fgsm_attack(self):
        epsilon = self.epsilon_slider.get()
        self.orig_text.delete("0.0", "end")
        self.adv_text.delete("0.0", "end")
        self.orig_text.insert("0.0", f"正在以 Epsilon={epsilon:.3f} 进行攻击...")
        self.adv_text.insert("0.0", "请稍候...")
        self.update()

        orig_pil, pert_pil, adv_pil, orig_txt, adv_txt = attack_core.generate_fgsm_attack(epsilon)

        # Update text
        self.orig_text.delete("0.0", "end")
        self.orig_text.insert("0.0", orig_txt)
        self.adv_text.delete("0.0", "end")
        self.adv_text.insert("0.0", adv_txt)

        # Update images
        w, h = self.orig_image_display.winfo_width(), self.orig_image_display.winfo_height()
        self.orig_image_display.configure(image=self.get_resized_ctk_image(orig_pil, w, h))
        self.pert_image_display.configure(image=self.get_resized_ctk_image(pert_pil, w, h))
        self.adv_image_display.configure(image=self.get_resized_ctk_image(adv_pil, w, h))

    def get_resized_ctk_image(self, pil_image, max_w, max_h):
        max_w = max(max_w, 100) # Ensure non-zero size
        max_h = max(max_h, 100)
        
        pil_image.thumbnail((max_w - 20, max_h - 20), Image.Resampling.LANCZOS)
        return customtkinter.CTkImage(light_image=pil_image, dark_image=pil_image, size=pil_image.size)

if __name__ == "__main__":
    # 解决SSL证书问题，以便下载模型
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    app = App()
    app.mainloop()
