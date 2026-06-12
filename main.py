import threading
import tkinter as tk
from tkinter import ttk
from ui.app import DinoSimilarityApp
from core.model import load_model


def main():
    root = tk.Tk()
    root.geometry('1380x800')
    root.minsize(1000, 650)

    app = DinoSimilarityApp(root)

    # ── 启动时加载模型 ──────────────────────────────
    def _do_load():
        """后台线程：加载模型"""
        load_model(device="cuda")
        # 加载完成后回到主线程解锁 UI
        root.after(0, _on_model_ready)

    def _on_model_ready():
        """模型就绪，解锁按钮，更新状态栏"""
        app.set_model_ready()

    # 锁住按钮，显示加载中
    app.set_model_loading()

    # 后台线程不阻塞 UI
    t = threading.Thread(target=_do_load, daemon=True)
    t.start()

    root.mainloop()


if __name__ == '__main__':
    main()
