"""Small local Tk interface. No server, analytics, uploads or automatic logging."""
import json
import threading
import tkinter as tk
from tkinter import ttk
from .predict import Predictor


def main():
    window = tk.Tk()
    window.title("CyberLens — Local scam classification")
    window.geometry("1000x760")
    ttk.Label(window, text="CyberLens", font=("Segoe UI", 22, "bold")).pack(anchor="w", padx=20, pady=(15,5))
    ttk.Label(window, text="English / Hinglish • Typo-tolerant • Runs locally").pack(anchor="w", padx=20)
    ttk.Label(window, text="Prototype: model scores are uncalibrated. Complaint labels have no benign class.\n"
              "Names/addresses may remain after rule-based masking. Input is not saved.").pack(anchor="w", padx=20, pady=10)
    entry = tk.Text(window, height=7, wrap="word", font=("Segoe UI", 12))
    entry.pack(fill="x", padx=20)
    controls = ttk.Frame(window)
    controls.pack(fill="x", padx=20, pady=10)
    backend = tk.StringVar(value="best")
    ttk.Label(controls, text="Model:").pack(side="left")
    ttk.Combobox(controls, textvariable=backend, values=("best", "baseline", "word2vec", "subword", "transformer", "hierarchical", "robust", "regularized"),
                 state="readonly", width=16).pack(side="left", padx=8)
    output = tk.Text(window, wrap="word", font=("Consolas", 11), state="disabled")
    output.pack(fill="both", expand=True, padx=20, pady=(0,20))
    predictors = {}
    def display(value):
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("end", value)
        output.configure(state="disabled")
        button.configure(state="normal")
    def run(text, mode):
        try:
            predictor = predictors.setdefault(mode, Predictor(backend=mode))
            data = predictor.predict(text)
            lines = ["MODEL INPUT (partially masked):", data["masked_text"], ""]
            for task, result in data["predictions"].items():
                lines.extend([task.upper(), result["scope"]])
                lines.append("Suggested type: "+result["suggested_type"])
                if "category_top" in result:
                    category = result["category_top"][0]
                    lines.append(f"Broad category: {category['label']} ({category['score']:.1%} aggregated score)")
                for item in result["top"]:
                    lines.append(f"  {item['score']:.1%}  {item['label']}")
                lines.extend(["Review needed: " + str(result["needs_review"]), ""])
            lines.append("Top-k labels are alternatives, not simultaneous confirmed findings.")
            window.after(0, display, "\n".join(lines))
        except Exception as exc:
            window.after(0, display, f"Could not classify: {exc}\nTrain this backend first.")
    def submit():
        text = entry.get("1.0", "end").strip()
        if not text:
            display("Enter a message first.")
            return
        button.configure(state="disabled")
        threading.Thread(target=run, args=(text, backend.get()), daemon=True).start()
    button = ttk.Button(controls, text="Classify locally", command=submit)
    button.pack(side="left", padx=8)
    window.mainloop()


if __name__ == "__main__":
    main()
