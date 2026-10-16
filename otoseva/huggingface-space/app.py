import gradio as gr
import math

LABELS = ["AOM", "ASOM", "CSOM/COM", "Normal"]
SCENARIOS = {
    "Routine / Normal-like": [0.08, 0.04, 0.03, 0.85],
    "AOM-like": [0.72, 0.10, 0.05, 0.13],
    "ASOM-like": [0.08, 0.78, 0.06, 0.08],
    "CSOM-like": [0.09, 0.08, 0.70, 0.13],
    "Uncertain / Defer": [0.31, 0.25, 0.20, 0.24],
}

def entropy_norm(p):
    h = -sum(v * math.log(v) for v in p if v > 0)
    return h / math.log(len(p))

def run_demo(image, scenario):
    p = SCENARIOS[scenario]
    conf = max(p)
    ent = entropy_norm(p)
    defer = conf < 0.60 or ent > 0.72
    routing = "DEFER → repeat capture / clinician review" if defer else "ACCEPT → routine clinician review"
    probs = {k: float(v) for k, v in zip(LABELS, p)}
    workflow = (
        "1. Capture: image received\n"
        "2. Quality gate: workflow check demonstrated\n"
        "3. MONAI preprocessing: standardized transform path demonstrated\n"
        f"4. Reliability: confidence={conf:.2f}, normalized entropy={ent:.2f}\n"
        f"5. Referral: {routing}"
    )
    note = (
        "DEMO ONLY — probabilities are illustrative routing scenarios, not predictions from the validated model. "
        "OtoSeva AI is human-supervised screening/referral support, not autonomous diagnosis."
    )
    return probs, routing, workflow, note

with gr.Blocks(title="OtoSeva AI") as demo:
    gr.Markdown(
        """
# 👂 OtoSeva AI
### MONAI-enabled human-supervised otoscopic screening & referral workflow

This public prototype demonstrates the **capture → quality gate → MONAI processing → reliability → referral** workflow.

> **Safety boundary:** This is a workflow demonstration, not a diagnostic medical device. Uploaded images are not used to produce clinical predictions in this demo.
"""
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy", label="Optional otoscopic image for workflow demonstration")
            scenario = gr.Dropdown(list(SCENARIOS.keys()), value="Routine / Normal-like", label="Illustrative scenario")
            run = gr.Button("Run workflow demo", variant="primary")
        with gr.Column():
            probs = gr.Label(label="Illustrative probability distribution")
            routing = gr.Textbox(label="Referral action")
    workflow = gr.Textbox(label="Workflow trace", lines=6)
    note = gr.Textbox(label="Evidence & safety note", lines=3)
    gr.Markdown(
        """
**Internal evidence summary:** 687 QA-clean images; 115 locked internal test images; macro ROC-AUC 0.873; ECE 0.082 after calibration; retained-set accuracy 0.815 at 80% coverage.

**Evidence boundary:** internal retrospective technical validation only. External device-diverse prospective validation is planned before clinical deployment.

[Product website](https://rawcdn.githack.com/singhankitsrf/singhankitsrf/db07c704b5e6d71668154c5b2f98dee72b8ba075/otoseva/index.html) · [GitHub portfolio](https://github.com/singhankitsrf/singhankitsrf/tree/main/otoseva)
"""
    )
    run.click(run_demo, inputs=[image, scenario], outputs=[probs, routing, workflow, note])

if __name__ == "__main__":
    demo.launch()
