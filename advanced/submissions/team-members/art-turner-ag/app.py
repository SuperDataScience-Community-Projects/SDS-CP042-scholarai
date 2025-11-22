import gradio as gr
from workflow import ResearchWorkflow
import os
from dotenv import load_dotenv

load_dotenv()

def run_research(topic, progress=gr.Progress()):
    workflow = ResearchWorkflow()
    
    def progress_callback(message, step=None, total_steps=None):
        if step is not None and total_steps is not None:
            progress(step / total_steps, desc=message)
        else:
            progress(0, desc=message)

    final_report, research_findings = workflow.run(topic, progress_callback=progress_callback)
    
    findings_display = ""
    for subtopic, finding in research_findings.items():
        findings_display += f"## {subtopic}\n{finding}\n\n---\n\n"
        
    return final_report, findings_display

with gr.Blocks(title="ScholarAI Advanced Research Agent") as demo:
    gr.Markdown("# 🎓 ScholarAI Advanced Research Agent")
    gr.Markdown("Enter a research topic to generate a comprehensive report using multi-agent collaboration.")
    
    with gr.Row():
        topic_input = gr.Textbox(label="Research Topic", placeholder="e.g., The Future of Quantum Computing")
        submit_btn = gr.Button("Start Research", variant="primary")
    
    with gr.Tabs():
        with gr.TabItem("Final Report"):
            report_output = gr.Markdown()
        with gr.TabItem("Detailed Findings"):
            findings_output = gr.Markdown()
            
    submit_btn.click(
        fn=run_research,
        inputs=[topic_input],
        outputs=[report_output, findings_output]
    )

if __name__ == "__main__":
    demo.launch()
