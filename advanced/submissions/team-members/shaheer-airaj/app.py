import gradio as gr
import asyncio
from research_workflow import run_research_pipeline

def run_research(query: str) -> tuple[str, str]:
    """
    Wrapper function to run async research pipeline in Gradio.
    Returns: (final_report, research_details)
    """
    if not query or query.strip() == "":
        return "Please enter a research topic.", ""
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_report, research_details = loop.run_until_complete(run_research_pipeline(query))
        loop.close()
        return final_report, research_details
    except Exception as e:
        return f"An error occurred: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="ScholarAI - Multi-Agent Research Assistant") as demo:
    gr.Markdown("# 🔴 ScholarAI - Multi-Agent Research Assistant")
    gr.Markdown("Enter a research topic and let our AI agents collaboratively research, analyze, and synthesize information for you.")
    
    query_input = gr.Textbox(
        label="Research Topic",
        placeholder="Enter your research question or topic (e.g., 'Impact of AI on healthcare')",
        lines=3
    )
    
    submit_btn = gr.Button("🔍 Start Research", variant="primary")
    
    gr.Markdown("### 📄 Final Synthesized Report")
    final_output = gr.Textbox(label="", lines=15)
    
    gr.Markdown("### 🔍 Detailed Research Findings")
    research_output = gr.Textbox(label="", lines=15)
    
    # Event handler
    submit_btn.click(
        fn=run_research,
        inputs=[query_input],
        outputs=[final_output, research_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
