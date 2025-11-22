from agents.topic_splitter import TopicSplitterAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
import concurrent.futures

class ResearchWorkflow:
    def __init__(self):
        self.splitter = TopicSplitterAgent()
        self.researcher = ResearcherAgent()
        self.synthesizer = SynthesizerAgent()

    def run(self, topic, progress_callback=None):
        def report_progress(message, step=None, total_steps=None):
            if progress_callback:
                progress_callback(message, step, total_steps)
            else:
                print(message)

        report_progress(f"Starting research on: {topic}", 0, 3)
        
        # 1. Split Topic
        report_progress("Splitting topic...", 0.1, 3)
        subtopics = self.splitter.run(topic)
        print(f"Sub-topics: {subtopics}")
        
        # 2. Parallel Research
        report_progress("Conducting research...", 1, 3)
        research_findings = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_subtopic = {executor.submit(self.researcher.run, subtopic): subtopic for subtopic in subtopics}
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_subtopic):
                subtopic = future_to_subtopic[future]
                try:
                    finding = future.result()
                    research_findings[subtopic] = finding
                except Exception as e:
                    print(f"Error researching {subtopic}: {e}")
                    research_findings[subtopic] = f"Error: {e}"
                
                completed_count += 1
                report_progress(f"Researched {subtopic}", 1 + (completed_count / len(subtopics)), 3)

        # 3. Synthesize
        report_progress("Synthesizing findings...", 2, 3)
        final_report = self.synthesizer.run(topic, research_findings)
        
        report_progress("Research complete!", 3, 3)
        return final_report, research_findings
