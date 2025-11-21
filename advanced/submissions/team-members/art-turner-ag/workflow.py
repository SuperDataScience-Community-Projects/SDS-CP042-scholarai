from agents.topic_splitter import TopicSplitterAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
import concurrent.futures

class ResearchWorkflow:
    def __init__(self):
        self.splitter = TopicSplitterAgent()
        self.researcher = ResearcherAgent()
        self.synthesizer = SynthesizerAgent()

    def run(self, topic):
        print(f"Starting research on: {topic}")
        
        # 1. Split Topic
        print("Splitting topic...")
        subtopics = self.splitter.run(topic)
        print(f"Sub-topics: {subtopics}")
        
        # 2. Parallel Research
        print("Conducting research...")
        research_findings = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_subtopic = {executor.submit(self.researcher.run, subtopic): subtopic for subtopic in subtopics}
            for future in concurrent.futures.as_completed(future_to_subtopic):
                subtopic = future_to_subtopic[future]
                try:
                    finding = future.result()
                    research_findings[subtopic] = finding
                except Exception as e:
                    print(f"Error researching {subtopic}: {e}")
                    research_findings[subtopic] = f"Error: {e}"

        # 3. Synthesize
        print("Synthesizing findings...")
        final_report = self.synthesizer.run(topic, research_findings)
        
        return final_report, research_findings
