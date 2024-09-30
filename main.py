import os
import logging
import json
import asyncio
import re
import aiofiles
import wikitextparser as wtp
from typing import List, Dict, Any, Type, TypeVar
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import networkx as nx
import instructor
import openai
import mistletoe
from mistletoe.ast_renderer import ASTRenderer

# Settings configuration
current_dir = os.getcwd()

class Settings(BaseSettings):
    DEFAULT_MODEL_IDENTIFIER: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 1000
    STYLE_GUIDE_PATH: str = Field(default=os.path.join(current_dir, "styleguide.txt"))
    ARTICLE_PATH: str = Field(default=os.path.join(current_dir, "article.md"))
    LOG_LEVEL: str = "INFO"
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    MOCK_EXTERNAL_CALLS: bool = Field(default=False)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MOCK_EXTERNAL_CALLS = False

settings = Settings()

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Constants
ROOT_NODE_ID = 'root'
UNKNOWN_CATEGORY = 'Unknown'

# Exception classes
class OmnipediaError(Exception):
    """Base exception for Omnipedia"""

class StyleGuideProcessingError(OmnipediaError):
    """Raised when there's an error processing the style guide"""

class ArticleParsingError(OmnipediaError):
    """Raised when there's an error parsing an article"""

class EvaluationError(OmnipediaError):
    """Raised when there's an error during the evaluation process"""

class LanguageModelError(OmnipediaError):
    """Raised when there's an error with the language model"""

# Data models
class Requirement(BaseModel):
    """
    Represents a requirement extracted from the style guide.

    Attributes:
        name (str): The name of the requirement.
        description (str): A detailed description of the requirement.
        applicable_sections (List[str]): A list of article sections where this requirement applies.
    """
    name: str
    description: str
    applicable_sections: List[str]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "applicable_sections": self.applicable_sections
        }

class StyleGuide(BaseModel):
    """
    Represents the processed style guide.

    Attributes:
        requirements (List[Requirement]): A list of requirements extracted from the style guide.
    """
    requirements: List[Requirement]

class ArticleNode(BaseModel):
    id: str
    title: str
    content: str
    level: int = 0
    children: List['ArticleNode'] = Field(default_factory=list)


class EvaluatedSection(BaseModel):
    """
    Represents an evaluated section of an article.

    Attributes:
        section (str): The identifier of the evaluated section.
        score (float): The evaluation score for the section.
        feedback (str): Feedback on the section's adherence to style guidelines.
        adherent_requirements (List[str]): Requirements that the section adheres to.
        templates (List[str]): Templates used in the section.
        wikilinks (List[str]): Wiki links found in the section.
        external_links (List[str]): External links found in the section.
        list_items (List[str]): List items found in the section.
    """
    section: str
    score: float
    feedback: str
    adherent_requirements: List[str]
    templates: List[str] = Field(default_factory=list)
    wikilinks: List[str] = Field(default_factory=list)
    external_links: List[str] = Field(default_factory=list)
    list_items: List[str] = Field(default_factory=list)

class EvaluatedArticle(BaseModel):
    """
    Represents the evaluated article containing evaluated sections.

    Attributes:
        evaluated_sections (List[EvaluatedSection]): A list of evaluated sections.
    """
    evaluated_sections: List[EvaluatedSection]

# Initialize logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility classes
class ArticleGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_id_to_index = {}
        self.current_index = 0

    def add_node(self, node_id: str, content: str, category: str):
        self.graph.add_node(node_id, data=ArticleNode(location=node_id, content=content, category=category))
        self.node_id_to_index[node_id] = self.current_index
        self.current_index += 1

    def add_edge(self, parent_id: str, child_id: str):
        self.graph.add_edge(parent_id, child_id)

    def get_node(self, node_id: str) -> ArticleNode:
        return self.graph.nodes[node_id]['data']

    def get_children(self, node_id: str):
        return list(self.graph.successors(node_id))

    def get_node_index(self, node_id: str) -> int:
        return self.node_id_to_index[node_id]


T = TypeVar('T', bound=BaseModel)
class LanguageModel:
    def __init__(self, model_identifier: str = settings.DEFAULT_MODEL_IDENTIFIER):
        self.model_identifier = model_identifier
        self.client = instructor.from_openai(openai.AsyncOpenAI())

    async def prompt(self, text: str, response_model: Type[T]) -> T:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_identifier,
                response_model=response_model,
                messages=[{"role": "user", "content": text}],
                max_tokens=settings.MAX_TOKENS
            )
            return response
        except Exception as e:
            logger.error(f"Language model API error: {e}", exc_info=True)
            raise LanguageModelError(f"Failed to get response from language model: {str(e)}")
# Parsers
class StyleGuideProcessor:
    def __init__(self, style_guide_path: str, llm: 'LanguageModel', requirements_json_path: str = "requirements.json"):
        self.style_guide_path = style_guide_path
        self.llm = llm
        self.requirements_json_path = requirements_json_path

    async def process(self) -> StyleGuide:
        try:
            # First, try to load requirements from JSON
            requirements = await self._load_requirements_from_json()
            
            if requirements:
                logger.info("Requirements loaded from JSON file.")
                return StyleGuide(requirements=requirements)
            
            # If no JSON file or it's empty, process the style guide
            logger.info("Processing style guide to extract requirements.")
            async with aiofiles.open(self.style_guide_path, 'r') as file:
                guide_text = await file.read()
            
            parsed_guide = wtp.parse(guide_text)
            sections = self._extract_sections(parsed_guide)
            requirements = await self._extract_requirements(sections)
            
            # Save the extracted requirements to JSON for future use
            await self._save_requirements_to_json(requirements)
            
            return StyleGuide(requirements=requirements)
        except Exception as e:
            logger.error(f"Error processing style guide: {e}", exc_info=True)
            raise StyleGuideProcessingError(f"Failed to process style guide: {str(e)}")

    async def _load_requirements_from_json(self) -> List[Requirement]:
        try:
            if os.path.exists(self.requirements_json_path):
                async with aiofiles.open(self.requirements_json_path, 'r') as f:
                    requirements_data = json.loads(await f.read())
                return [Requirement(**req) for req in requirements_data]
            return []
        except Exception as e:
            logger.error(f"Error loading requirements from JSON: {e}", exc_info=True)
            return []

    async def _save_requirements_to_json(self, requirements: List[Requirement]):
        try:
            requirements_data = [req.to_dict() for req in requirements]
            async with aiofiles.open(self.requirements_json_path, 'w') as f:
                await f.write(json.dumps(requirements_data, indent=2))
            logger.info(f"Requirements saved to {self.requirements_json_path}")
        except Exception as e:
            logger.error(f"Error saving requirements to JSON: {e}", exc_info=True)

    def _extract_sections(self, parsed_guide: wtp.WikiText) -> List[dict]:
        sections = []
        for section in parsed_guide.sections:
            if section.title:
                sections.append({
                    "title": section.title.strip(),
                    "content": section.contents.strip()
                })
        return sections

    async def _extract_requirements(self, sections: List[dict]) -> List[Requirement]:
        all_requirements = []
        standardized_sections = [
            "Lead",
            "Gene",
            "Protein",
            "Structure",
            "Species, Tissue, and Subcellular Distribution",
            "Function",
            "Interactions",
            "Clinical Significance",
            "History/Discovery",
            "References",
            "Images and Diagrams",
            "Navigation Box",
            "Categories",
            "All Sections"
        ]
        for section in sections:
            prompt = f"""
            Analyze the following style guide section and extract actionable writing rules:

            Section Title: {section['title']}
            Content:
            {section['content']}

            For each rule, provide:
            - name: A brief title for the requirement
            - description: Detailed explanation of the requirement
            - applicable_sections: List of article sections where this requirement applies. Use only the following standardized section titles:
            {', '.join(standardized_sections)}

            Return the results as a list of Requirement objects.
            """
            try:
                response = await self.llm.prompt(prompt, List[Requirement])
                all_requirements.extend(response)
            except Exception as e:
                logger.error(f"Error processing section {section['title']}: {e}")
                all_requirements.append(Requirement(
                    name=f"Error in section: {section['title']}",
                    description=f"Failed to process this section: {str(e)}",
                    applicable_sections=[],
                ))

        logger.info(f"Extracted {len(all_requirements)} requirements")
        return all_requirements

    
    async def _summarize_sections(self, sections: List[dict]) -> List[dict]:
        summarized_sections = []
        for section in sections:
            summary_prompt = f""" 
            Here are your instructions:

            1. Carefully read and analyze the text provided.
            2. Identify key guidelines, rules, or recommendations in the Style Guide (text provided).
            3. For each guideline you identify:
            a. Create a new JSON object.
            b. Include a "name" field that succinctly describes the guideline.
            c. Include a "description" field that explains the guideline in more detail.
            d. Include a "location" field that lists where this guideline applies.
            4. Present your findings as a list of these JSON objects.

            Remember to focus on substantial guidelines or rules, not minor details. Your goal is to make a comprehensive list of requirements that a writer should follow.

            {section["content"]}

            Begin your analysis now.
            """
            try:
                response = await self.llm.prompt(summary_prompt, List[Requirement])
                summarized_sections.append({
                    "title": section["title"],
                    "content": [req.dict() for req in response]
                })
            except Exception as e:
                logger.error(f"Error summarizing section {section['title']}: {e}")
                summarized_sections.append({
                    "title": section["title"],
                    "content": [{"name": "Error", "description": f"Failed to summarize: {str(e)}", "applicable_sections": []}]
                })

        async with aiofiles.open("summarized.json", 'w') as f:
            await f.write(json.dumps(summarized_sections, indent=2))
        
        return summarized_sections

class ArticleParser:
    def __init__(self):
        self.ast_renderer = ASTRenderer()
        self.current_id = 0

    def parse(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Parse Markdown to AST using mistletoe
        ast_json = self.ast_renderer.render(mistletoe.Document(content))

        # Parse the JSON string into a Python object
        ast = json.loads(ast_json)

        return self._process_ast(ast)

    def _process_ast(self, ast: Dict[str, Any]) -> ArticleNode:
        root = ArticleNode(id="root", title="Overall Title", content="", level=0)
        if 'children' in ast:
            self._process_children(ast['children'], root)
        else:
            raise ArticleParsingError("AST structure is not as expected")
        return root

    def _process_children(self, children, root: ArticleNode):
        section_stack = [root]

        for item in children:
            item_type = item.get('type')
            if item_type == 'Heading':
                heading_level = item.get('level', 1)
                new_section = self._create_section(item, heading_level)

                # Adjust the stack based on heading level
                while section_stack and section_stack[-1].level >= heading_level:
                    section_stack.pop()
                section_stack[-1].children.append(new_section)
                section_stack.append(new_section)
            elif item_type in ['Paragraph', 'List']:
                current_section = section_stack[-1]
                current_section.content += self._extract_content(item)
            elif item.get('children'):
                self._process_children(item['children'], root=section_stack[-1])

    def _create_section(self, heading: Dict[str, Any], level: int) -> ArticleNode:
        self.current_id += 1
        title = self._extract_text(heading.get('children', []))
        return ArticleNode(
            id=f"section_{self.current_id}",
            title=title,
            content="",
            level=level,
        )

    def _extract_content(self, node: Dict[str, Any]) -> str:
        if node['type'] == 'Paragraph':
            return self._extract_text(node.get('children', [])) + "\n\n"
        elif node['type'] == 'List':
            return self._extract_list(node) + "\n\n"
        return ""

    def _extract_text(self, children) -> str:
        texts = []
        for child in children:
            if child['type'] == 'RawText':
                texts.append(child['content'])
            elif 'children' in child:
                texts.append(self._extract_text(child['children']))
        return ' '.join(texts)

    def _extract_list(self, node: Dict[str, Any]) -> str:
        list_items = []
        for item in node.get('children', []):
            item_text = self._extract_text(item.get('children', []))
            list_items.append(f"- {item_text}")
        return "\n".join(list_items)

    def serialize_article_structure(self, node: ArticleNode) -> dict:
        """Serialize the ArticleNode structure to a dictionary."""
        return {
            "id": node.id,
            "title": node.title,
            "content": node.content,
            "level": node.level,
            "children": [self.serialize_article_structure(child) for child in node.children]
        }

class ArticleEvaluator:
    def __init__(self, llm: LanguageModel, requirements: List[Requirement]):
        self.llm = llm
        self.requirements = requirements

    async def evaluate(self, article_node: ArticleNode) -> List[EvaluatedSection]:
        try:
            sections_to_evaluate = self._prepare_sections_for_evaluation(article_node)
            print(sections_to_evaluate)
            evaluated_sections = []
            for section in sections_to_evaluate:
                prompt = self._generate_evaluation_prompt(section)
                result = await self.llm.prompt(prompt, EvaluatedSection)
                evaluated_sections.append(result)

            return evaluated_sections
        except Exception as e:
            logger.error(f"Error evaluating article: {e}", exc_info=True)
            raise EvaluationError(f"Failed to evaluate article: {str(e)}")

    def _prepare_sections_for_evaluation(self, article_node: ArticleNode) -> List[Dict[str, Any]]:
        sections_to_evaluate = []
        self._collect_sections(article_node, sections_to_evaluate)
        return sections_to_evaluate

    def _collect_sections(self, node: ArticleNode, sections: List[Dict[str, Any]]):
        sections.append({
            "section_id": node.id,
            "title": node.title,
            "content": node.content,
        })
        for child in node.children:
            self._collect_sections(child, sections)

    def _generate_evaluation_prompt(self, section: Dict[str, Any]) -> str:
        # Convert Requirement objects to dictionaries
        serializable_requirements = [
            {
                "name": req.name,
                "description": req.description,
                "applicable_sections": req.applicable_sections
            }
            for req in self.requirements
        ]

        return f"""
        Evaluate the following article section against the style guide requirements:

        Section ID: {section['section_id']}
        Title: {section['title']}
        Content:
        {section['content']}

        Style Guide Requirements:
        {json.dumps(serializable_requirements, indent=2)}

        Provide:
        1. A score from 0 to 1 (1 being perfect adherence)
        2. Feedback explaining the score
        3. List of adherent requirements

        Return the result as an EvaluatedSection object.
        """

# Main application class
class Omnipedia:
    def __init__(self, style_guide_path: str, language_model: LanguageModel, requirements_path: str):
        self.llm = language_model
        self.style_guide_processor = StyleGuideProcessor(style_guide_path, language_model, requirements_path)
        self.style_guide = None
        self.evaluator = None
        self.article_parser = ArticleParser()

    async def initialize(self):
        try:
            self.style_guide = await self.style_guide_processor.process()
            self.evaluator = ArticleEvaluator(self.llm, self.style_guide.requirements)
        except Exception as e:
            logger.error(f"Error initializing Omnipedia: {e}", exc_info=True)
            raise

    async def evaluate_article(self, article_path: str):
        try:
            article_node = self.article_parser.parse(article_path)
            
            # Serialize and save the article structure
            article_structure = self.article_parser.serialize_article_structure(article_node)
            with open("article_structure.json", 'w') as file:
                json.dump(article_structure, file, indent=2)
            logger.info("Article structure saved to article_structure.json")

            return await self.evaluator.evaluate(article_node)
        except Exception as e:
            logger.error(f"Error evaluating article: {e}", exc_info=True)
            raise
    
    async def save_requirements_to_json(self, filename: str):
        try:
            requirements_data = [req.dict() for req in self.style_guide.requirements]
            async with aiofiles.open(filename, 'w') as f:
                await f.write(json.dumps(requirements_data, indent=2))
            logger.info(f"Requirements saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving requirements to JSON: {e}", exc_info=True)
            raise
# Main function
async def main():
    try:
        language_model = LanguageModel(settings.DEFAULT_MODEL_IDENTIFIER)
        omnipedia = Omnipedia(settings.STYLE_GUIDE_PATH, language_model, "summarized.json")

        await omnipedia.initialize()
        
        evaluated_sections = await omnipedia.evaluate_article(settings.ARTICLE_PATH)
        print("Evaluation complete.")
        for section in evaluated_sections:
            print(f"Section: {section.section}")
            print(f"Score: {section.score}")
            print(f"Feedback: {section.feedback}")
            print("---")

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}", exc_info=True)
