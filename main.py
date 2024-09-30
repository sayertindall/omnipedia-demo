import os
import logging
import json
import asyncio
import aiofiles 
import wikitextparser as wtp  
from typing import List, Dict, Any, Type, TypeVar
from pydantic import BaseModel, Field  
from pydantic_settings import BaseSettings  
import instructor  
import openai  
import mistletoe  
from mistletoe.ast_renderer import ASTRenderer 

# Settings configuration
current_dir = os.getcwd()  # Get the current working directory

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or default values.
    """
    DEFAULT_MODEL_IDENTIFIER: str = "gpt-3.5-turbo"  # Default language model identifier
    MAX_TOKENS: int = 1000  # Maximum number of tokens for language model responses
    STYLE_GUIDE_PATH: str = Field(default=os.path.join(current_dir, "styleguide.txt"))  # Path to the style guide file
    ARTICLE_PATH: str = Field(default=os.path.join(current_dir, "article.md"))  # Path to the article file
    LOG_LEVEL: str = "INFO"  # Logging level
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")  # OpenAI API key from environment variable

    class Config:
        env_file = ".env"  # File to load environment variables from
        env_file_encoding = "utf-8"  # Encoding of the .env file

settings = Settings()  # Instantiate settings with values from the environment or defaults

# Set the OpenAI API key as an environment variable for the OpenAI client
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Exception classes for custom error handling
class OmnipediaError(Exception):
    """Base exception for Omnipedia."""
    pass

class StyleGuideProcessingError(OmnipediaError):
    """Raised when there's an error processing the style guide."""
    pass

class ArticleParsingError(OmnipediaError):
    """Raised when there's an error parsing an article."""
    pass

class EvaluationError(OmnipediaError):
    """Raised when there's an error during the evaluation process."""
    pass

class LanguageModelError(OmnipediaError):
    """Raised when there's an error with the language model."""
    pass

# Data models using Pydantic for validation
class Requirement(BaseModel):
    """
    Represents a requirement extracted from the style guide.
    """
    name: str  # Name of the requirement
    description: str  # Description of the requirement
    applicable_sections: List[str]  # Sections of the article where this requirement applies

    def to_dict(self):
        """Convert the Requirement instance to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "applicable_sections": self.applicable_sections
        }

class StyleGuide(BaseModel):
    """
    Represents the style guide containing a list of requirements.
    """
    requirements: List[Requirement]

class ArticleNode(BaseModel):
    """
    Represents a node in the article structure (e.g., a section).
    """
    id: str  # Unique identifier for the node
    title: str  # Title of the section
    content: str  # Content of the section
    level: int = 0  # Heading level (e.g., 1 for H1)
    children: List['ArticleNode'] = Field(default_factory=list)  # Child sections

class EvaluatedSection(BaseModel):
    """
    Represents the evaluation result of a section.
    """
    section: str  # Section identifier or title
    score: float  # Score from 0 to 1 indicating adherence to the style guide
    feedback: str  # Feedback on the section

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the EvaluatedSection instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the EvaluatedSection.
        """
        return {
            "section": self.section,
            "score": self.score,
            "feedback": self.feedback
        }

# Initialize logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # Logger for this module

# Type variable for language model responses
T = TypeVar('T', bound=BaseModel)

class LanguageModel:
    """
    Interface for interacting with the language model.
    """
    def __init__(self, model_identifier: str = settings.DEFAULT_MODEL_IDENTIFIER):
        self.model_identifier = model_identifier  # Model to use
        self.client = instructor.from_openai(openai.AsyncOpenAI())  # Initialize the OpenAI client

    async def prompt(self, text: str, response_model: Type[T]) -> T:
        """
        Send a prompt to the language model and parse the response into the specified model.

        Args:
            text (str): The prompt text.
            response_model (Type[T]): The Pydantic model to parse the response into.

        Returns:
            T: An instance of the response model containing the parsed response.
        """
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

# Parsers for the style guide and article
class StyleGuideProcessor:
    """
    Processes the style guide to extract requirements.
    """
    def __init__(
        self,
        style_guide_path: str,
        llm: 'LanguageModel',
        requirements_json_path: str = "requirements.json"
    ):
        self.style_guide_path = style_guide_path  # Path to the style guide
        self.llm = llm  # Language model instance
        self.requirements_json_path = requirements_json_path  # Path to save/load requirements

    async def process(self) -> StyleGuide:
        """
        Process the style guide to extract requirements.

        Returns:
            StyleGuide: An instance containing the extracted requirements.
        """
        try:
            # Try to load existing requirements from JSON
            requirements = await self._load_requirements_from_json()
            if requirements:
                logger.info("Requirements loaded from JSON file.")
                return StyleGuide(requirements=requirements)

            logger.info("Processing style guide to extract requirements.")
            # Read the style guide file asynchronously
            async with aiofiles.open(self.style_guide_path, 'r') as file:
                guide_text = await file.read()

            # Parse the style guide using wikitextparser
            parsed_guide = wtp.parse(guide_text)
            # Extract sections from the parsed guide
            sections = self._extract_sections(parsed_guide)
            # Extract requirements from the sections using the language model
            requirements = await self._extract_requirements(sections)
            # Save the requirements to JSON for future use
            await self._save_requirements_to_json(requirements)

            return StyleGuide(requirements=requirements)
        except Exception as e:
            logger.error(f"Error processing style guide: {e}", exc_info=True)
            raise StyleGuideProcessingError(f"Failed to process style guide: {str(e)}")

    async def _load_requirements_from_json(self) -> List[Requirement]:
        """
        Load requirements from a JSON file if it exists.

        Returns:
            List[Requirement]: A list of requirements.
        """
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
        """
        Save requirements to a JSON file.

        Args:
            requirements (List[Requirement]): The requirements to save.
        """
        try:
            requirements_data = [req.to_dict() for req in requirements]
            async with aiofiles.open(self.requirements_json_path, 'w') as f:
                await f.write(json.dumps(requirements_data, indent=2))
            logger.info(f"Requirements saved to {self.requirements_json_path}")
        except Exception as e:
            logger.error(f"Error saving requirements to JSON: {e}", exc_info=True)

    def _extract_sections(self, parsed_guide: wtp.WikiText) -> List[dict]:
        """
        Extract sections from the parsed style guide.

        Args:
            parsed_guide (wtp.WikiText): The parsed style guide.

        Returns:
            List[dict]: A list of sections with titles and content.
        """
        sections = []
        for section in parsed_guide.sections:
            if section.title:
                sections.append({
                    "title": section.title.strip(),
                    "content": section.contents.strip()
                })
        return sections

    async def _extract_requirements(self, sections: List[dict]) -> List[Requirement]:
        """
        Extract requirements from the style guide sections.

        Args:
            sections (List[dict]): The sections to process.

        Returns:
            List[Requirement]: A list of extracted requirements.
        """
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
                # Use the language model to extract requirements from the section
                response = await self.llm.prompt(prompt, List[Requirement])
                all_requirements.extend(response)
            except Exception as e:
                logger.error(f"Error processing section {section['title']}: {e}")
                # Append an error requirement to indicate the failure
                all_requirements.append(Requirement(
                    name=f"Error in section: {section['title']}",
                    description=f"Failed to process this section: {str(e)}",
                    applicable_sections=[],
                ))
        logger.info(f"Extracted {len(all_requirements)} requirements")
        return all_requirements

class ArticleParser:
    """
    Parses the article markdown file into an ArticleNode structure.
    """
    def __init__(self):
        self.ast_renderer = ASTRenderer()  # Renderer to convert markdown to AST
        self.current_id = 0  # Counter for unique section IDs

    def parse(self, filename):
        """
        Parse the markdown article file.

        Args:
            filename (str): The path to the markdown file.

        Returns:
            ArticleNode: The root node of the parsed article.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        # Render the markdown content to an AST
        ast_json = self.ast_renderer.render(mistletoe.Document(content))
        ast = json.loads(ast_json)

        # Process the AST to build the article structure
        return self._process_ast(ast)

    def _process_ast(self, ast: Dict[str, Any]) -> ArticleNode:
        """
        Process the AST recursively to build the article structure.

        Args:
            ast (Dict[str, Any]): The AST of the article.

        Returns:
            ArticleNode: The root node of the article.
        """
        # Initialize the root node
        root = ArticleNode(id="root", title="Overall Title", content="", level=0)
        if 'children' in ast:
            self._process_children(ast['children'], root)
        else:
            raise ArticleParsingError("AST structure is not as expected")
        return root

    def _process_children(self, children, root: ArticleNode):
        """
        Process child nodes of the AST.

        Args:
            children (List[Dict[str, Any]]): The child nodes.
            root (ArticleNode): The current parent node.
        """
        section_stack = [root]  # Stack to keep track of section hierarchy

        for item in children:
            item_type = item.get('type')
            if item_type == 'Heading':
                # Process heading to create a new section
                heading_level = item.get('level', 1)
                new_section = self._create_section(item, heading_level)

                # Adjust the section stack based on heading levels
                while section_stack and section_stack[-1].level >= heading_level:
                    section_stack.pop()
                section_stack[-1].children.append(new_section)
                section_stack.append(new_section)
            elif item_type in ['Paragraph', 'List']:
                # Append content to the current section
                current_section = section_stack[-1]
                current_section.content += self._extract_content(item)
            elif item.get('children'):
                # Recursively process child nodes
                self._process_children(item['children'], root=section_stack[-1])

    def _create_section(self, heading: Dict[str, Any], level: int) -> ArticleNode:
        """
        Create a new article section node from a heading.

        Args:
            heading (Dict[str, Any]): The heading node.
            level (int): The heading level.

        Returns:
            ArticleNode: The new section node.
        """
        self.current_id += 1  # Increment the section ID counter
        title = self._extract_text(heading.get('children', []))  # Extract the heading text
        return ArticleNode(
            id=f"section_{self.current_id}",
            title=title,
            content="",
            level=level,
        )

    def _extract_content(self, node: Dict[str, Any]) -> str:
        """
        Extract content from a paragraph or list node.

        Args:
            node (Dict[str, Any]): The node to extract content from.

        Returns:
            str: The extracted content.
        """
        if node['type'] == 'Paragraph':
            return self._extract_text(node.get('children', [])) + "\n\n"
        elif node['type'] == 'List':
            return self._extract_list(node) + "\n\n"
        return ""

    def _extract_text(self, children) -> str:
        """
        Recursively extract text from child nodes.

        Args:
            children (List[Dict[str, Any]]): The child nodes.

        Returns:
            str: The extracted text.
        """
        texts = []
        for child in children:
            if child['type'] == 'RawText':
                texts.append(child['content'])
            elif 'children' in child:
                texts.append(self._extract_text(child['children']))
        return ' '.join(texts)

    def _extract_list(self, node: Dict[str, Any]) -> str:
        """
        Extract content from a list node.

        Args:
            node (Dict[str, Any]): The list node.

        Returns:
            str: The extracted list content.
        """
        list_items = []
        for item in node.get('children', []):
            item_text = self._extract_text(item.get('children', []))
            list_items.append(f"- {item_text}")
        return "\n".join(list_items)

    def serialize_article_structure(self, node: ArticleNode) -> dict:
        """
        Serialize the article structure to a dictionary.

        Args:
            node (ArticleNode): The root node of the article.

        Returns:
            dict: The serialized article structure.
        """
        return {
            "id": node.id,
            "title": node.title,
            "content": node.content,
            "level": node.level,
            "children": [self.serialize_article_structure(child) for child in node.children]
        }

class ArticleEvaluator:
    """
    Evaluates the article against the style guide requirements.
    """
    def __init__(self, llm: LanguageModel, requirements: List[Requirement]):
        self.llm = llm  # Language model instance
        self.requirements = requirements  # List of style guide requirements

    async def evaluate(self, article_node: ArticleNode) -> List[EvaluatedSection]:
        """
        Evaluate the article sections.

        Args:
            article_node (ArticleNode): The root node of the article.

        Returns:
            List[EvaluatedSection]: The evaluation results.
        """
        try:
            # Prepare the sections for evaluation
            sections_to_evaluate = self._prepare_sections_for_evaluation(article_node)
            evaluated_sections = []
            for section in sections_to_evaluate:
                # Generate the prompt for the language model
                prompt = self._generate_evaluation_prompt(section)
                # Get the evaluation result from the language model
                result = await self.llm.prompt(prompt, EvaluatedSection)
                evaluated_sections.append(result)
            return evaluated_sections
        except Exception as e:
            logger.error(f"Error evaluating article: {e}", exc_info=True)
            raise EvaluationError(f"Failed to evaluate article: {str(e)}")

    def _prepare_sections_for_evaluation(self, article_node: ArticleNode) -> List[Dict[str, Any]]:
        """
        Prepare the article sections for evaluation.

        Args:
            article_node (ArticleNode): The root node of the article.

        Returns:
            List[Dict[str, Any]]: A list of sections to evaluate.
        """
        sections_to_evaluate = []
        self._collect_sections(article_node, sections_to_evaluate)
        return sections_to_evaluate

    def _collect_sections(self, node: ArticleNode, sections: List[Dict[str, Any]]):
        """
        Recursively collect sections from the article.

        Args:
            node (ArticleNode): The current node.
            sections (List[Dict[str, Any]]): The list to collect sections into.
        """
        sections.append({
            "section_id": node.id,
            "title": node.title,
            "content": node.content,
        })
        for child in node.children:
            self._collect_sections(child, sections)

    def _generate_evaluation_prompt(self, section: Dict[str, Any]) -> str:
        """
        Generate a prompt for evaluating a section.

        Args:
            section (Dict[str, Any]): The section to evaluate.

        Returns:
            str: The generated prompt.
        """
        # Prepare the requirements for inclusion in the prompt
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
    """
    Main class for the Omnipedia application.
    """
    def __init__(
        self,
        style_guide_path: str,
        language_model: LanguageModel,
        requirements_path: str
    ):
        self.llm = language_model  # Language model instance
        self.style_guide_processor = StyleGuideProcessor(style_guide_path, language_model, requirements_path)
        self.style_guide = None  # To hold the processed style guide
        self.evaluator = None  # To hold the article evaluator
        self.article_parser = ArticleParser()  # Article parser instance

    async def initialize(self):
        """
        Initialize the application by processing the style guide.
        """
        try:
            # Process the style guide to extract requirements
            self.style_guide = await self.style_guide_processor.process()
            # Initialize the evaluator with the extracted requirements
            self.evaluator = ArticleEvaluator(self.llm, self.style_guide.requirements)
        except Exception as e:
            logger.error(f"Error initializing Omnipedia: {e}", exc_info=True)
            raise

    async def evaluate_article(self, article_path: str):
        """
        Evaluate an article.

        Args:
            article_path (str): The path to the article file.

        Returns:
            List[EvaluatedSection]: The evaluation results.
        """
        try:
            # Parse the article to get the article structure
            article_node = self.article_parser.parse(article_path)
            # Serialize and save the article structure to a JSON file
            article_structure = self.article_parser.serialize_article_structure(article_node)
            with open("article_structure.json", 'w') as file:
                json.dump(article_structure, file, indent=2)
            logger.info("Article structure saved to article_structure.json")

            # Evaluate the article using the evaluator
            return await self.evaluator.evaluate(article_node)
        except Exception as e:
            logger.error(f"Error evaluating article: {e}", exc_info=True)
            raise

# Main function to run the application
async def main():
    """
    Main entry point for the application.
    """
    try:
        # Initialize the language model
        language_model = LanguageModel(settings.DEFAULT_MODEL_IDENTIFIER)
        # Create an instance of Omnipedia
        omnipedia = Omnipedia(settings.STYLE_GUIDE_PATH, language_model, "requirements.json")

        # Initialize the application (process the style guide)
        await omnipedia.initialize()

        # Evaluate the article specified in the settings
        evaluated_sections = await omnipedia.evaluate_article(settings.ARTICLE_PATH)
        print("Evaluation complete.")
        # Print the evaluation results
        for section in evaluated_sections:
            print(f"Section: {section.section}")
            print(f"Score: {section.score}")
            print(f"Feedback: {section.feedback}")
            print("---")
        
        # Convert EvaluatedSection objects to dictionaries
        serializable_sections = [section.to_dict() for section in evaluated_sections]
        
        # Save the serializable sections to JSON
        with open("evaluated_sections.json", 'w') as file:
            json.dump(serializable_sections, file, indent=2)

        print("Evaluation results saved to evaluated_sections.json")

    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"An error occurred: {e}", exc_info=True)
