# Omnipedia README

## Introduction

Welcome to **Omnipedia**, a comprehensive tool designed to parse and evaluate articles against a predefined style guide using advanced language models. This README provides a thorough explanation of the codebase, including file names, class structures (ontology), and a high-level system diagram to help you get up to speed as quickly as possible.

## Overview

Omnipedia automates the process of:

1. **Processing a Style Guide**: Parses the style guide to extract actionable writing requirements.
2. **Parsing Articles**: Converts articles into a structured format for analysis.
3. **Evaluating Articles**: Assesses articles against the extracted style guide requirements using a language model (e.g., OpenAI's GPT-3.5 Turbo).
4. **Providing Feedback**: Generates scores and feedback for each section of the article based on adherence to the style guide.

## File Structure

The project consists of a single main Python script and several auxiliary files:

- **main.py**: The primary script containing all classes and functions.
- **styleguide.txt**: The style guide document to be processed.
- **article.md**: The article to be evaluated.
- **requirements.json**: Cached requirements extracted from the style guide.
- **article_structure.json**: Serialized structure of the parsed article.
- **.env**: Environment file containing the `OPENAI_API_KEY`.

## Dependencies

The project relies on the following Python packages:

- `os`
- `logging`
- `json`
- `asyncio`
- `aiofiles`
- `wikitextparser`
- `pydantic`
- `pydantic-settings`
- `instructor` (Assumed to be a custom or third-party package for interfacing with OpenAI)
- `openai`
- `mistletoe`

Ensure all dependencies are installed:

```bash
pip install aiofiles wikitextparser pydantic pydantic-settings openai mistletoe
```

## Settings and Configuration

The `Settings` class handles configuration using environment variables and defaults:

- **DEFAULT_MODEL_IDENTIFIER**: The language model to use (default: `"gpt-3.5-turbo"`).
- **MAX_TOKENS**: Maximum tokens for the language model response.
- **STYLE_GUIDE_PATH**: Path to the style guide file.
- **ARTICLE_PATH**: Path to the article file.
- **LOG_LEVEL**: Logging level (default: `"INFO"`).
- **OPENAI_API_KEY**: API key for OpenAI, loaded from the `.env` file or environment variable.

## Exception Handling

Custom exceptions are defined for better error management:

- **OmnipediaError**: Base exception class.
  - **StyleGuideProcessingError**
  - **ArticleParsingError**
  - **EvaluationError**
  - **LanguageModelError**

## Data Models (Ontology)

The code utilizes Pydantic's `BaseModel` for data validation and management.

### Classes and Their Relationships

```plaintext
+------------------------+
|       Requirement      |
+------------------------+
| name                   |
| description            |
| applicable_sections    |
+------------------------+

+------------+
| StyleGuide |
+------------+
| requirements -> List[Requirement]
+------------+

+-------------+
| ArticleNode |
+-------------+
| id          |
| title       |
| content     |
| level       |
| children    -> List[ArticleNode]
+-------------+

+-------------------+
|  EvaluatedSection |
+-------------------+
| section           |
| score             |
| feedback          |
| adherent_requirements |
| templates         |
| wikilinks         |
| external_links    |
| list_items        |
+-------------------+
```

## Core Components

### 1. LanguageModel

Handles communication with the language model API.

- **Methods**:
  - `prompt(text: str, response_model: Type[T]) -> T`: Sends a prompt to the language model and returns a response of type `T`.

### 2. StyleGuideProcessor

Processes the style guide to extract requirements.

- **Methods**:
  - `process() -> StyleGuide`: Main method to process the style guide.
  - `_load_requirements_from_json()`: Loads requirements from `requirements.json` if available.
  - `_save_requirements_to_json(requirements)`: Saves requirements to `requirements.json`.
  - `_extract_sections(parsed_guide)`: Extracts sections from the parsed guide.
  - `_extract_requirements(sections)`: Extracts requirements from sections using the language model.

### 3. ArticleParser

Parses the article into a structured format (`ArticleNode` tree).

- **Methods**:
  - `parse(filename) -> ArticleNode`: Parses the article file.
  - `_process_ast(ast)`: Processes the abstract syntax tree.
  - `_process_children(children, root)`: Processes child nodes recursively.
  - `_create_section(heading, level)`: Creates a new `ArticleNode`.
  - `_extract_content(node)`: Extracts content from nodes.
  - `_extract_text(children)`: Extracts text from child nodes.
  - `_extract_list(node)`: Extracts list items.
  - `serialize_article_structure(node)`: Serializes the article structure to JSON.

### 4. ArticleEvaluator

Evaluates the parsed article against the style guide requirements.

- **Methods**:
  - `evaluate(article_node) -> List[EvaluatedSection]`: Evaluates each section.
  - `_prepare_sections_for_evaluation(article_node)`: Prepares sections for evaluation.
  - `_collect_sections(node, sections)`: Collects sections recursively.
  - `_generate_evaluation_prompt(section)`: Generates prompts for the language model.

### 5. Omnipedia (Main Application)

Integrates all components to process and evaluate articles.

- **Methods**:
  - `__init__(style_guide_path, language_model, requirements_path)`: Initializes components.
  - `initialize()`: Processes the style guide and prepares the evaluator.
  - `evaluate_article(article_path)`: Parses and evaluates the article.

## Workflow

1. **Initialization**:

   - Load settings and initialize logging.
   - Set up the language model with the specified API key and model identifier.
   - Create an instance of `Omnipedia`.

2. **Style Guide Processing**:

   - Use `StyleGuideProcessor` to parse the style guide (`styleguide.txt`).
   - Extract requirements and save them to `requirements.json`.

3. **Article Parsing**:

   - Use `ArticleParser` to parse the article (`article.md`).
   - Generate an `ArticleNode` tree representing the article's structure.
   - Serialize the structure to `article_structure.json`.

4. **Article Evaluation**:

   - Use `ArticleEvaluator` to evaluate each section of the article.
   - Generate scores and feedback based on adherence to the style guide.

5. **Output**:
   - Display the evaluation results for each section.

## High-Level System Diagram

```plaintext
+----------------+      +--------------------+      +-----------------+
|                |      |                    |      |                 |
|  styleguide.txt| ---> |StyleGuideProcessor | ---> | requirements.json|
|                |      |                    |      |                 |
+----------------+      +--------------------+      +-----------------+
                                                           |
                                                           v
+----------------+      +---------------+      +-----------------+
|                |      |               |      |                 |
|   article.md   | ---> | ArticleParser | ---> | ArticleNode Tree|
|                |      |               |      |                 |
+----------------+      +---------------+      +-----------------+
                                                           |
                                                           v
                                        +--------------------------------+
                                        |                                |
                                        |        ArticleEvaluator        |
                                        |    (uses LanguageModel API)    |
                                        |                                |
                                        +--------------------------------+
                                                           |
                                                           v
                                                +--------------------+
                                                |                    |
                                                | Evaluation Results |
                                                |                    |
                                                +--------------------+
```

## How to Run

1. **Set Up Environment**:

   - Ensure all dependencies are installed.
   - Place your OpenAI API key in a `.env` file or set it as an environment variable `OPENAI_API_KEY`.

2. **Prepare Files**:

   - Place your style guide in `styleguide.txt`.
   - Place the article to be evaluated in `article.md`.

3. **Execute the Script**:

   ```bash
   python main.py
   ```

4. **View Results**:

   - Evaluation results will be displayed in the console.
   - Serialized article structure saved to `article_structure.json`.

## Conclusion

Omnipedia automates the evaluation of articles against a style guide using advanced language models. By parsing both the style guide and the article, it provides detailed feedback on adherence, helping improve the quality and consistency of written content.

---

**Note**: Ensure that the `instructor` module is correctly installed and configured, as it appears to be a custom or less common library for interfacing with the OpenAI API.
