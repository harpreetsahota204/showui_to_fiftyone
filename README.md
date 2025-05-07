# Parsing the datasets from ShowUI to FiftyOne format

<img src="showui_web.gif">
⬆️ Example of the Web dataset parsed into FiftyOne format

## Download the parsed datasets from Hugging Face

  If you haven''t already, install FiftyOne:

  ```bash
  pip install -U fiftyone
  ```

  ## Usage

  ```python
  import fiftyone as fo
  from fiftyone.utils.huggingface import load_from_hub

  # Load the web dataset
  web_dataset = load_from_hub("Voxel51/ShowUI_Web")
  
  # Load the desktop dataset
  desktopdataset = load_from_hub("Voxel51/ShowUI_desktop")
  
  # Launch the App with either dataset

  session = fo.launch_app(web_dataset)
  ```

### Dataset Description

The ShowUI dataset is a carefully curated collection of GUI screenshots and element annotations created specifically for training vision-language-action models for GUI visual agents. It includes two key splits:

**Web Split:** A custom-collected corpus of 22,000 web interface screenshots with 576,000 annotated UI elements across 22 representative website scenarios (including Airbnb, Booking, AMD, Apple, etc.). This split deliberately focuses on visually rich interactive elements (buttons, checkboxes, etc.) while filtering out static text components, which comprised about 40% of the original annotations.

**Desktop Split:** An augmented version of OmniAct Desktop data, containing 100 screenshots with 8,000 element annotations from 15 different software applications across iOS, Windows, and Linux desktop environments. The original dataset's 2,000 basic element annotations were enhanced through GPT-4o-assisted annotation to generate three distinct query types for each element: appearance descriptions, spatial relationship descriptions, and intentional queries.

- **Curated by:** Show Lab, National University of Singapore and Microsoft
- **Language(s) (NLP):** en
- **License:** Not specified in the paper

### Dataset Sources

- **Repository:** https://github.com/showlab/ShowUI (main project repository)
- **HF Dataset Repository:** https://huggingface.co/datasets/showlab/ShowUI-desktop and https://huggingface.co/datasets/showlab/ShowUI-web
- **Paper:** https://arxiv.org/abs/2411.17465


## Uses

### Direct Use

The dataset is designed for training vision-language-action models for GUI visual agents operating across multiple platforms:

**Common Uses:**
- Training models to visually identify UI elements
- Element grounding (mapping textual queries to visual elements)
- Supporting development of cross-platform UI navigation systems

**Web Split Specific:**
- Learning interactive web element identification
- Web navigation tasks requiring visual element recognition
- Distinguishing between different types of web UI components

**Desktop Split Specific:**
- Navigating desktop software interfaces
- Understanding diverse query formulations for the same UI element
- Mapping user intent to appropriate UI elements

### Out-of-Scope Use

While not explicitly stated in the paper, this dataset would likely be unsuitable for:
- Training models for mobile interfaces exclusively
- General image understanding unrelated to UI navigation
- Applications requiring user data or personalized interfaces
- Training text-recognition systems (especially for the Web split, which deliberately excludes static text)

## Dataset Structure **Web Split:**
- 22,000 web screenshots across 22 representative website scenarios
- 576,000 annotated UI elements (filtered from an original 926,000)
- An average of 26 elements per screenshot
- Focuses on interactive visual components (buttons, checkboxes, etc.)
- Deliberately excludes static text elements

### FiftyOne Dataset Structure

**Basic Info:** 21,988 web UI screenshots with interaction annotations

**Core Fields:**
- `instructions`: ListField(StringField) - List of potential text instructions or UI element texts
- `detections`: EmbeddedDocumentField(Detections) containing multiple Detection objects:
  - `label`: Element type (e.g., "ListItem")
  - `bounding_box`: A list of relative bounding box coordinates in [0, 1] in the following format: `[<top-left-x>, <top-left-y>, <width>, <height>]`
  - `text`: Text content of element
- `keypoints`: EmbeddedDocumentField(Keypoints) containing interaction points:
  - `label`: Element type (e.g., "ListItem")
  - `points`:  A list of `(x, y)` keypoints in `[0, 1] x [0, 1]`
  - `text`: Text content associated with the interaction point

The dataset captures web interface elements and interaction points with detailed text annotations for web interaction research. Each element has both its bounding box coordinates and a corresponding interaction point, allowing for both element detection and precise interaction targeting.


## Dataset Structure **Desktop Split:**
- 100 desktop screenshots from 15 different applications across iOS, Windows, and Linux
- 8,000 element annotations (expanded from original 2,000)
- Each element contains four types of annotations:
  1. Original element name from OmniAct
  2. Appearance description (visual characteristics)
  3. Spatial relationship description (position relative to other UI elements)
  4. Intentional query (purpose or user intent)

### FiftyOne Dataset Structure
**Basic Info:** 7,496 desktop UI screenshots with interaction annotations

**Core Fields:**
- `instruction`: StringField - Task description in structured format
- `action_detections`: EmbeddedDocumentField(Detection) containing target element:
  - `label`: Action type (e.g., "action")
  - `bounding_box`: A list of relative bounding box coordinates in [0, 1] in the following format: `[<top-left-x>, <top-left-y>, <width>, <height>]`
- `action_keypoints`: EmbeddedDocumentField(Keypoints) containing interaction points:
  - `label`: Action type (e.g., "action")
  - `points`: A list of `(x, y)` keypoints in `[0, 1] x [0, 1]`
- `query_type`: EmbeddedDocumentField(Classification) - Type of instruction query (e.g., "original")
- `interfaces`: EmbeddedDocumentField(Classification) - Interface application type (e.g., "audible")

The dataset captures desktop application interactions with detailed UI element annotations and action points for interface interaction research across different desktop applications.

## Dataset Creation

### Curation Rationale

The authors developed this dataset based on careful analysis of existing GUI datasets and their limitations:

**Web Split:** Most existing web datasets contain a high proportion of static text elements (around 40%) that provide limited value for training visual GUI agents, since modern Vision-Language Models already possess strong OCR capabilities. The authors focused on collecting visually distinctive interactive elements that would better enable models to learn UI navigation skills.

**Desktop Split:** Desktop data was identified as particularly valuable but challenging to collect automatically. The original OmniAct dataset provided manually annotated elements, but with limited annotation diversity (only element names). To improve model training for desktop UI navigation, they needed more diverse query types that reflected how users might describe or search for UI elements.

### Source Data

#### Data Collection and Processing

**Web Split:**
1. Developed a custom parser using PyAutoGUI 
2. Selected 22 representative website scenarios (including Airbnb, Booking, AMD, Apple, etc.)
3. Collected multiple screenshots per scenario to maximize annotation coverage
4. Initially gathered 926,000 element annotations across 22,000 screenshots
5. Filtered out elements classified as static text, retaining 576,000 visually interactive elements
6. Focused on elements tagged with categories like "Button" or "Checkbox"

**Desktop Split:**
1. Started with the existing OmniAct Desktop dataset (100 screenshots, 2,000 elements)
2. Identified the original elements and their bounding boxes
3. Created visual prompts with red bounding boxes highlighting target elements
4. Used GPT-4o with a specific prompt template (detailed in Appendix B.2 of the paper)
5. Generated three distinct query types per element
6. This process expanded the dataset to 8,000 elements without requiring additional screenshots

#### Who are the source data producers?

**Web Split:** The data was collected from 22 publicly accessible websites across various domains (e-commerce, technology, travel, etc.).

**Desktop Split:** The original data came from OmniAct containing screenshots from 15 different software applications across iOS, Windows, and Linux platforms.

The specific annotations were produced by the authors of the ShowUI paper (Show Lab, National University of Singapore and Microsoft).

### Annotations

#### Annotation process

**Web Split:**
- Used a custom parser built with PyAutoGUI
- Classified elements by type (Button, Checkbox, etc. versus static text)
- Filtered elements based on these classifications

**Desktop Split:**
1. Extracted original element names from OmniAct
2. Prompted GPT-4o with screenshots containing highlighted target elements
3. Used a specific prompt template requesting three types of descriptions:
   - Appearance-based (15 words or fewer)
   - Spatial relationship-based (15 words or fewer)  
   - Situation/intention-based (15 words or fewer)
4. Formatted responses in JSON structure

#### Who are the annotators?

**Web Split:** Annotations appear to have been programmatically generated through the custom parser developed by the authors.

**Desktop Split:** The enhanced annotations were produced using GPT-4o, prompted by the authors of the ShowUI paper.

#### Personal and Sensitive Information

Not explicitly addressed in the paper, but the dataset appears to focus on public website interfaces and standard desktop applications rather than personal content.

## Bias, Risks, and Limitations

The paper doesn't explicitly discuss biases or limitations, but potential limitations include:

**Web Split:**
- Limited to 22 website scenarios, which may not represent the full diversity of web interfaces
- Filtering out static text could limit the model's ability to handle text-heavy interfaces
- Potential overrepresentation of popular or mainstream websites

**Desktop Split:**
- Limited to 15 software applications with only 100 screenshots
- Generated descriptions might reflect biases present in GPT-4o
- May not fully represent all desktop UI paradigms or element types

**Common Limitations:**
- Limited to English language interfaces
- May not capture the full range of accessibility features or alternative UI designs
- Imbalance between splits (22,000 web screenshots vs. 100 desktop screenshots)

### Recommendations

Users should be aware of the deliberate design choices in each split:

- The Web split intentionally excludes static text elements, making it complementary to text-focused datasets
- The Desktop split is relatively small but enriched with diverse query formulations
- For comprehensive GUI navigation models, both splits should be used with appropriate balancing strategies (as demonstrated in the paper's ablation studies)
- Additional data sources may be needed for specific platforms or application types not well-represented in these splits

## Citation 

**BibTeX:**
```bibtex
@misc{lin2024showui,
      title={ShowUI: One Vision-Language-Action Model for GUI Visual Agent}, 
      author={Kevin Qinghong Lin and Linjie Li and Difei Gao and Zhengyuan Yang and Shiwei Wu and Zechen Bai and Weixian Lei and Lijuan Wang and Mike Zheng Shou},
      year={2024},
      eprint={2411.17465},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17465}, 
}
```

**APA:**
Lin, K. Q., Li, L., Gao, D., Yang, Z., Wu, S., Bai, Z., Lei, S. W., Wang, L., & Shou, M. Z. (2024). ShowUI: One Vision-Language-Action Model for GUI Visual Agent. arXiv preprint arXiv:2411.17465.
