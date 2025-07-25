# AI Foundation Seminar Tutorial

This repository contains materials for an AI Foundation seminar, covering fundamental concepts of AI, Machine Learning, Deep Learning, Natural Language Processing, Transformers, and Large Language Models (LLMs), including agent-based approaches and related services. It is designed to provide hands-on experience, primarily utilizing Jupyter Notebooks.

## Repository Structure

The repository is organized into several folders, each focusing on a specific area of AI, along with supplementary documents:

  * `1_AX_trend`: AI Transformation trends.
  * `2_ML_basic`: Basic Machine Learning concepts.
  * `3_DL_foundation`: Deep Learning foundations.
  * `4_NLP`: Natural Language Processing.
  * `5_transformer`: Transformer models.
  * `6_LLM_agent_vibe`: LLM Agent concepts and vibe coding.
  * `7_service`: AI services related topics.
  * `8_AX_reference`: AI Transformation references.
  * `AI_foundation_and_trend.pdf`: A PDF document possibly detailing AI foundations and trends.
  * `AI_foundation_syllabus.docx`: The syllabus for the AI Foundation seminar.
  * `LICENSE`: Contains the MIT License information.
  * `LLM-lesson-plan.docx`: A lesson plan related to LLMs.
  * `README.md`: This README file.

## Getting Started: Development Environment Setup

This section outlines the prerequisites and installation steps to prepare your working environment for a smooth hands-on experience. All materials can be downloaded from this repository.

### 1\. Account Sign-up

Visit the following websites to sign up for accounts. [cite\_start]Some services are paid, and it's recommended to set usage limits or subscribe within a certain budget (e.g., $20) for initial experience[cite: 20, 31, 32]:

  * [cite\_start]**Colab Pro:** [https://colab.research.google.com/signup](https://colab.research.google.com/signup) (Paid) [cite: 21]
  * [cite\_start]**ChatGPT:** (Paid) [cite: 22]
  * [cite\_start]**ChatGPT API (Pay as you go):** [https://platform.openai.com/settings/organization/billing/overview](https://platform.openai.com/settings/organization/billing/overview) (Paid, set a limit, e.g., $8) [cite: 23]
  * [cite\_start]**Claude:** [https://claude.ai/](https://claude.ai/) (Free) [cite: 24]
  * [cite\_start]**GitHub:** [https://github.com/](https://github.com/) (Free) [cite: 25]
  * [cite\_start]**GitHub Copilot:** [https://github.com/features/copilot/plans](https://github.com/features/copilot/plans) (Paid, $10/month) [cite: 26]
  * [cite\_start]**Hugging Face:** [https://huggingface.co](https://huggingface.co) (Free) [cite: 27] - [cite\_start]Required for LLM, Transformer, and Stable Diffusion models [cite: 34]
  * [cite\_start]**Hugging Face API Token:** [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Free) [cite: 28]
  * [cite\_start]**Stable Diffusion - Kling:** [https://app.klingai.com/global/membership/membership-plan](https://app.klingai.com/global/membership/membership-plan) (Paid, $6.99/month for Standard) [cite: 29, 32]
  * [cite\_start]**Figma:** [https://www.figma.com](https://www.figma.com) (Optional) [cite: 30]

[cite\_start]**Note:** Record your IDs and Passwords for each account, as they will be used during tool installation[cite: 33].

### 2\. Project Development Service Accounts

[cite\_start]The following AI services are recommended for project development[cite: 35, 36]:

  * [cite\_start]**Gemini API:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (Google AI Key for Gemini) [cite: 37]
  * [cite\_start]**SerpAPI:** [https://serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) (Google Search API) [cite: 38]
  * [cite\_start]**Tavily:** [https://app.tavily.com/home](https://app.tavily.com/home) (Web Search) [cite: 39]
  * [cite\_start]**Weights & Biases (wandb):** [https://docs.wandb.ai/quickstart/](https://docs.wandb.ai/quickstart/) (Model training & fine-tuning monitoring/logging tool) [cite: 40]
  * [cite\_start]**LangSmith:** [https://docs.smith.langchain.com/administration/how\_to\_guides/organization\_management/create\_account\_api\_key](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key) (LangChain logging & debug tool) [cite: 41]
  * [cite\_start]**Visily AI:** [https://app.visily.ai](https://app.visily.ai) (AI planning tool) [cite: 42]

### 3\. Colab Setup

1.  [cite\_start]Open `colab-env.ipynb` from the following link in Google Colab[cite: 44, 45]: [https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/1-1.prepare](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/1-1.prepare)
2.  [cite\_start]Connect to your Google Drive to save practice files[cite: 46].
3.  [cite\_start]Set up the API keys you created earlier in the "Secrets" menu of your Colab account, as shown in the provided image[cite: 47].

### 4\. Development Tools Installation

[cite\_start]It is recommended to install these tools before the hands-on sessions to save time[cite: 49, 50]. [cite\_start]Please install stable versions, as the latest versions may cause package installation errors[cite: 51]. [cite\_start]Ensure you check the "Add to PATH" option during installation if available[cite: 53, 54].

  * [cite\_start]**Python (Recommended 3.11):** [https://www.python.org](https://www.python.org) [cite: 56, 57, 110]
      * [cite\_start]For Mac users, refer to the Python installation guide: [https://www.youtube.com](https://www.youtube.com) [cite: 58, 111]
      * [cite\_start]Verify installation by running `python --version` in the terminal[cite: 59].
  * [cite\_start]**NVIDIA Driver (for NVIDIA GPU users):** [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx) [cite: 60, 61, 108]
      * [cite\_start]Skip GPU-related steps if you do not have an NVIDIA GPU[cite: 55].
      * [cite\_start]Verify installation by running `nvidia-smi` in the terminal[cite: 62].
  * [cite\_start]**CUDA Toolkit (for NVIDIA GPU users):** [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) [cite: 63, 64, 109]
      * [cite\_start]Check GPU and driver compatibility during installation[cite: 65].
      * [cite\_start]Add CUDA path to environment variables[cite: 66].
      * [cite\_start]Verify CUDA version by running `nvcc -V` in the terminal[cite: 77].
  * [cite\_start]**GitHub Tools:** [https://docs.github.com/ko/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop](https://docs.github.com/ko/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop) [cite: 67, 68]
  * [cite\_start]**Anaconda (Recommended 24.0 or higher):** [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/) [cite: 70, 71, 72, 112]
  * [cite\_start]**PyTorch Library:** Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install the CPU version or the GPU version compatible with your CUDA driver[cite: 73, 74, 75, 76, 77, 114].
  * **Python Packages (using Anaconda virtual environment):**
      * [cite\_start]Create a conda virtual environment named `venv_lmm` with Python 3.11[cite: 79, 80]:
        ```bash
        conda create --name venv_lmm python=3.11
        ```
      * [cite\_start]Activate the environment[cite: 79, 81]:
        ```bash
        conda activate venv_lmm
        ```
      * [cite\_start]Install core packages[cite: 79, 82, 83]:
        ```bash
        pip install pandas numpy
        pip install ollama openai transformers huggingface_hub langchain
        ```
  * [cite\_start]**Docker (Optional):** Required for container-based operations[cite: 84, 85]. [cite\_start]Visit [https://www.docker.com/get-started/](https://www.docker.com/get-started/) to install[cite: 86].
  * [cite\_start]**Ollama:** Required for local LLM AI tools[cite: 87, 88]. [cite\_start]Visit [https://www.ollama.com/](https://www.ollama.com/) to install[cite: 89].
  * **Code Editors & IDEs:**
      * [cite\_start]**Sublime Text:** [https://www.sublimetext.com/](https://www.sublimetext.com/) [cite: 90, 91]
      * [cite\_start]**Visual Studio Code (VS Code):** [https://code.visualstudio.com/download](https://code.visualstudio.com/download) [cite: 90, 92]
          * [cite\_start]Refer to the provided video for detailed installation and Python extension setup[cite: 92].
          * [cite\_start]After VS Code installation, install GitHub Copilot and GitHub Copilot Chat[cite: 93]. [cite\_start]Refer to the provided videos for guidance[cite: 93, 94, 95].
  * [cite\_start]**Claude Desktop:** [https://claude.ai/download](https://claude.ai/download) [cite: 96, 97]

### 5\. Other Tools (Optional)

[cite\_start]Install these if time permits[cite: 98, 99]:

  * [cite\_start]**Blender:** For LLM-based graphic modeling[cite: 100, 101]. [cite\_start]Visit [https://www.blender.org/download/](https://www.blender.org/download/) to install[cite: 102].
  * [cite\_start]**DaVinci Resolve 20 Public Beta:** [https://www.blackmagicdesign.com/products/davinciresolve](https://www.blackmagicdesign.com/products/davinciresolve) (Optional) [cite: 103, 104]

## Usage

Once the development environment is set up, you can navigate through the Jupyter notebooks (.ipynb files) within the repository's folders (e.g., `1_AX_trend`, `2_ML_basic`, `3_DL_foundation`, etc.) to explore various AI topics and hands-on examples.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For inquiries, please refer to the project's GitHub page.
