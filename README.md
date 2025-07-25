<img width="925" height="723" alt="image" src="https://github.com/user-attachments/assets/12698895-0ead-43c4-9a3a-92fbc90ed3f8" /># AI Foundation Seminar Tutorial

This repository contains materials for an AI Foundation seminar, covering fundamental concepts of AI, Machine Learning, Deep Learning, Natural Language Processing, Transformers, and Large Language Models (LLMs), including agent-based approaches and related services. It is designed to provide hands-on experience, primarily utilizing Jupyter Notebooks. In reference, you can learn [How to develop AI agent with LLM](https://github.com/mac999/LLM-RAG-Agent-Tutorial), [computer vision with deep learning](https://github.com/mac999/computer_vision_deeplearning) and [AI for media art](https://github.com/mac999/llm-media-art-demo).
<br/><br/>
<img height="200" src="https://github.com/mac999/AI_foundation_tutorial/blob/main/3_DL_foundation/net.png"/>
<img height="200" src="https://github.com/mac999/AI_foundation_tutorial/blob/main/5_transformer/transformer-architecture.PNG"/>

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
* `AI_foundation_and_trend.pdf`: A PDF slide document possibly detailing AI foundations and trends.
* `AI_foundation_syllabus.docx`: The syllabus for the AI Foundation seminar.
* `LICENSE`: Contains the MIT License information.
* `LLM-lesson-plan.docx`: A lesson plan related to LLMs.
* `README.md`: This README file.

## Getting Started: Development Environment Setup

This section outlines the prerequisites and installation steps to prepare your working environment for a smooth hands-on experience. All materials can be downloaded from this repository.

### 1. Account Sign-up

Visit the following websites to sign up for accounts. Some services are paid, and it's recommended to set usage limits or subscribe within a certain budget (e.g., $20) for initial experience:

* **Colab Pro:** [https://colab.research.google.com/signup](https://colab.research.google.com/signup) (Paid)
* **ChatGPT:** (Paid)
* **ChatGPT API (Pay as you go):** [https://platform.openai.com/settings/organization/billing/overview](https://platform.openai.com/settings/organization/billing/overview) (Paid, set a limit, e.g., $8)
* **Claude:** [https://claude.ai/](https://claude.ai/) (Free)
* **GitHub:** [https://github.com/](https://github.com/) (Free)
* **GitHub Copilot:** [https://github.com/features/copilot/plans](https://github.com/features/copilot/plans) (Paid, $10/month)
* **Hugging Face:** [https://huggingface.co](https://huggingface.co) (Free) - Required for LLM, Transformer, and Stable Diffusion models
* **Hugging Face API Token:** [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Free)
* **Stable Diffusion - Kling:** [https://app.klingai.com/global/membership/membership-plan](https://app.klingai.com/global/membership/membership-plan) (Paid, $6.99/month for Standard)
* **Figma:** [https://www.figma.com](https://www.figma.com) (Optional)

**Note:** Record your IDs and Passwords for each account, as they will be used during tool installation.

### 2. Project Development Service Accounts

The following AI services are recommended for project development:

* **Gemini API:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (Google AI Key for Gemini)
* **SerpAPI:** [https://serpapi.com/manage-api-key](https://serpapi.com/manage-api-key) (Google Search API)
* **Tavily:** [https://app.tavily.com/home](https://app.tavily.com/home) (Web Search)
* **Weights & Biases (wandb):** [https://docs.wandb.ai/quickstart/](https://docs.wandb.ai/quickstart/) (Model training & fine-tuning monitoring/logging tool)
* **LangSmith:** [https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key) (LangChain logging & debug tool)
* **Visily AI:** [https://app.visily.ai](https://app.visily.ai) (AI planning tool)

### 3. Colab Setup

1.  Open `colab-env.ipynb` from the following link in Google Colab: [https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/1-1.prepare](https://github.com/mac999/LLM-RAG-Agent-Tutorial/tree/main/1-1.prepare)
2.  Connect to your Google Drive to save practice files.
3.  Set up the API keys you created earlier in the "Secrets" menu of your Colab account, as shown in the provided image.

### 4. Development Tools Installation

It is recommended to install these tools before the hands-on sessions to save time. Please install stable versions, as the latest versions may cause package installation errors. Ensure you check the "Add to PATH" option during installation if available.

* **Python (Recommended 3.11):** [https://www.python.org](https://www.python.org)
    * For Mac users, refer to the Python installation guide: [https://www.youtube.com](https://www.youtube.com)
    * Verify installation by running `python --version` in the terminal.
* **NVIDIA Driver (for NVIDIA GPU users):** [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
    * Skip GPU-related steps if you do not have an NVIDIA GPU.
    * Verify installation by running `nvidia-smi` in the terminal.
* **CUDA Toolkit (for NVIDIA GPU users):** [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
    * Check GPU and driver compatibility during installation.
    * Add CUDA path to environment variables.
    * Verify CUDA version by running `nvcc -V` in the terminal.
* **GitHub Tools:** [https://docs.github.com/ko/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop](https://docs.github.com/ko/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop)
* **Anaconda (Recommended 24.0 or higher):** [https://docs.anaconda.com/anaconda/install/](https://docs.anaconda.com/anaconda/install/)
* **PyTorch Library:** Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install the CPU version or the GPU version compatible with your CUDA driver.
* **Python Packages (using Anaconda virtual environment):**
    * Create a conda virtual environment named `venv_lmm` with Python 3.11:
        ```bash
        conda create --name venv_lmm python=3.11
        ```
    * Activate the environment:
        ```bash
        conda activate venv_lmm
        ```
    * Install core packages:
        ```bash
        pip install pandas numpy
        pip install ollama openai transformers huggingface_hub langchain
        ```
* **Docker (Optional):** Required for container-based operations. Visit [https://www.docker.com/get-started/](https://www.docker.com/get-started/) to install.
* **Ollama:** Required for local LLM AI tools. Visit [https://www.ollama.com/](https://www.ollama.com/) to install.
* **Code Editors & IDEs:**
    * **Sublime Text:** [https://www.sublimetext.com/](https://www.sublimetext.com/)
    * **Visual Studio Code (VS Code):** [https://code.visualstudio.com/download](https://code.visualstudio.com/download)
        * Refer to the provided video for detailed installation and Python extension setup.
        * After VS Code installation, install GitHub Copilot and GitHub Copilot Chat. Refer to the provided videos for guidance.
* **Claude Desktop:** [https://claude.ai/download](https://claude.ai/download)

### 5. Other Tools (Optional)

Install these if time permits:

* **Blender:** For LLM-based graphic modeling. Visit [https://www.blender.org/download/](https://www.blender.org/download/) to install.
* **DaVinci Resolve 20 Public Beta:** [https://www.blackmagicdesign.com/products/davinciresolve](https://www.blackmagicdesign.com/products/davinciresolve) (Optional)

## Usage

Once the development environment is set up, you can navigate through the Jupyter notebooks (.ipynb files) within the repository's folders (e.g., `1_AX_trend`, `2_ML_basic`, `3_DL_foundation`, etc.) to explore various AI topics and hands-on examples.

# Author
Taewook Kang (laputa99999@gmail.com)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For inquiries, please send me email (laputa99999@gmail.com) or refer to the project's GitHub page.
