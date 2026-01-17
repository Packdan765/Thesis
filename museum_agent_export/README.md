# Museum Dialogue Agent

A trained reinforcement learning agent that acts as a museum guide. The agent selects dialogue actions using a learned policy and generates natural language responses using an LLM.

## 🚀 Quick Setup (3 Steps)

1. **Install:** `pip install -r requirements.txt`
2. **Set API Key:** Choose one method below ⬇️
3. **Run:** `python run_agent.py`

## 📝 Setting Your API Key

**EASIEST METHOD:** Create a `.env` file in this directory:

```bash
# Copy the example
cp env.example .env

# Edit .env and add your key:
GROQ_API_KEY=your-actual-key-here
```

**OR** set environment variable in your terminal (see full instructions below).

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For OpenAI or Anthropic, install additional packages:
```bash
pip install openai      # For OpenAI
pip install anthropic   # For Anthropic
```

### 2. Configure API Key

**You need to set your API key before running the agent.** Choose one method below:

#### Method 1: Environment Variables

**⚠️ IMPORTANT: Use the correct syntax for your terminal!**

**Windows PowerShell (PS> prompt):**
```powershell
# For Groq (default) - Use $env: syntax in PowerShell!
$env:GROQ_API_KEY="your-key-here"

# Verify it's set:
echo $env:GROQ_API_KEY

# For OpenAI
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your-key-here"

# For Anthropic
$env:LLM_PROVIDER="anthropic"
$env:ANTHROPIC_API_KEY="your-key-here"
```

**Windows Command Prompt (C:\> prompt):**
```cmd
REM For Groq (default) - Use set command in CMD
set GROQ_API_KEY=your-key-here

REM Verify it's set:
echo %GROQ_API_KEY%

REM For OpenAI
set LLM_PROVIDER=openai
set OPENAI_API_KEY=your-key-here

REM For Anthropic
set LLM_PROVIDER=anthropic
set ANTHROPIC_API_KEY=your-key-here
```

**How to tell which terminal you're using:**
- **PowerShell:** Prompt shows `PS C:\...>` - Use `$env:VARIABLE="value"`
- **Command Prompt:** Prompt shows `C:\...>` - Use `set VARIABLE=value`

**Linux/Mac:**
```bash
# For Groq (default)
export GROQ_API_KEY="your-key-here"

# For OpenAI
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-key-here"

# For Anthropic
export LLM_PROVIDER="anthropic"
export ANTHROPIC_API_KEY="your-key-here"
```

**Note:** Environment variables set this way only last for the current terminal session. To make them permanent, add them to your shell profile (`.bashrc`, `.zshrc`, etc.).

#### Method 2: Using .env File (Easiest!)

1. **Copy the example file:**
   ```bash
   # Windows
   copy env.example .env
   
   # Linux/Mac
   cp env.example .env
   ```

2. **Edit `.env` file** (open it in any text editor) and replace `your_groq_api_key_here` with your actual key:
   ```bash
   # For Groq (default) - just uncomment and fill in:
   GROQ_API_KEY=sk-groq-your-actual-key-here
   
   # OR for OpenAI - uncomment these two lines:
   # LLM_PROVIDER=openai
   # OPENAI_API_KEY=sk-your-actual-key-here
   
   # OR for Anthropic - uncomment these two lines:
   # LLM_PROVIDER=anthropic
   # ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
   ```

3. **Save the file** - The agent will automatically load it when you run it!

**Example `.env` file for Groq:**
```
GROQ_API_KEY=sk-groq-abc123xyz789...
```

**Example `.env` file for OpenAI:**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-abc123xyz789...
```

#### Get Your API Keys

- **Groq (Free tier available):** https://console.groq.com/
- **OpenAI:** https://platform.openai.com/api-keys
- **Anthropic:** https://console.anthropic.com/

#### Verify Your API Key is Set

**If using environment variables:**
```bash
# Windows PowerShell
echo $env:GROQ_API_KEY

# Windows CMD
echo %GROQ_API_KEY%

# Linux/Mac
echo $GROQ_API_KEY
```

**If using .env file:**
Just run the agent - it will automatically load from `.env`. If you see an error about missing API key, check that:
1. Your `.env` file is in the same directory as `run_agent.py`
2. The key is on a line starting with `GROQ_API_KEY=` (no spaces around `=`)
3. The line is not commented out (doesn't start with `#`)

**Quick Test:**
```bash
# Check if API key is configured
python check_api_key.py

# Or try listing models
python run_agent.py --list-models
```

If these work without errors, your API key is configured correctly!

### 3. Run the Agent

#### Command Line Interface

```bash
# Use default model
python run_agent.py

# Use specific model
python run_agent.py --model models/H2_SMDP_Augmented.pt

# List available models
python run_agent.py --list-models
```

**CLI Commands:**
- `exhibits` - List all exhibits
- `exhibit N` - Switch to exhibit number N (1-5)
- `reset` - Reset conversation
- `quit` or `exit` - Exit

#### Python API

```python
from agent_api import MuseumAgent

# Initialize agent
agent = MuseumAgent()

# Get response
result = agent.respond(
    user_message="What is this painting?",
    exhibit="King_Caspar"
)

print(result["response"])
```

See `example_usage.py` for more examples.

## API Reference

### MuseumAgent Class

```python
from agent_api import MuseumAgent

# Initialize
agent = MuseumAgent(model_path="models/H2_MDP_Augmented.pt")
```

**Parameters:**
- `model_path` (str, optional): Path to model checkpoint. Default: `"models/H2_MDP_Augmented.pt"`

**Methods:**

#### `respond(user_message, exhibit=None)`

Get agent response to user message.

**Parameters:**
- `user_message` (str, required): What the visitor said
- `exhibit` (str, optional): Which exhibit they're at. If not provided, uses current exhibit.

**Returns:**
```python
{
    "response": str,        # Agent's natural language response
    "action": str,          # Selected action (e.g., "Explain/ExplainNewFact")
    "exhibit": str,         # Current exhibit name
    "facts_remaining": int  # Number of facts left to share
}
```

**Example:**
```python
result = agent.respond("Tell me about this", exhibit="Turban")
print(result["response"])  # "This painting shows..."
```

#### `reset()`

Reset conversation to start fresh.

```python
agent.reset()
```

#### `get_exhibits()`

Get list of available exhibit names.

```python
exhibits = agent.get_exhibits()
# Returns: ["King_Caspar", "Turban", "Dom_Miguel", "Pedro_Sunda", "Diego_Bemba"]
```

## LLM Providers

The agent supports multiple LLM providers. Configure via environment variables:

| Provider | Required Env Vars | Default Model | Notes |
|----------|-------------------|---------------|-------|
| **Groq** | `GROQ_API_KEY` | `llama-3.1-8b` | Free tier, fast inference |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY` | `gpt-3.5-turbo` | Paid, high quality |
| **Anthropic** | `LLM_PROVIDER=anthropic`<br>`ANTHROPIC_API_KEY` | `claude-3-haiku-20240307` | Paid, excellent quality |

**Model Selection:**

Set `LLM_MODEL` environment variable to use a different model:

```bash
# Use GPT-4
export LLM_MODEL="gpt-4"

# Use Claude Sonnet
export LLM_MODEL="claude-3-sonnet-20240229"

# Use Llama 3.3
export LLM_MODEL="llama-3.3"
```

**Python Configuration:**

```python
import os

# Use OpenAI GPT-4
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "gpt-4"
os.environ["OPENAI_API_KEY"] = "your-key"

from agent_api import MuseumAgent
agent = MuseumAgent()
```

## Available Models

| Model File | Type | Description |
|------------|------|-------------|
| `H1_MDP_Baseline.pt` | Flat (MDP) | Basic reward function |
| `H2_MDP_Augmented.pt` | Flat (MDP) | Augmented reward (recommended for flat) |
| `H1_SMDP_Baseline.pt` | Hierarchical (SMDP) | Basic reward function |
| `H2_SMDP_Augmented.pt` | Hierarchical (SMDP) | Augmented reward (recommended for HRL) |

**Model Selection:**

```python
# Use H2 mdp model, peforms the best so far.
agent = MuseumAgent(model_path="models/H2_MDP_Augmented.pt")
```

## Exhibits

The agent knows about 5 museum exhibits:

- **King_Caspar** - Painting of one of the three magi
- **Turban** - Portrait of a boy in oriental attire
- **Dom_Miguel** - Colonial-era portrait
- **Pedro_Sunda** - Historical figure portrait
- **Diego_Bemba** - Historical figure portrait

## Command Line Usage

### Basic Commands

```bash
# Run with default model
python run_agent.py

# Specify model
python run_agent.py --model models/H2_SMDP_Augmented.pt

# List available models
python run_agent.py --list-models
```

### Interactive Commands

Once in the chat interface:

- Type a message to chat with the agent
- `exhibits` - Show all exhibits and fact coverage
- `exhibit N` - Switch to exhibit number N (1-5)
- `exhibit <name>` - Switch to exhibit by name (e.g., `exhibit Turban`)
- `reset` - Start a new conversation
- `quit` or `exit` - Exit the program

## Project Structure

```
museum_agent_export/
├── agent_api.py          # Python API (main interface)
├── run_agent.py         # Command-line interface
├── example_usage.py      # Usage examples
├── requirements.txt      # Python dependencies
├── env.example          # Environment variable template
├── README.md            # This file
├── models/              # Trained model checkpoints (.pt files)
│   ├── H1_MDP_Baseline.pt
│   ├── H2_MDP_Augmented.pt
│   ├── H1_SMDP_Baseline.pt
│   └── H2_SMDP_Augmented.pt
├── data/                # Knowledge base
│   └── museum_knowledge_graph.json
└── src/                 # Source code
    ├── agent/           # Agent implementations
    └── utils/           # Utilities (LLM, DialogueBERT, etc.)
```

## Examples

### Basic Usage

```python
from agent_api import MuseumAgent

agent = MuseumAgent()
result = agent.respond("What is this painting?", exhibit="King_Caspar")
print(result["response"])
```

### Multi-turn Conversation

```python
agent = MuseumAgent()

# First message
result1 = agent.respond("Hello!", exhibit="King_Caspar")

# Follow-up
result2 = agent.respond("Tell me more", exhibit="King_Caspar")

# Switch exhibits
result3 = agent.respond("What about this one?", exhibit="Turban")
```

### Using Different Models

```python
# Flat model
flat_agent = MuseumAgent(model_path="models/H2_MDP_Augmented.pt")

# Hierarchical model
hrl_agent = MuseumAgent(model_path="models/H2_SMDP_Augmented.pt")
```

## Troubleshooting

### "GROQ_API_KEY environment variable not set"

**If using PowerShell:**
```powershell
# Make sure you use $env: syntax (NOT "set")
$env:GROQ_API_KEY="your-key-here"

# Verify it worked:
echo $env:GROQ_API_KEY
```

**If using Command Prompt:**
```cmd
REM Use "set" command (NOT $env:)
set GROQ_API_KEY=your-key-here

REM Verify it worked:
echo %GROQ_API_KEY%
```

**Common mistakes:**
- ❌ Using `set` in PowerShell (use `$env:` instead)
- ❌ Using `$env:` in CMD (use `set` instead)
- ❌ Forgetting quotes around the key value
- ✅ **Easiest fix:** Use `.env` file method (see Method 2 above)

### "Model not found"
- Check that model file exists in `models/` directory
- Use `python run_agent.py --list-models` to see available models
- Make sure you're running from the `museum_agent_export` directory

### "package not installed"
- Install required package: `pip install groq` (or `openai`, `anthropic`)
- Make sure you ran `pip install -r requirements.txt` first

### "Invalid API key"
- Verify your API key is correct (no extra spaces)
- Check that you've set the correct environment variable for your provider
- Try using the `.env` file method instead

### Still having issues?

**Quick test - Create a `.env` file:**
1. In the `museum_agent_export` folder, create a file named `.env`
2. Add this line: `GROQ_API_KEY=your-actual-key-here`
3. Save and run: `python run_agent.py`

This method works automatically and doesn't require setting environment variables!

## License

[Add your license information here]
