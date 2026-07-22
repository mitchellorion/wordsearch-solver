# Automated Puzzle Solver Bot

This repository contains an autonomous Python automation tool designed to interact with and solve grid-based mobile puzzle games via ADB (Android Debug Bridge). It leverages visual LLMs (like GPT-5.6) to analyze the game board and orchestrate precise swiping and clicking interactions.

## Features
- **Zero-Touch Automation:** Automatically detects the board, solves the puzzle, and progresses to the next level.
- **LLM-Powered Vision:** Uses OpenAI's GPT models (Terra or Sol) for advanced OCR, grid mapping, and coordinate extraction.
- **Dynamic Image Scaling:** Auto-compresses high-resolution screenshots to stay within strict token limits while maintaining OCR accuracy.
- **Intelligent Recovery:** Detects when grid extraction fails and automatically forces a restart of the target app.
- **Autonomous Agent included:** Features an integrated `sol_agent.py` script powered by the `/v1/responses` deep reasoning API. You can launch this agent to autonomously optimize the codebase or interact with the file system.

## Prerequisites
- Python 3.10+
- An active OpenAI API Key configured in your environment (`.env`).
- Android Debug Bridge (`adb`) installed on your system.
- An Android device or emulator (e.g., BlueStacks, Genymotion, or physical device) connected via USB or network with **USB Debugging** enabled.

## Installation
1. Clone this repository to your local machine.
2. Install the required Python dependencies:
   ```bash
   pip install openai pillow python-dotenv httpx numpy
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

## Usage

### 1. The Solver Bot
To launch the main automation loop, ensure your Android device/emulator is active on the puzzle screen and run:
```bash
python bot.py
```

**Optional Arguments:**
- `--force`: Forces the interactive calibration prompt on startup, allowing you to manually define the board bounding box.
- `--model <terra|sol>`: Selects which GPT-5.6 model to use for puzzle solving. Defaults to `terra` for cost efficiency.

### 2. The Autonomous Agent (Sol)
To launch the local coding assistant for maintenance or optimization tasks:
```bash
python sol_agent.py
```
This boots up an interactive terminal where you can chat with the `gpt-5.6-sol` model. It has full tool-calling capabilities to read your files, edit code, and run terminal commands.

## Architecture Notes
- The bot relies heavily on relative coordinate mapping. It takes full-screen screenshots, crops to the active board area, and calculates relative drag gestures for the device resolution.
- Ensure your emulator or device is not heavily letterboxed, as this can offset the ADB coordinate taps.
