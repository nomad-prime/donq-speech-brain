# System Prompt Configuration

This application supports a flexible system prompt configuration similar to how `.env` files work.

## Priority Order (highest to lowest)

1. **`.system_prompt.txt`** - Hidden local file (🔒 local, not committed)
2. **`system_prompt.local.txt`** - Local override (🔒 local, not committed)  
3. **`system_prompt.txt`** - Standard local file (🔒 local, not committed)
4. **`system_prompt.default.txt`** - Default/shared prompt (📋 default, committed to repo)
5. **`system.txt`** - Alternative name (📋 default, committed to repo)
6. **`prompt.txt`** - Alternative name (📋 default, committed to repo)

## Usage

### For Personal/Local Use
Create any of these files (they won't be committed to git):
```bash
# Recommended for personal use
echo "Your personal system prompt here" > .system_prompt.txt

# Alternative
echo "Your personal system prompt here" > system_prompt.local.txt
```

### For Team/Default Use
Edit the committed default file:
```bash
# This will be shared with everyone using the repo
echo "Team default system prompt" > system_prompt.default.txt
```

## Current Status

- ✅ **`system_prompt.default.txt`** - Contains the default voice interaction prompt
- ✅ **`.system_prompt.txt`** - Your personal local prompt (if it exists)
- ✅ **`.gitignore`** - Configured to ignore local prompt files

## Example Personal Prompt

```txt
You are my personal AI assistant. Respond in a casual, friendly tone. 
When I ask about work topics, assume I'm working on software development.
Keep responses concise but feel free to add personality to your answers.
```