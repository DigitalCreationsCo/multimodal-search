# Add multimodal_search to agent configuration file

For example, in claude_desktop_config.json:

json```
{
  "mcpServers": {
    "multimodal_search": {
      "command": "python",
      "args": [
        "/absolute/path/to/multimodal_search/mcp_server.py"
      ]
    }
  }
}
```
