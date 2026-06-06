# Add semantic-multimodal to agent configuration file

For example, in claude_desktop_config.json:

json```
{
  "mcpServers": {
    "semantic-multimodal": {
      "command": "python",
      "args": [
        "/absolute/path/to/semantic-mulitmodal/mcp_server.py"
      ]
    }
  }
}
```