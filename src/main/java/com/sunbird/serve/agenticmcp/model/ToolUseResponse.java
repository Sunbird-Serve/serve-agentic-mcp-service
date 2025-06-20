package com.sunbird.serve.agenticmcp.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public class ToolUseResponse {
    @JsonProperty("type")
    private String type;

    @JsonProperty("tool_use")
    private ToolUse toolUse;

    public static class ToolUse {
        @JsonProperty("id")
        private String id;

        @JsonProperty("name")
        private String name;

        @JsonProperty("input")
        private Map<String, Object> input;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public Map<String, Object> getInput() {
            return input;
        }

        public void setInput(Map<String, Object> input) {
            this.input = input;
        }
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public ToolUse getToolUse() {
        return toolUse;
    }

    public void setToolUse(ToolUse toolUse) {
        this.toolUse = toolUse;
    }
} 