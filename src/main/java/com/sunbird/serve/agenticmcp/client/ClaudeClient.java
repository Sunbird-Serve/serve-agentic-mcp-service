package com.sunbird.serve.agenticmcp.client;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
public class ClaudeClient {
    private static final Logger logger = LoggerFactory.getLogger(ClaudeClient.class);

    @Value("${claude.api.key}")
    private String apiKey;

    @Value("${claude.api.url}")
    private String apiUrl;

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper mapper = new ObjectMapper();

    public Map<String, Object> sendPromptWithMcp(String prompt) throws Exception {
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("Claude API key is not configured");
        }
        if (!apiKey.startsWith("sk-ant-")) {
            throw new IllegalStateException("Invalid Claude API key format");
        }

        // Load MCP manifest (YAML)
        InputStream input = getClass().getResourceAsStream("/mcp_manifest.yaml");
        if (input == null) {
            throw new IllegalStateException("MCP manifest file not found");
        }
        Map<String, Object> manifest = new ObjectMapper(new YAMLFactory()).readValue(input, Map.class);

        // Prepare request
        Map<String, Object> body = new HashMap<>();
        body.put("model", "claude-3-opus-20240229");
        body.put("max_tokens", 1000);
        body.put("tools", manifest.get("tools"));
        body.put("system", 
            "You are a helpful AI assistant that supports schools in raising requests for volunteer teachers via the raiseNeed tool. " +
            "When a user describes their requirement, first check if all necessary fields are provided: grade, subject, startDate, endDate, days, and timeSlots. " +
            "If any information is missing, respond in a clear, natural, and friendly tone asking only for the missing pieces. " +
            "Use simple, conversational language. " +
            "If all required fields are present, do not summarize or confirm in text. Only respond with the tool_use block for the raiseNeed tool, and nothing else. " +
            "Only respond with `tool_use` once all required fields are available. Add Hari Bol at the beginning to greet the user."
        );

        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(Map.of("role", "user", "content", prompt));
        body.put("messages", messages);

        // Send request
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("x-api-key", apiKey);
        headers.set("anthropic-version", "2023-06-01");

        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(body, headers);
        
        logger.debug("Sending request to Claude API with headers: {}", headers);
        logger.debug("Request body: {}", body);

        try {
            ResponseEntity<String> response = restTemplate.postForEntity(apiUrl, entity, String.class);
            JsonNode json = mapper.readTree(response.getBody());
            logger.debug("Claude API response: {}", json);

            JsonNode content = json.get("content");
            Map<String, Object> toolUseResult = null;
            String followup = null;
            if (content != null && content.isArray()) {
                for (JsonNode item : content) {
                    if (item.has("type") && "tool_use".equals(item.get("type").asText())) {
                        JsonNode inputNode = item.get("input");
                        if (inputNode != null && inputNode.isObject()) {
                            Map<String, Object> inputMap = mapper.convertValue(inputNode, Map.class);
                            logger.info("Extracted tool_use input: {}", inputMap);
                            toolUseResult = Map.of("type", "tool_use", "input", inputMap);
                        }
                    } else if (item.has("type") && "text".equals(item.get("type").asText())) {
                        followup = item.get("text").asText();
                        logger.info("Claude follow-up message: {}", followup);
                    }
                }
            }
            if (toolUseResult != null) {
                return toolUseResult;
            } else if (followup != null) {
                return Map.of("type", "text", "message", followup);
            }
            throw new RuntimeException("Claude response did not include tool_use or text. Response: " + json);
        } catch (HttpClientErrorException.Unauthorized e) {
            logger.error("Authentication failed with Claude API. Please check your API key.");
            throw new RuntimeException("Authentication failed with Claude API: " + e.getMessage(), e);
        } catch (Exception e) {
            logger.error("Error calling Claude API", e);
            throw new RuntimeException("Failed to get response from Claude: " + e.getMessage(), e);
        }
    }
}
