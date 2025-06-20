package com.sunbird.serve.agenticmcp.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.sunbird.serve.agenticmcp.client.ClaudeClient;
import com.sunbird.serve.agenticmcp.client.RaiseNeedToolHandler;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.regex.Pattern;

@Service
public class ClaudeAgentService {
    private static final Logger logger = LoggerFactory.getLogger(ClaudeAgentService.class);

    @Autowired
    private ClaudeClient claudeClient;

    @Autowired
    private RaiseNeedToolHandler raiseNeedToolHandler;

    private String cleanClaudeMessage(String message) {
        if (message == null) return null;
        // Remove <thinking>...</thinking> or <thinking>... (if not closed)
        return message.replaceAll("(?s)<thinking>.*?</thinking>", "")
                      .replaceAll("(?s)<thinking>.*", "")
                      .trim();
    }

    private boolean isSummaryInsteadOfToolUse(String message) {
        if (message == null) return false;
        String lower = message.toLowerCase();
        return lower.contains("all required fields are present") ||
               lower.contains("i can proceed with calling the raiseneed tool") ||
               lower.contains("i can proceed with calling the tool");
    }

    public Map<String, Object> handlePrompt(String prompt) throws Exception {
        logger.info("Processing prompt: {}", prompt);

        Map<String, Object> response = claudeClient.sendPromptWithMcp(prompt);
        logger.debug("Received response from Claude: {}", response);

        Map<String, Object> result = new HashMap<>();

        if ("tool_use".equals(response.get("type"))) {
            logger.info("Calling raiseNeed tool handler with input: {}", response.get("input"));
            String toolResponse = raiseNeedToolHandler.handle((Map<String, Object>) response.get("input"));
            result.put("message", toolResponse);
            result.put("toolUse", true);
        } else if ("text".equals(response.get("type"))) {
            String message = cleanClaudeMessage((String) response.get("message"));
            logger.info("Returning Claude follow-up message to user: {}", message);
            // Fallback: If message is a summary/confirmation, re-prompt Claude
            if (isSummaryInsteadOfToolUse(message)) {
                logger.info("Detected summary/confirmation instead of tool_use. Re-prompting Claude for tool_use block only.");
                String followupPrompt = "Please respond only with the tool_use block for the raiseNeed tool using the information you just summarized. Do not include any explanation or summary.";
                Map<String, Object> followupResponse = claudeClient.sendPromptWithMcp(followupPrompt);
                if ("tool_use".equals(followupResponse.get("type"))) {
                    logger.info("Calling raiseNeed tool handler with input: {}", followupResponse.get("input"));
                    String toolResponse = raiseNeedToolHandler.handle((Map<String, Object>) followupResponse.get("input"));
                    result.put("message", toolResponse);
                    result.put("toolUse", true);
                } else {
                    // If still not tool_use, return the follow-up message
                    String followupMsg = cleanClaudeMessage((String) followupResponse.get("message"));
                    result.put("message", followupMsg);
                    result.put("toolUse", false);
                }
            } else {
                result.put("message", message);
                result.put("toolUse", false);
            }
        } else {
            result.put("message", "No actionable response from Claude.");
            result.put("toolUse", false);
        }

        return result;
    }
}
