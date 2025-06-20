package com.sunbird.serve.agenticmcp.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.sunbird.serve.agenticmcp.service.ClaudeAgentService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

@RestController
@RequestMapping("/agent")
@Tag(name = "Claude Agent", description = "APIs for interacting with Claude AI agent")
public class ClaudeAgentController {
    private static final Logger logger = LoggerFactory.getLogger(ClaudeAgentController.class);

    @Autowired
    private ClaudeAgentService claudeService;

    @Operation(
        summary = "Process user prompt with Claude AI",
        description = "Takes a user prompt and processes it using Claude AI to extract structured information",
        responses = {
            @ApiResponse(
                responseCode = "200",
                description = "Successfully processed the prompt",
                content = @Content(mediaType = "application/json")
            ),
            @ApiResponse(
                responseCode = "400",
                description = "Invalid request payload"
            ),
            @ApiResponse(
                responseCode = "500",
                description = "Internal server error"
            )
        }
    )
    @PostMapping("/agent-raise-need")
public ResponseEntity<Map<String, Object>> raiseNeed(
    @Parameter(description = "Request containing the user prompt", required = true)
    @RequestBody Map<String, String> request
) {
    try {
        String userPrompt = request.get("prompt");
        if (userPrompt == null || userPrompt.trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of(
                "message", "Prompt cannot be empty",
                "toolUse", false
            ));
        }

        Map<String, Object> response = claudeService.handlePrompt(userPrompt);
        return ResponseEntity.ok(response);

    } catch (Exception e) {
        logger.error("Error processing prompt", e);
        return ResponseEntity.internalServerError().body(Map.of(
            "message", "Error processing prompt: " + e.getMessage(),
            "toolUse", false
        ));
    }
}

}
