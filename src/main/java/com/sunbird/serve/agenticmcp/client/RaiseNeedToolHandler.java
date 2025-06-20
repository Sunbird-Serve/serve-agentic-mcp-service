package com.sunbird.serve.agenticmcp.client;

import com.sunbird.serve.agenticmcp.model.*;
import org.springframework.stereotype.Component;
import org.springframework.http.*;
import org.springframework.web.client.RestTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.time.Instant;
import java.util.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.fasterxml.jackson.databind.SerializationFeature;

@Component
public class RaiseNeedToolHandler {
    private static final Logger logger = LoggerFactory.getLogger(RaiseNeedToolHandler.class);
    private final RestTemplate restTemplate = new RestTemplate();
    private static final String BACKEND_URL = "https://serve-v1.evean.net/api/v1/serve-need/need/raise";

    public String handle(Map<String, Object> input) {
        try {
            logger.info("Processing raise need request with input: {}", input);

            // Extract time slots from input
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> timeSlots = (List<Map<String, Object>>) input.get("timeSlots");
            if (timeSlots == null || timeSlots.isEmpty()) {
                logger.error("No time slots provided in input");
                return "❌ Error: No time slots provided";
            }

            // Convert time slots to TimeSlotRequest objects
            List<TimeSlotRequest> slotRequests = timeSlots.stream()
                .map(slot -> TimeSlotRequest.builder()
                    .day((String) slot.get("day"))
                    .startTime(Instant.parse((String) slot.get("startTime")))
                    .endTime(Instant.parse((String) slot.get("endTime")))
                    .build())
                .toList();

            logger.debug("Converted time slots: {}", slotRequests);

            List<String> days = (List<String>) input.get("days");
            String daysStr = String.join(",", days);

            // Create occurrence request
            OccurrenceRequest occurrence = OccurrenceRequest.builder()
                .startDate(Instant.parse((String) input.get("startDate")))
                .endDate(Instant.parse((String) input.get("endDate")))
                .days(daysStr)
                .frequency((String) input.getOrDefault("frequency", "WEEKLY"))
                .timeSlots(slotRequests)
                .build();

            logger.debug("Created occurrence request: {}", occurrence);

            // Create need requirement request
            NeedRequirementRequest needReq = NeedRequirementRequest.builder()
                .skillDetails("Teaching - " + input.get("subject"))
                .volunteersRequired("1")
                .occurrence(occurrence)
                .priority("medium")
                .build();

            logger.debug("Created need requirement request: {}", needReq);

            // Create need request
            NeedRequest need = NeedRequest.builder()
                .needTypeId("e916a99a-554d-44a6-a714-44d227849ac0")
                .name("Volunteer Teaching Need - " + input.get("subject") + " Grade " + input.get("grade"))
                .description("Need raised by agent")
                .status(NeedStatus.New)
                .userId("1-adc2c7e3-8514-4486-b6f1-4713a0be2539")
                .entityId("3540ae2b-6886-4961-870c-2dbd920c2c54")
                .requirementId(null)
                .build();

            logger.debug("Created need request: {}", need);

            // Build final raise need request
            RaiseNeedRequest request = RaiseNeedRequest.builder()
                .needRequest(need)
                .needRequirementRequest(needReq)
                .build();

            // Configure ObjectMapper for proper date handling
            ObjectMapper mapper = new ObjectMapper();
            mapper.registerModule(new JavaTimeModule());
            mapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
            
            // Convert request to JSON string
            String json = mapper.writeValueAsString(request);
            logger.info("Final JSON payload: {}", json);

            // Call backend
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("x-request-source", "agentic-mcp");
            
            // Create request entity with JSON string
            HttpEntity<String> entity = new HttpEntity<>(json, headers);
            
            ResponseEntity<String> response = restTemplate.postForEntity(
                BACKEND_URL,
                entity,
                String.class
            );

            logger.info("Received response from backend: {}", response.getBody());

            if (response.getStatusCode() == HttpStatus.OK || response.getStatusCode() == HttpStatus.CREATED) {
                if (response.getStatusCode() == HttpStatus.CREATED) {
                    return "✅ Your need is created successfully!";
                } else {
                    return "✅ Need Raised: " + response.getBody();
                }
            } else {
                logger.error("Backend returned error status: {}", response.getStatusCode());
                return "❌ Error: Backend returned status " + response.getStatusCode();
            }
        } catch (Exception e) {
            logger.error("Error raising need", e);
            return "❌ Error: " + e.getMessage();
        }
    }
}
