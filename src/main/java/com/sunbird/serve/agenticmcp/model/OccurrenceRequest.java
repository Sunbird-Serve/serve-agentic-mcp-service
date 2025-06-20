package com.sunbird.serve.agenticmcp.model;

import com.fasterxml.jackson.annotation.JsonProperty;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.List;

@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class OccurrenceRequest {

    private Instant startDate;
    private Instant endDate;
    private String days;
    private String frequency;
    @JsonProperty("timeSlots")
    private List<TimeSlotRequest> timeSlots;
}
