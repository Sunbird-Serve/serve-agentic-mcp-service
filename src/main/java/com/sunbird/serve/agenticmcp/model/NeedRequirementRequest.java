package com.sunbird.serve.agenticmcp.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class NeedRequirementRequest {

    private String skillDetails;
    private String volunteersRequired;
    private OccurrenceRequest occurrence;
    private String priority;
}
