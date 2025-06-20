package com.sunbird.serve.agenticmcp.model;
import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class RaiseNeedRequest {
    @JsonProperty("needRequest")
    private NeedRequest needRequest;
    @JsonProperty("needRequirementRequest")
    private NeedRequirementRequest needRequirementRequest;
}
